#include "NVEncoder.hpp"
#define NOMINMAX
#include <algorithm>
#include <iostream>
#include <vector>

static std::string headGUID(const GUID &g) {
    const uint8_t *p = reinterpret_cast<const uint8_t *>(&g);
    char buf[64];
#ifdef _DEBUG
    sprintf(buf, "%02X%02X%02X%02X-%02X%02X%02X%02X", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
#endif
    return std::string(buf);
}

bool NVENCEncoder::init(CUcontext cuCtx, int w, int h, int fps_num, int fps_den) {
    width = w;
    height = h;
    fpsn = fps_num;
    fpsd = fps_den ? fps_den : 1;
    gopLen = (std::max) ((2 * fpsn) / fpsd, 1);// старт: ~2 сек, потом можно перезадать через reconfigure

    NVENCSTATUS st = NvEncodeAPICreateInstance(&f);
    if (st != NV_ENC_SUCCESS) {
        std::cerr << "NvEncodeAPICreateInstance failed: " << st << "\n";
        return false;
    }

    uint32_t maxApi = 0;
    if (NvEncodeAPIGetMaxSupportedVersion(&maxApi) == NV_ENC_SUCCESS) {
        std::cout << "NVENC: Header API=" << NVENCAPI_VERSION << " Driver API=" << maxApi << "\n";
        if (NVENCAPI_VERSION > maxApi) {
            std::cerr << "NVENC: Driver too old for headers. Update NVIDIA driver.\n";
            return false;
        }
    }

    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open = {NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER};
    open.device = cuCtx;
    open.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    open.apiVersion = NVENCAPI_VERSION;
    st = f.nvEncOpenEncodeSessionEx(&open, &enc);
    if (st != NV_ENC_SUCCESS) {
        std::cerr << "nvEncOpenEncodeSessionEx failed: " << st << "\n";
        return false;
    }

    uint32_t count = 0, got = 0;
    st = f.nvEncGetEncodePresetCount(enc, NV_ENC_CODEC_H264_GUID, &count);
    std::cout << "nvEncGetEncodePresetCount(H264) st=" << st << " count=" << count << "\n";
    if (st != NV_ENC_SUCCESS || count == 0) return false;

    std::vector<GUID> presets(count);
    st = f.nvEncGetEncodePresetGUIDs(enc, NV_ENC_CODEC_H264_GUID, presets.data(), count, &got);
    std::cout << "nvEncGetEncodePresetGUIDs st=" << st << " got=" << got << "\n";
    if (st != NV_ENC_SUCCESS || got == 0) return false;

    auto try_init = [&](const GUID &preset) -> NVENCSTATUS {
        NV_ENC_INITIALIZE_PARAMS ip = {NV_ENC_INITIALIZE_PARAMS_VER};
        ip.encodeGUID = NV_ENC_CODEC_H264_GUID;
        ip.presetGUID = preset;
        ip.tuningInfo = NV_ENC_TUNING_INFO_UNDEFINED;
        ip.encodeWidth = width;
        ip.encodeHeight = height;
        ip.darWidth = width;
        ip.darHeight = height;
        ip.frameRateNum = fpsn;
        ip.frameRateDen = fpsd;
        ip.maxEncodeWidth = width;
        ip.maxEncodeHeight = height;
        ip.enablePTD = 1;
        ip.encodeConfig = nullptr;
        NVENCSTATUS s = f.nvEncInitializeEncoder(enc, &ip);
        std::cout << "Init with preset head=" << headGUID(preset) << " st=" << s << "\n";
        return s;
    };

    if (try_init(presets[0]) == NV_ENC_SUCCESS) {
        std::cout << "NVENC initialized with presets[0]\n";
        return true;
    }
    for (uint32_t i = 1; i < got; ++i)
        if (try_init(presets[i]) == NV_ENC_SUCCESS) {
            std::cout << "NVENC initialized with presets[" << i << "]\n";
            return true;
        }
    GUID zero{};
    if (try_init(zero) == NV_ENC_SUCCESS) {
        std::cout << "NVENC initialized with zero preset GUID\n";
        return true;
    }

    std::cerr << "NVENC init failed: no preset GUID accepted\n";
    return false;
}

bool NVENCEncoder::encodeNV12(CUdeviceptr d_nv12, uint32_t pitch, std::vector<uint8_t> &outBitstream, bool &isKey, int frameIdx, bool forceIdrNow) {
    outBitstream.clear();
    isKey = false;

    NV_ENC_REGISTER_RESOURCE reg = {NV_ENC_REGISTER_RESOURCE_VER};
    reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
    reg.resourceToRegister = (void *) d_nv12;
    reg.width = width;
    reg.height = height;
    reg.pitch = pitch;
    reg.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
    NVENCSTATUS st = f.nvEncRegisterResource(enc, &reg);
    if (st != NV_ENC_SUCCESS) {
        std::cerr << "nvEncRegisterResource: " << st << "\n";
        return false;
    }

    NV_ENC_MAP_INPUT_RESOURCE map = {NV_ENC_MAP_INPUT_RESOURCE_VER};
    map.registeredResource = reg.registeredResource;
    st = f.nvEncMapInputResource(enc, &map);
    if (st != NV_ENC_SUCCESS) {
        std::cerr << "nvEncMapInputResource: " << st << "\n";
        f.nvEncUnregisterResource(enc, reg.registeredResource);
        return false;
    }

    NV_ENC_CREATE_BITSTREAM_BUFFER bs = {NV_ENC_CREATE_BITSTREAM_BUFFER_VER};
    st = f.nvEncCreateBitstreamBuffer(enc, &bs);
    if (st != NV_ENC_SUCCESS) {
        std::cerr << "nvEncCreateBitstreamBuffer: " << st << "\n";
        f.nvEncUnmapInputResource(enc, map.mappedResource);
        f.nvEncUnregisterResource(enc, reg.registeredResource);
        return false;
    }

    NV_ENC_PIC_PARAMS pic = {NV_ENC_PIC_PARAMS_VER};
    pic.inputBuffer = map.mappedResource;
    pic.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
    pic.inputWidth = width;
    pic.inputHeight = height;
    pic.outputBitstream = bs.bitstreamBuffer;
    pic.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;

    bool needIdr = false;
    if (forceIdrNow) {
        needIdr = true;
#ifdef _DEBUG
        std::cout << "[ENC] FORCE IDR due to dup chain\n";
#endif
    }
    if (forceIdrCount > 0) needIdr = true;
    if (gopLen > 0 && (frameIdx - lastKeyIdx) >= gopLen) needIdr = true;
    if (tinyCount >= 3) {
        needIdr = true;
        tinyCount = 0;
#ifdef _DEBUG
        std::cout << "[ENC] FORCE IDR due to tiny P chain\n";
#endif
    }

    if (needIdr) {
        pic.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
#ifdef _DEBUG
        if (forceIdrCount > 0) std::cout << "[ENC] FORCE IDR, left=" << forceIdrCount << " frameIdx=" << frameIdx << "\n";
#endif
    }

    st = f.nvEncEncodePicture(enc, &pic);
    if (st != NV_ENC_SUCCESS) {
        std::cerr << "nvEncEncodePicture: " << st << "\n";
        f.nvEncDestroyBitstreamBuffer(enc, bs.bitstreamBuffer);
        f.nvEncUnmapInputResource(enc, map.mappedResource);
        f.nvEncUnregisterResource(enc, reg.registeredResource);
        return false;
    }

    NV_ENC_LOCK_BITSTREAM lock = {NV_ENC_LOCK_BITSTREAM_VER};
    lock.outputBitstream = bs.bitstreamBuffer;
    st = f.nvEncLockBitstream(enc, &lock);
    if (st == NV_ENC_SUCCESS) {
        outBitstream.resize(lock.bitstreamSizeInBytes);
        memcpy(outBitstream.data(), lock.bitstreamBufferPtr, lock.bitstreamSizeInBytes);
        isKey = (lock.pictureType == NV_ENC_PIC_TYPE_IDR || lock.pictureType == NV_ENC_PIC_TYPE_I);

        auto *p = (uint8_t *) lock.bitstreamBufferPtr;
        int n = std::min<int>(lock.bitstreamSizeInBytes, 8);
#ifdef _DEBUG
        for (int i = 0; i < n; i++) std::cout << " " << std::hex << (int) p[i] << std::dec;
        std::cout << "[ENC] head:\n";
        std::cout << "[ENC] Bitstream size=" << lock.bitstreamSizeInBytes
                  << " picType=" << (int) lock.pictureType
                  << " key=" << (isKey ? 1 : 0) << "\n";
#endif
        if (!isKey && lock.bitstreamSizeInBytes < 200) tinyCount++;
        else
            tinyCount = 0;
        if (isKey) lastKeyIdx = frameIdx;
        if (forceIdrCount > 0) --forceIdrCount;

        f.nvEncUnlockBitstream(enc, bs.bitstreamBuffer);
    } else {
        std::cerr << "nvEncLockBitstream: " << st << "\n";
    }

    f.nvEncDestroyBitstreamBuffer(enc, bs.bitstreamBuffer);
    f.nvEncUnmapInputResource(enc, map.mappedResource);
    f.nvEncUnregisterResource(enc, reg.registeredResource);

    return !outBitstream.empty();
}

bool NVENCEncoder::reconfigureBitrate(int avg_bps, int max_bps, int gop_len) {
    if (!enc) return false;
    if (gop_len > 0) gopLen = gop_len;

    NV_ENC_RECONFIGURE_PARAMS rp = {NV_ENC_RECONFIGURE_PARAMS_VER};
    rp.reInitEncodeParams.version = NV_ENC_INITIALIZE_PARAMS_VER;

    NV_ENC_CONFIG cfg = {NV_ENC_CONFIG_VER};
    rp.reInitEncodeParams.encodeGUID = NV_ENC_CODEC_H264_GUID;
    rp.reInitEncodeParams.tuningInfo = NV_ENC_TUNING_INFO_UNDEFINED;
    rp.reInitEncodeParams.encodeWidth = width;
    rp.reInitEncodeParams.encodeHeight = height;
    rp.reInitEncodeParams.darWidth = width;
    rp.reInitEncodeParams.darHeight = height;
    rp.reInitEncodeParams.frameRateNum = fpsn;
    rp.reInitEncodeParams.frameRateDen = fpsd;
    rp.reInitEncodeParams.maxEncodeWidth = width;
    rp.reInitEncodeParams.maxEncodeHeight = height;
    rp.reInitEncodeParams.enablePTD = 1;
    rp.reInitEncodeParams.encodeConfig = &cfg;

    cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
    cfg.rcParams.averageBitRate = avg_bps;
    cfg.rcParams.maxBitRate = max_bps;
    cfg.rcParams.vbvBufferSize = max_bps;
    cfg.rcParams.vbvInitialDelay = max_bps / 2;

    cfg.gopLength = gopLen;// чаще ключи
    cfg.frameIntervalP = 1;// без B
    cfg.encodeCodecConfig.h264Config.idrPeriod = gopLen;
    cfg.encodeCodecConfig.h264Config.repeatSPSPPS = 1;

    rp.forceIDR = 1;

    NVENCSTATUS st = f.nvEncReconfigureEncoder(enc, &rp);
    if (st != NV_ENC_SUCCESS) {
        std::cerr << "nvEncReconfigureEncoder: " << st << "\n";
        return false;
    }
    return true;
}

void NVENCEncoder::destroy() {
    if (enc) {
        f.nvEncDestroyEncoder(enc);
        enc = nullptr;
    }
}
