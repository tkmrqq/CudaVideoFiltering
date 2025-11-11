#include "NVEncoder.hpp"
#include <algorithm>
#include <array>
#include <iostream>

static bool guidEquals(const GUID &a, const GUID &b) {
    return !memcmp(&a, &b, sizeof(GUID));
}

static bool containsGUID(const GUID *arr, uint32_t n, const GUID &g) {
    for (uint32_t i = 0; i < n; ++i)
        if (guidEquals(arr[i], g)) return true;
    return false;
}

bool NVENCEncoder::init(CUcontext cuCtx, int w, int h, int fps_num, int fps_den) {
    width = w;
    height = h;
    fpsn = fps_num;
    fpsd = fps_den ? fps_den : 1;

    // 0) API function table
    NVENCSTATUS st = NvEncodeAPICreateInstance(&f);
    if (st != NV_ENC_SUCCESS) {
        std::cerr << "NvEncodeAPICreateInstance failed: " << st << "\n";
        return false;
    }

    // 1) DLL/API version check
    uint32_t maxApi = 0;
    NVENCSTATUS stv = NvEncodeAPIGetMaxSupportedVersion(&maxApi);
    if (stv == NV_ENC_SUCCESS) {
        std::cout << "NVENC: Header API=" << NVENCAPI_VERSION
                  << " Driver API=" << maxApi << std::endl;
        if (NVENCAPI_VERSION > maxApi) {
            std::cerr << "NVENC: Driver too old for headers. Update NVIDIA driver.\n";
            return false;
        }
    } else {
        std::cerr << "NVENC: NvEncodeAPIGetMaxSupportedVersion failed\n";
    }

    // 2) Open encode session
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open = {NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER};
    open.device = cuCtx;
    open.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    open.apiVersion = NVENCAPI_VERSION;
    st = f.nvEncOpenEncodeSessionEx(&open, &enc);
    if (st != NV_ENC_SUCCESS) {
        std::cerr << "nvEncOpenEncodeSessionEx failed: " << st << "\n";
        return false;
    }

    // 3) Enumerate H.264 presets supported by DLL
    uint32_t count = 0, got = 0;
    st = f.nvEncGetEncodePresetCount(enc, NV_ENC_CODEC_H264_GUID, &count);
    std::cout << "nvEncGetEncodePresetCount(H264) st=" << st << " count=" << count << std::endl;
    if (st != NV_ENC_SUCCESS || count == 0) return false;

    std::vector<GUID> presets(count);
    st = f.nvEncGetEncodePresetGUIDs(enc, NV_ENC_CODEC_H264_GUID, presets.data(), count, &got);
    std::cout << "nvEncGetEncodePresetGUIDs st=" << st << " got=" << got << std::endl;
    if (st != NV_ENC_SUCCESS || got == 0) return false;

    auto printGUID = [](const GUID &g) {
        const uint8_t *p = reinterpret_cast<const uint8_t *>(&g);
        char buf[64];
        sprintf(buf, "%02X%02X%02X%02X-%02X%02X%02X%02X", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
        return std::string(buf);
    };

    // 4) Try Initialize with presetGUID directly (no encodeConfig) — fastest, most compatible
    auto try_init_with_preset = [&](const GUID &preset) -> NVENCSTATUS {
        NV_ENC_INITIALIZE_PARAMS ip = {NV_ENC_INITIALIZE_PARAMS_VER};
        ip.encodeGUID = NV_ENC_CODEC_H264_GUID;
        ip.presetGUID = preset;
        ip.tuningInfo = NV_ENC_TUNING_INFO_UNDEFINED;// безопасно
        ip.encodeWidth = width;
        ip.encodeHeight = height;
        ip.darWidth = width;
        ip.darHeight = height;
        ip.frameRateNum = fpsn;
        ip.frameRateDen = fpsd;
        ip.maxEncodeWidth = width;
        ip.maxEncodeHeight = height;
        ip.enablePTD = 1;
        ip.encodeConfig = nullptr;// ключевой момент: доверяем presetGUID
        NVENCSTATUS s = f.nvEncInitializeEncoder(enc, &ip);
        std::cout << "Init with preset head=" << printGUID(preset) << " st=" << s << std::endl;
        return s;
    };

    // 4a) First preset as chosen
    if (try_init_with_preset(presets[0]) == NV_ENC_SUCCESS) {
        std::cout << "NVENC initialized with presets[0]\n";
        return true;
    }

    // 4b) Try other presets from list
    for (uint32_t i = 1; i < got; ++i) {
        if (try_init_with_preset(presets[i]) == NV_ENC_SUCCESS) {
            std::cout << "NVENC initialized with presets[" << i << "]\n";
            return true;
        }
    }

    // 4c) Try zero preset GUID (let driver fully choose defaults)
    GUID zero{};// all zeros
    if (try_init_with_preset(zero) == NV_ENC_SUCCESS) {
        std::cout << "NVENC initialized with zero preset GUID\n";
        return true;
    }

    std::cerr << "NVENC init failed: no preset GUID accepted\n";
    return false;
}


bool NVENCEncoder::encodeNV12(CUdeviceptr d_nv12, uint32_t pitch, std::vector<uint8_t> &outBitstream, bool &isKey) {
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
        f.nvEncUnlockBitstream(enc, bs.bitstreamBuffer);
    } else {
        std::cerr << "nvEncLockBitstream: " << st << "\n";
    }

    f.nvEncDestroyBitstreamBuffer(enc, bs.bitstreamBuffer);
    f.nvEncUnmapInputResource(enc, map.mappedResource);
    f.nvEncUnregisterResource(enc, reg.registeredResource);
    return !outBitstream.empty();
}

void NVENCEncoder::destroy() {
    if (enc) {
        f.nvEncDestroyEncoder(enc);
        enc = nullptr;
    }
}
