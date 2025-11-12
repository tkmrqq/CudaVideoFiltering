#pragma once
// #include "MP4Muxer.hpp"
#include "MkvMuxer.hpp"
#include "NVEncoder.hpp"
#include "cuda_iface.hpp"
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <nvcuvid.h>
#include <string>
#include <vector>

class NVENCEncoder;
class MkvMuxer;

class NVDecoder {
public:
    NVDecoder(CUcontext ctx, int width, int height)
        : cuCtx(ctx), cuParser(nullptr), cuDecoder(nullptr), width(width), height(height) {}

    ~NVDecoder() {
        if (cuParser) cuvidDestroyVideoParser(cuParser);
        if (cuDecoder) cuvidDestroyDecoder(cuDecoder);
    }

    void setFrameCallback(std::function<void(const std::vector<uint8_t> &, int, int, int)> cb) {
        frameCallback = std::move(cb);
    }

    // Инициализация парсера, колбэки static!
    bool init(cudaVideoCodec codec) {
        CUVIDPARSERPARAMS parserParams = {};
        parserParams.CodecType = cudaVideoCodec_H264;
        parserParams.ulMaxNumDecodeSurfaces = 8;
        parserParams.ulMaxDisplayDelay = 0;
        parserParams.pUserData = this;
        parserParams.pfnSequenceCallback = &NVDecoder::HandleVideoSequence;
        parserParams.pfnDecodePicture = &NVDecoder::HandlePictureDecode;
        parserParams.pfnDisplayPicture = &NVDecoder::HandlePictureDisplay;
        return cuvidCreateVideoParser(&cuParser, &parserParams) == CUDA_SUCCESS;
    }

    // main loop: передаете пакет после init
    void decodePacket(const AVPacket *pkt) {
        CUVIDSOURCEDATAPACKET pkt_in = {};
        pkt_in.payload = pkt ? pkt->data : nullptr;
        pkt_in.payload_size = pkt ? pkt->size : 0;
        pkt_in.flags = (pkt_in.payload && pkt_in.payload_size) ? 0 : CUVID_PKT_ENDOFSTREAM;
        cuvidParseVideoData(cuParser, &pkt_in);
    }

    //* Внешние полезные данные
    int width, height;// сохраняем для работы ядра
    void attachEncoderMux(NVENCEncoder *enc, MkvMuxer *mux) {
        this->encoder = enc;
        this->muxer = mux;
    }

    // void attachMP4(MP4Muxer *mp4m) { this->mp4 = mp4m; }

private:
    //frame helpers
    CUdeviceptr last_nv12_dup = 0;
    uint32_t last_nv12_pitch = 0;
    bool have_dup = false;
    int64_t frameIndex = 0;// общий счётчик кадров для muxer
    int dupCount = 0;

    //NVENC CLASSES
    NVENCEncoder *encoder = nullptr;
    MkvMuxer *muxer = nullptr;
    // MP4Muxer *mp4 = nullptr;

    CUcontext cuCtx;
    CUvideoparser cuParser;
    CUvideodecoder cuDecoder;
    std::function<void(const std::vector<uint8_t> &, int, int, int)> frameCallback;

    // Sequence-callback: создать декодер под параметры потока
    static int CUDAAPI HandleVideoSequence(void *userData, CUVIDEOFORMAT *format) {
        NVDecoder *self = reinterpret_cast<NVDecoder *>(userData);
        CUcontext old = nullptr;
        cuCtxPushCurrent(self->cuCtx);

        std::cout << "=== NVDEC Sequence Callback ===" << std::endl;
        std::cout << "Codec: " << format->codec << std::endl;
        std::cout << "Width: " << format->coded_width << " Height: " << format->coded_height << std::endl;
        std::cout << "ChromaFormat: " << format->chroma_format << std::endl;
        std::cout << "BitDepth: " << (int) format->bit_depth_luma_minus8 + 8 << std::endl;
        std::cout << "MinNumDecodeSurfaces: " << format->min_num_decode_surfaces << std::endl;

        CUVIDDECODECREATEINFO info = {};
        info.CodecType = cudaVideoCodec_H264;
        info.ulWidth = format->coded_width;
        info.ulHeight = format->coded_height;
        info.ulNumDecodeSurfaces = format->min_num_decode_surfaces;
        info.ChromaFormat = format->chroma_format;
        info.OutputFormat = cudaVideoSurfaceFormat_NV12;
        info.bitDepthMinus8 = format->bit_depth_luma_minus8;
        info.ulCreationFlags = cudaVideoCreate_Default;
        info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
        info.ulTargetWidth = format->coded_width;
        info.ulTargetHeight = format->coded_height;
        info.ulNumOutputSurfaces = 2;

        if (info.ulWidth == 0 || info.ulHeight == 0) {
            std::cerr << "ERROR: Width or Height is 0!" << std::endl;
            return 0;
        }

        CUresult res = cuvidCreateDecoder(&self->cuDecoder, &info);
        if (res != CUDA_SUCCESS) {
            std::cerr << "Failed to create NVDEC decoder! CUresult=" << res << std::endl;
            return 0;
        }

        self->width = (int) format->coded_width;
        self->height = (int) format->coded_height;

        printf("NVDEC SEQ: codec=%d, W=%d, H=%d, chroma=%d, bdm8=%d\n",
               (int) info.CodecType, (int) info.ulWidth, (int) info.ulHeight,
               (int) info.ChromaFormat, (int) info.bitDepthMinus8);
        return 1;
    }

    static int CUDAAPI HandlePictureDecode(void *userData, CUVIDPICPARAMS *pic) {
        NVDecoder *self = reinterpret_cast<NVDecoder *>(userData);
        cuvidDecodePicture(self->cuDecoder, pic);
        return 1;
    }

    // Display-callback: появляется NV12 — здесь делаем cuda ядро и save кадра
    static int CUDAAPI HandlePictureDisplay(void *userData, CUVIDPARSERDISPINFO *dispInfo) {
        NVDecoder *self = reinterpret_cast<NVDecoder *>(userData);
        CUcontext old = nullptr;
        cuCtxPushCurrent(self->cuCtx);

        CUVIDPROCPARAMS vpp = {};
        vpp.progressive_frame = dispInfo->progressive_frame;
        vpp.top_field_first = dispInfo->top_field_first;

#ifdef _DEBUG
        std::cout << "[DISP] pts=" << dispInfo->timestamp
                  << " picidx=" << dispInfo->picture_index
                  << " prog=" << dispInfo->progressive_frame << "\n";
#endif

        // Map NVDEC output frame
        CUdeviceptr nv12dev;
        unsigned int pitch = 0;
        CUresult mres = cuvidMapVideoFrame(self->cuDecoder, dispInfo->picture_index, &nv12dev, &pitch, &vpp);

        bool use_duplicate = (mres != CUDA_SUCCESS || pitch == 0);
        if (use_duplicate) self->dupCount++;
        else
            self->dupCount = 0;
        std::vector<uint8_t> bs;
        bool isKey = false;

        if (!use_duplicate) {
            // 1) NV12 -> RGB (GPU)
            unsigned char *d_rgb = nullptr;
            cudaMalloc(&d_rgb, self->width * self->height * 3);
            NV12ToRGB(nv12dev, self->width, self->height, pitch, d_rgb);

            // 2) Prewitt (GPU)
            unsigned char *d_rgb_out = nullptr;
            cudaMalloc(&d_rgb_out, self->width * self->height * 3);
            prewittColorCUDA_device(d_rgb, d_rgb_out, self->width, self->height);
            cudaFree(d_rgb);

            // 3) RGB -> NV12 (GPU, с собственным pitch_out)
            unsigned char *d_nv12_out = nullptr;
            size_t pitch_out = 0;
            cudaMallocPitch(&d_nv12_out, &pitch_out, self->width, self->height * 3 / 2);
            RGBToNV12(d_rgb_out, self->width, self->height, d_nv12_out, (int) pitch_out);
            cudaFree(d_rgb_out);

            size_t bytes_nv12 = pitch_out * self->height + pitch_out * (self->height / 2);
#ifdef _DEBUG
            std::cout << "[DISP] RGB->NV12 pitch_out=" << pitch_out
                      << " bytes=" << bytes_nv12 << "\n";
#endif
            // 4) Кодируем текущий кадр
            if (self->encoder && self->muxer) {
                bool forceIdrNow = (self->dupCount >= 2);
                // Первый кадр — принудительный IDR (через NVENC picFlags в encodeNV12, см. ниже) или через Reconfigure forceIDR
                self->encoder->encodeNV12((CUdeviceptr) d_nv12_out, (uint32_t) pitch_out, bs, isKey, (int) self->frameIndex, forceIdrNow);
                // MKV mux (заголовок пишем лениво при первом keyframe внутри muxer)
                self->muxer->write(bs.data(), (int) bs.size(), isKey, self->frameIndex++);
                // if (self->mp4) self->mp4->write(bs.data(), (int) bs.size(), isKey, self->frameIndex++);
            }

            // 5) Обновляем дупликат
            if (self->have_dup && self->last_nv12_dup) {
                cuMemFree(self->last_nv12_dup);
                self->last_nv12_dup = 0;
            }

#ifdef _DEBUG
            std::cout << "[DUP] store bytes=" << bytes_nv12
                      << " pitch=" << pitch_out << "\n";
#endif


            // Храним собственную копию NV12 буфера (чтобы не зависеть от жизни surface)
            CUdeviceptr dup;
            cuMemAlloc(&dup, pitch_out * self->height * 3 / 2);
            cuMemcpyDtoD(dup, (CUdeviceptr) d_nv12_out, pitch_out * self->height * 3 / 2);
            self->last_nv12_dup = dup;
            self->last_nv12_pitch = (uint32_t) pitch_out;
            self->have_dup = true;

            cudaFree(d_nv12_out);
            cuvidUnmapVideoFrame(self->cuDecoder, nv12dev);

        } else {
            // Дубликат предыдущего кадра, чтобы не было дыр в PTS
            if (self->have_dup && self->last_nv12_dup && self->encoder && self->muxer) {
                size_t dup_bytes = self->last_nv12_pitch * self->height + self->last_nv12_pitch * (self->height / 2);
#ifdef _DEBUG
                std::cout << "[DUP] use bytes=" << dup_bytes
                          << " pitch=" << self->last_nv12_pitch << "\n";
#endif
                bool forceIdrNow = (self->dupCount >= 2);
                self->encoder->encodeNV12(self->last_nv12_dup, self->last_nv12_pitch, bs, isKey, (int) self->frameIndex, forceIdrNow);
                self->muxer->write(bs.data(), (int) bs.size(), isKey, self->frameIndex++);
                // if (self->mp4) self->mp4->write(bs.data(), (int) bs.size(), isKey, self->frameIndex++);
            }
            // Если дубликата ещё нет (самый первый кадр и сразу bad pitch) — просто пропустим
        }
#ifdef _DEBUG
        std::cout << "[DISP] map res=" << mres << " pitch=" << pitch << " w=" << self->width << " h=" << self->height << "\n";
#endif
        cuCtxPopCurrent(&old);// ВЫКЛ контекст
        return 1;
    }
};
