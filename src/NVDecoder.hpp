#pragma once
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

private:
    //frame helpers
    CUdeviceptr last_nv12_dup = 0;
    uint32_t last_nv12_pitch = 0;
    bool have_dup = false;
    int64_t frameIndex = 0;// общий счётчик кадров для muxer

    //NVENC CLASSES
    NVENCEncoder *encoder = nullptr;
    MkvMuxer *muxer = nullptr;

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

        CUVIDPROCPARAMS vpp = {};
        vpp.progressive_frame = dispInfo->progressive_frame;
        vpp.top_field_first = dispInfo->top_field_first;

        // Map NVDEC output frame
        CUdeviceptr nv12dev;
        unsigned int pitch = 0;
        CUresult mres = cuvidMapVideoFrame(self->cuDecoder, dispInfo->picture_index, &nv12dev, &pitch, &vpp);
        if (mres != CUDA_SUCCESS || pitch == 0) {
            std::cerr << "ERROR: cuvidMapVideoFrame failed, CUresult=" << mres << ", pitch=" << pitch << std::endl;
            return 1;
        }

        std::cout << "Display pitch: " << pitch << " width: " << self->width << std::endl;

        // Выделим RGB буфер на GPU
        unsigned char *d_rgb;
        cudaMalloc(&d_rgb, self->width * self->height * 3);

        // Разделяем NV12
        unsigned char *yPlane = (unsigned char *) nv12dev;
        unsigned char *uvPlane = yPlane + pitch * self->height;

        // Запустим ядро
        NV12ToRGB(nv12dev, self->width, self->height, pitch, d_rgb);

        unsigned char *d_rgb_out = nullptr;
        cudaMalloc(&d_rgb_out, self->width * self->height * 3);
        prewittColorCUDA_device(d_rgb, d_rgb_out, self->width, self->height);

        std::cout << "Display frame pts: " << dispInfo->timestamp
                  << " picture_index: " << dispInfo->picture_index
                  << " progressive_frame: " << dispInfo->progressive_frame << std::endl;

        //RGB -> NV12
        unsigned char *d_nv12_out = nullptr;
        size_t pitch_out = 0;
        cudaMallocPitch(&d_nv12_out, &pitch_out, self->width, self->height * 3 / 2);
        RGBToNV12(d_rgb_out, self->width, self->height, d_nv12_out, (int) pitch_out);

        std::vector<uint8_t> bs;
        bool isKey = false;
        if (self->encoder && self->muxer) {
            if (self->encoder->encodeNV12((CUdeviceptr) d_nv12_out, (uint32_t) pitch_out, bs, isKey)) {
                static int64_t frameIndex = 0;
                self->muxer->write(bs.data(), (int) bs.size(), isKey, frameIndex++);
            }
        }

        // Копируем кадр в host
        // std::vector<uint8_t> rgb_host(self->width * self->height * 3);
        // cudaMemcpy(rgb_host.data(), d_rgb_out, rgb_host.size(), cudaMemcpyDeviceToHost);
        // cudaFree(d_rgb);

        cuvidUnmapVideoFrame(self->cuDecoder, nv12dev);

        // Внешний пользовательский обработчик: saveImage, фильтр и т.д.
        // if (self->frameCallback)
        //     self->frameCallback(rgb_host, self->width, self->height, dispInfo->timestamp);

        return 1;
    }
};
