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
        if (cuParser)
            cuvidDestroyVideoParser(cuParser);
        if (cuDecoder)
            cuvidDestroyDecoder(cuDecoder);
    }

    void setFrameCallback(std::function<void(const std::vector<uint8_t> &, int, int, int)> cb) {
        frameCallback = std::move(cb);
    }

    bool init(cudaVideoCodec codec) {
        CUVIDPARSERPARAMS parserParams = {};
        parserParams.CodecType = codec;// Используем переданный кодек
        parserParams.ulMaxNumDecodeSurfaces = 8;
        parserParams.ulMaxDisplayDelay = 0;
        parserParams.pUserData = this;
        parserParams.pfnSequenceCallback = &NVDecoder::HandleVideoSequence;
        parserParams.pfnDecodePicture = &NVDecoder::HandlePictureDecode;
        parserParams.pfnDisplayPicture = &NVDecoder::HandlePictureDisplay;

        CUresult result = cuvidCreateVideoParser(&cuParser, &parserParams);
        if (result != CUDA_SUCCESS) {
            std::cerr << "Failed to create video parser: " << result << std::endl;
            return false;
        }
        return true;
    }

    void decodePacket(const AVPacket *pkt) {
        if (!cuParser) {
            std::cerr << "Video parser not initialized!" << std::endl;
            return;
        }

        CUVIDSOURCEDATAPACKET pkt_in = {};
        pkt_in.payload = pkt ? pkt->data : nullptr;
        pkt_in.payload_size = pkt ? pkt->size : 0;
        pkt_in.flags = (pkt_in.payload && pkt_in.payload_size) ? 0 : CUVID_PKT_ENDOFSTREAM;
        pkt_in.timestamp = pkt ? pkt->pts : 0;// Важно сохранять временные метки

        CUresult result = cuvidParseVideoData(cuParser, &pkt_in);
        if (result != CUDA_SUCCESS) {
            std::cerr << "Failed to parse video data: " << result << std::endl;
        }
    }

    int width, height;
    void attachEncoderMux(NVENCEncoder *enc, MkvMuxer *mux) {
        this->encoder = enc;
        this->muxer = mux;
    }

private:
    CUdeviceptr last_nv12_dup = 0;
    uint32_t last_nv12_pitch = 0;
    bool have_dup = false;
    int64_t frameIndex = 0;

    NVENCEncoder *encoder = nullptr;
    MkvMuxer *muxer = nullptr;

    CUcontext cuCtx;
    CUvideoparser cuParser;
    CUvideodecoder cuDecoder;
    std::function<void(const std::vector<uint8_t> &, int, int, int)> frameCallback;

    static int CUDAAPI HandleVideoSequence(void *userData, CUVIDEOFORMAT *format) {
        NVDecoder *self = reinterpret_cast<NVDecoder *>(userData);

        // Сохраняем и восстанавливаем контекст
        CUcontext oldCtx = nullptr;
        cuCtxPushCurrent(self->cuCtx);

        std::cout << "=== NVDEC Sequence Callback ===" << std::endl;
        std::cout << "Codec: " << format->codec << std::endl;
        std::cout << "Width: " << format->coded_width << " Height: " << format->coded_height << std::endl;
        std::cout << "ChromaFormat: " << format->chroma_format << std::endl;
        std::cout << "BitDepth: " << (int) format->bit_depth_luma_minus8 + 8 << std::endl;
        std::cout << "MinNumDecodeSurfaces: " << format->min_num_decode_surfaces << std::endl;

        // Если декодер уже существует, уничтожаем старый
        if (self->cuDecoder) {
            cuvidDestroyDecoder(self->cuDecoder);
            self->cuDecoder = nullptr;
        }

        CUVIDDECODECREATEINFO info = {};
        info.CodecType = format->codec;// Используем кодек из формата
        info.ulWidth = format->coded_width;
        info.ulHeight = format->coded_height;
        info.ulNumDecodeSurfaces = format->min_num_decode_surfaces + 2;// Добавляем запас
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
            cuCtxPopCurrent(&oldCtx);
            return 0;
        }

        CUresult res = cuvidCreateDecoder(&self->cuDecoder, &info);
        if (res != CUDA_SUCCESS) {
            std::cerr << "Failed to create NVDEC decoder! CUresult=" << res << std::endl;
            cuCtxPopCurrent(&oldCtx);
            return 0;
        }

        self->width = (int) format->coded_width;
        self->height = (int) format->coded_height;

        printf("NVDEC SEQ: codec=%d, W=%d, H=%d, chroma=%d, bdm8=%d\n",
               (int) info.CodecType, (int) info.ulWidth, (int) info.ulHeight,
               (int) info.ChromaFormat, (int) info.bitDepthMinus8);

        cuCtxPopCurrent(&oldCtx);
        return info.ulNumDecodeSurfaces;// Возвращаем количество поверхностей
    }

    static int CUDAAPI HandlePictureDecode(void *userData, CUVIDPICPARAMS *pic) {
        NVDecoder *self = reinterpret_cast<NVDecoder *>(userData);
        CUcontext oldCtx = nullptr;
        cuCtxPushCurrent(self->cuCtx);

        if (!self->cuDecoder) {
            std::cerr << "Decoder not initialized in HandlePictureDecode!" << std::endl;
            cuCtxPopCurrent(&oldCtx);
            return 0;
        }

        CUresult result = cuvidDecodePicture(self->cuDecoder, pic);
        if (result != CUDA_SUCCESS) {
            std::cerr << "Failed to decode picture: " << result << std::endl;
        }

        cuCtxPopCurrent(&oldCtx);
        return (result == CUDA_SUCCESS) ? 1 : 0;
    }

    static int CUDAAPI HandlePictureDisplay(void *userData, CUVIDPARSERDISPINFO *dispInfo) {
        NVDecoder *self = reinterpret_cast<NVDecoder *>(userData);
        CUcontext oldCtx = nullptr;
        cuCtxPushCurrent(self->cuCtx);

        if (!self->cuDecoder) {
            std::cerr << "Decoder not initialized in HandlePictureDisplay!" << std::endl;
            cuCtxPopCurrent(&oldCtx);
            return 0;
        }

        CUVIDPROCPARAMS vpp = {};
        vpp.progressive_frame = dispInfo->progressive_frame;
        vpp.top_field_first = dispInfo->top_field_first;
        vpp.second_field = 0;// Всегда первое поле

        CUdeviceptr nv12dev = 0;
        unsigned int pitch = 0;
        CUresult mres = cuvidMapVideoFrame(self->cuDecoder, dispInfo->picture_index, &nv12dev, &pitch, &vpp);

        if (mres != CUDA_SUCCESS || pitch == 0 || nv12dev == 0) {
            std::cerr << "Failed to map video frame: " << mres << " pitch: " << pitch << std::endl;
            cuCtxPopCurrent(&oldCtx);
            return 0;// Возвращаем 0 при ошибке
        }

        // Проверяем корректность размеров
        if (self->width <= 0 || self->height <= 0) {
            std::cerr << "Invalid dimensions: " << self->width << "x" << self->height << std::endl;
            cuvidUnmapVideoFrame(self->cuDecoder, nv12dev);
            cuCtxPopCurrent(&oldCtx);
            return 0;
        }

        try {
            // Выделяем память для RGB
            unsigned char *d_rgb = nullptr;
            cudaError_t cudaErr = cudaMalloc(&d_rgb, self->width * self->height * 3);
            if (cudaErr != cudaSuccess || !d_rgb) {
                throw std::runtime_error("Failed to allocate RGB memory");
            }

            // Конвертируем NV12 to RGB
            NV12ToRGB(nv12dev, self->width, self->height, pitch, d_rgb);

            // Выделяем память для выходного RGB
            unsigned char *d_rgb_out = nullptr;
            cudaErr = cudaMalloc(&d_rgb_out, self->width * self->height * 3);
            if (cudaErr != cudaSuccess || !d_rgb_out) {
                cudaFree(d_rgb);
                throw std::runtime_error("Failed to allocate output RGB memory");
            }

            // Применяем фильтр
            prewittColorCUDA_device(d_rgb, d_rgb_out, self->width, self->height);


            // Выделяем память для выходного NV12
            unsigned char *d_nv12_out = nullptr;
            size_t pitch_out = 0;
            cudaErr = cudaMallocPitch(&d_nv12_out, &pitch_out, self->width, self->height * 3 / 2);
            if (cudaErr != cudaSuccess || !d_nv12_out) {
                cudaFree(d_rgb);
                cudaFree(d_rgb_out);
                throw std::runtime_error("Failed to allocate output NV12 memory");
            }

            // Конвертируем обратно в NV12
            RGBToNV12(d_rgb_out, self->width, self->height, d_nv12_out, (int) pitch_out);

            // Кодируем и мультиплексируем
            if (self->encoder && self->muxer) {
                std::vector<uint8_t> encodedData;
                bool isKeyFrame = false;

                if (self->encoder->encodeNV12((CUdeviceptr) d_nv12_out, (uint32_t) pitch_out,
                                              encodedData, isKeyFrame, 0)) {
                    self->muxer->write(encodedData.data(), (int) encodedData.size(),
                                       isKeyFrame, self->frameIndex++);
                }
            }

            // Освобождаем ресурсы
            cudaFree(d_rgb);
            cudaFree(d_rgb_out);
            cudaFree(d_nv12_out);
        } catch (const std::exception &e) {
            std::cerr << "Error in frame processing: " << e.what() << std::endl;
        }

        // Всегда размаппируем кадр
        cuvidUnmapVideoFrame(self->cuDecoder, nv12dev);
        cuCtxPopCurrent(&oldCtx);
        return 1;
    }
};