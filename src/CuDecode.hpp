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

class CuDecode {
public:
    int width, height;
    CuDecode(CUcontext ctx, int width, int height);
    ~CuDecode();

    void setFrameCallback(std::function<void(const std::vector<uint8_t> &, int, int, int)> cb);
    bool init(cudaVideoCodec codec);
    void decodePacket(const AVPacket *pkt);
    void attachEncoderMux(NVENCEncoder *enc, MkvMuxer *mux);

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

    static int CUDAAPI HandleVideoSequence(void *userData, CUVIDEOFORMAT *format);

    static int CUDAAPI HandlePictureDecode(void *userData, CUVIDPICPARAMS *pic);

    static int CUDAAPI HandlePictureDisplay(void *userData, CUVIDPARSERDISPINFO *dispInfo);
};