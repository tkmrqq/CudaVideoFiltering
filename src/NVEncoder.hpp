#pragma once
#include "cuda_iface.hpp"
#include <cstdint>
#include <nvEncodeAPI.h>
#include <vector>

class NVENCEncoder {
public:
    bool init(CUcontext cuCtx, int w, int h, int fps_num, int fps_den);
    // frameIdx — индекс кадра в выходной шкале, forceIdrNow — форсировать IDR для этого кадра
    bool encodeNV12(CUdeviceptr d_nv12, uint32_t pitch, std::vector<uint8_t> &outBitstream, bool &isKey, int frameIdx, bool forceIdrNow = false);
    void destroy();
    // gop_len в кадрах: для 1080p24 начни с fps (каждую секунду) или fps/2 при сильном смазе
    bool reconfigureBitrate(int avg_bps, int max_bps, int gop_len);
    ~NVENCEncoder() { destroy(); }

private:
    NV_ENCODE_API_FUNCTION_LIST f = {NV_ENCODE_API_FUNCTION_LIST_VER};
    void *enc = nullptr;
    int width = 0, height = 0, fpsn = 0, fpsd = 1;

    // Управление ключевыми кадрами
    int forceIdrCount = 2;    // принудить IDR на первых двух кадрах
    int gopLen = 0;           // целевой GOP
    int lastKeyIdx = -1000000;// последний ключевой индекс
    int tinyCount = 0;        // подряд очень маленькие P-кадры (<200 байт)
};
