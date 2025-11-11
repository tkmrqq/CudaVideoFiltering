#pragma once
#include "cuda_iface.hpp"
#include <nvEncodeAPI.h>
#include <vector>

class NVENCEncoder {
public:
    bool init(CUcontext cuCtx, int w, int h, int fps_num, int fps_den);
    bool encodeNV12(CUdeviceptr d_nv12, uint32_t pitch, std::vector<uint8_t> &outBitstream, bool &isKey);
    void destroy();
    ~NVENCEncoder() { destroy(); }

private:
    NV_ENCODE_API_FUNCTION_LIST f = {NV_ENCODE_API_FUNCTION_LIST_VER};
    void *enc = nullptr;
    int width = 0, height = 0, fpsn = 0, fpsd = 1;
};
