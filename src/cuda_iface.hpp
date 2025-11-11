#pragma once
#include <cuda.h>
#include <vector>
extern "C" void NV12ToRGB(CUdeviceptr d_nv12, int width, int height, int pitch, unsigned char *d_rgb);
extern "C" void prewittColorCUDA(unsigned char *img, std::vector<uint8_t> result, int width, int height);
extern "C" void prewittColorCUDA_device(const unsigned char *d_rgb_in, unsigned char *d_rgb_out, int width, int height);
extern "C" void RGBToNV12(const unsigned char *d_rgb, int width, int height, unsigned char *d_nv12, int pitch);