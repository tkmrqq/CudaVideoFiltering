#include "ImageProcessing.h"
#include "libs.h"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void prewittKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
        return;

    int Gx[3][3] = {
            {-1, 0, 1},
            {-1, 0, 1},
            {-1, 0, 1}};
    int Gy[3][3] = {
            {1, 1, 1},
            {0, 0, 0},
            {-1, -1, -1}};

    float sumX = 0, sumY = 0;
    for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx) {
            int pixel = input[(y + ky) * width + (x + kx)];
            sumX += Gx[ky + 1][kx + 1] * pixel;
            sumY += Gy[ky + 1][kx + 1] * pixel;
        }

    float mag = sqrtf(sumX * sumX + sumY * sumY);
    mag = fminf(255.0f, fmaxf(0.0f, mag));

    output[y * width + x] = static_cast<unsigned char>(mag);
}

extern "C" void prewittCUDA(unsigned char *img, unsigned char *result, int width, int height) {
    unsigned char *d_input, *d_output;
    size_t size = width * height * sizeof(unsigned char);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, img, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // замер времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    prewittKernel<<<grid, block>>>(d_input, d_output, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA time: " << milliseconds << " ms\n";

    cudaMemcpy(result, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

__global__ void splitChannelsKernel(
        const unsigned char *input,
        unsigned char *r, unsigned char *g, unsigned char *b,
        int totalPixels) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= totalPixels) return;

#pragma unroll
    for (int p = 0; p < 4; ++p) {
        int i = idx + p;
        if (i >= totalPixels) break;

        int base = i * 3;
        r[i] = input[base + 0];
        g[i] = input[base + 1];
        b[i] = input[base + 2];
    }
}

__global__ void mergeChannelsKernel(
        const unsigned char *r, const unsigned char *g, const unsigned char *b,
        unsigned char *output,
        int totalPixels) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= totalPixels) return;

#pragma unroll
    for (int p = 0; p < 4; ++p) {
        int i = idx + p;
        if (i >= totalPixels) break;

        int base = i * 3;
        output[base + 0] = r[i];
        output[base + 1] = g[i];
        output[base + 2] = b[i];
    }
}

extern "C" void prewittColorCUDA(unsigned char *img, std::vector<uint8_t> result, int width, int height) {
    if (!img || !result.data()) return;

    int totalPixels = width * height;
    size_t bytesGray = totalPixels;
    size_t bytesRGB = totalPixels * 3;

    unsigned char *d_input, *d_output;
    unsigned char *d_r, *d_g, *d_b;
    unsigned char *d_r_out, *d_g_out, *d_b_out;

    // cudaMalloc(&d_input, bytesRGB);
    cudaMalloc(&d_output, bytesRGB);
    cudaMalloc(&d_r, bytesGray);
    cudaMalloc(&d_g, bytesGray);
    cudaMalloc(&d_b, bytesGray);
    cudaMalloc(&d_r_out, bytesGray);
    cudaMalloc(&d_g_out, bytesGray);
    cudaMalloc(&d_b_out, bytesGray);

    // cudaMemcpy(d_input, img, bytesRGB, cudaMemcpyHostToDevice);

    int threads = 256;
    int logicalThreads = (totalPixels + 4 - 1) / 4;
    int blocks = (logicalThreads + threads - 1) / threads;

    splitChannelsKernel<<<blocks, threads>>>(img, d_r, d_g, d_b, totalPixels);
    cudaDeviceSynchronize();

    dim3 block2d(16, 16);
    dim3 grid2d((width + block2d.x - 1) / block2d.x, (height + block2d.y - 1) / block2d.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    prewittKernel<<<grid2d, block2d>>>(d_r, d_r_out, width, height);
    prewittKernel<<<grid2d, block2d>>>(d_g, d_g_out, width, height);
    prewittKernel<<<grid2d, block2d>>>(d_b, d_b_out, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA colored Prewitt time: " << milliseconds << " ms\n";

    cudaDeviceSynchronize();

    mergeChannelsKernel<<<blocks, threads>>>(d_r_out, d_g_out, d_b_out, d_output, totalPixels);
    cudaDeviceSynchronize();

    cudaMemcpy(result.data(), d_output, bytesRGB, cudaMemcpyDeviceToHost);

    // cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_r_out);
    cudaFree(d_g_out);
    cudaFree(d_b_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// split, prewitt on each channel, merge - всё на GPU
extern "C" void prewittColorCUDA_device(
        const unsigned char *d_rgb_in,// device
        unsigned char *d_rgb_out,     // device
        int width, int height) {
    const int totalPixels = width * height;
    const size_t bytesGray = totalPixels;   // 1 byte per pixel
    const size_t bytesRGB = totalPixels * 3;// 3 channels

    unsigned char *d_r = nullptr, *d_g = nullptr, *d_b = nullptr;
    unsigned char *d_r_out = nullptr, *d_g_out = nullptr, *d_b_out = nullptr;

    cudaMalloc(&d_r, bytesGray);
    cudaMalloc(&d_g, bytesGray);
    cudaMalloc(&d_b, bytesGray);
    cudaMalloc(&d_r_out, bytesGray);
    cudaMalloc(&d_g_out, bytesGray);
    cudaMalloc(&d_b_out, bytesGray);

    int threads = 256;
    int logicalThreads = (totalPixels + 4 - 1) / 4;
    int blocks = (logicalThreads + threads - 1) / threads;

    // Разложение на каналы
    splitChannelsKernel<<<blocks, threads>>>(d_rgb_in, d_r, d_g, d_b, totalPixels);
    cudaDeviceSynchronize();

    dim3 block2d(16, 16);
    dim3 grid2d((width + block2d.x - 1) / block2d.x,
                (height + block2d.y - 1) / block2d.y);

    // Превитт по каналам
    prewittKernel<<<grid2d, block2d>>>(d_r, d_r_out, width, height);
    prewittKernel<<<grid2d, block2d>>>(d_g, d_g_out, width, height);
    prewittKernel<<<grid2d, block2d>>>(d_b, d_b_out, width, height);
    cudaDeviceSynchronize();

    // Слияние
    mergeChannelsKernel<<<blocks, threads>>>(d_r_out, d_g_out, d_b_out, d_rgb_out, totalPixels);
    cudaDeviceSynchronize();

    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_r_out);
    cudaFree(d_g_out);
    cudaFree(d_b_out);
}

__global__ void nv12_to_rgb_kernel(
        const unsigned char *__restrict__ yPlane,
        const unsigned char *__restrict__ uvPlane,
        unsigned char *__restrict__ rgb,
        int width, int height, int pitchY, int pitchUV) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Индексация: pitch — это количество байт между строками!
    int Y = yPlane[y * pitchY + x];

    // В NV12 UV interleaved, лежит с выравниванием (то есть не после width, а после pitch)
    int uvRow = (y / 2) * pitchUV;
    int uvCol = (x / 2) * 2;
    int U = uvPlane[uvRow + uvCol + 0] - 128;
    int V = uvPlane[uvRow + uvCol + 1] - 128;

    int R = (int) (Y + 1.402f * V);
    int G = (int) (Y - 0.344f * U - 0.714f * V);
    int B = (int) (Y + 1.772f * U);

    R = min(max(R, 0), 255);
    G = min(max(G, 0), 255);
    B = min(max(B, 0), 255);

    int rgbIdx = (y * width + x) * 3;
    rgb[rgbIdx + 0] = R;
    rgb[rgbIdx + 1] = G;
    rgb[rgbIdx + 2] = B;
}

extern "C" void NV12ToRGB(
        CUdeviceptr d_nv12, int width, int height, int pitch,
        unsigned char *d_rgb) {
    unsigned char *yPlane = (unsigned char *) d_nv12;
    unsigned char *uvPlane = yPlane + pitch * height;
    dim3 block(16, 16), grid((width + 15) / 16, (height + 15) / 16);
    nv12_to_rgb_kernel<<<grid, block>>>(
            yPlane, uvPlane, d_rgb, width, height, pitch, pitch);
    cudaDeviceSynchronize();
}


// RGB (interleaved) -> NV12 (Y full-res + interleaved UV at half-res)
// d_rgb:   width*height*3
// d_nv12:  pitch*height  + pitch*(height/2) * 2
__global__ void rgb_to_nv12_kernel(
        const unsigned char *__restrict__ rgb,
        unsigned char *__restrict__ yPlane,
        unsigned char *__restrict__ uvPlane,
        int width, int height, int pitchY, int pitchUV) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;// 0..width-1
    int y = blockIdx.y * blockDim.y + threadIdx.y;// 0..height-1
    if (x >= width || y >= height) return;

    int rgbIdx = (y * width + x) * 3;
    float R = rgb[rgbIdx + 0];
    float G = rgb[rgbIdx + 1];
    float B = rgb[rgbIdx + 2];

    // ITU-R BT.601 full range approx
    float Yf = 0.257f * R + 0.504f * G + 0.098f * B + 16.0f;
    float Uf = -0.148f * R - 0.291f * G + 0.439f * B + 128.0f;
    float Vf = 0.439f * R - 0.368f * G - 0.071f * B + 128.0f;

    // write Y
    yPlane[y * pitchY + x] = (unsigned char) min(max((int) (Yf + 0.5f), 0), 255);

    // write UV for top-left pixel of 2x2 block
    if ((x % 2 == 0) && (y % 2 == 0)) {
        // простое усреднение 2x2
        float Ru = R, Gu = G, Bu = B;
        float R2 = R, G2 = G, B2 = B, R3 = R, G3 = G, B3 = B, R4 = R, G4 = G, B4 = B;
        if (x + 1 < width) {
            int rgbIdx2 = (y * width + (x + 1)) * 3;
            R2 = rgb[rgbIdx2 + 0];
            G2 = rgb[rgbIdx2 + 1];
            B2 = rgb[rgbIdx2 + 2];
        }
        if (y + 1 < height) {
            int rgbIdx3 = ((y + 1) * width + x) * 3;
            R3 = rgb[rgbIdx3 + 0];
            G3 = rgb[rgbIdx3 + 1];
            B3 = rgb[rgbIdx3 + 2];
        }
        if (x + 1 < width && y + 1 < height) {
            int rgbIdx4 = ((y + 1) * width + (x + 1)) * 3;
            R4 = rgb[rgbIdx4 + 0];
            G4 = rgb[rgbIdx4 + 1];
            B4 = rgb[rgbIdx4 + 2];
        }
        float Rm = 0.25f * (Ru + R2 + R3 + R4);
        float Gm = 0.25f * (Gu + G2 + G3 + G4);
        float Bm = 0.25f * (Bu + B2 + B3 + B4);

        float U = -0.148f * Rm - 0.291f * Gm + 0.439f * Bm + 128.0f;
        float V = 0.439f * Rm - 0.368f * Gm - 0.071f * Bm + 128.0f;

        int uvIndex = (y / 2) * pitchUV + (x / 2) * 2;
        uvPlane[uvIndex + 0] = (unsigned char) min(max((int) (U + 0.5f), 0), 255);
        uvPlane[uvIndex + 1] = (unsigned char) min(max((int) (V + 0.5f), 0), 255);
    }
}

extern "C" void RGBToNV12(const unsigned char *d_rgb, int width, int height, unsigned char *d_nv12, int pitch) {
    unsigned char *yPlane = d_nv12;
    unsigned char *uvPlane = d_nv12 + pitch * height;
    dim3 block(16, 16), grid((width + 15) / 16, (height + 15) / 16);
    rgb_to_nv12_kernel<<<grid, block>>>(d_rgb, yPlane, uvPlane, width, height, pitch, pitch);
    cudaDeviceSynchronize();
}
