#include "ImageProcessing.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <filesystem>

void assertImage(const unsigned char *h_img, const unsigned char *d_img, int size) {
    assert(sizeof(h_img) == sizeof(d_img));
    int diff = 0;
    for (size_t i = 0; i < size; i++) {
        if (h_img[i] != d_img[i]) {
            diff++;
        }
    }
    if (diff == 0) {
        std::cout << "Image arrays are egual!" << std::endl;
    } else {
        std::cout << "Diff in image arrays: " << diff << std::endl;
    }
}

unsigned char *loadImage(const std::string &filepath, int &width, int &height, int &channels) {
    unsigned char *img = stbi_load(filepath.c_str(), &width, &height, &channels, 0);
    return img;
}

void freeImage(unsigned char *img) {
    stbi_image_free(img);
}

void negativeFilter(int width, int height, int channels, unsigned char *&img) {
    int size = width * height * channels;
    for (int i = 0; i < size; i++) {
        img[i] = 255 - img[i];
    }
}

unsigned char *grayFilter(int width, int height, int channels, const unsigned char *img) {
    if (!img) {
        std::cerr << "Error: img == nullptr" << std::endl;
        return nullptr;
    }

    unsigned char *gray = new unsigned char[width * height];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            unsigned char r = img[idx + 0];
            unsigned char g = img[idx + 1];
            unsigned char b = img[idx + 2];

            gray[y * width + x] = static_cast<unsigned char>(
                    0.299f * r + 0.587f * g + 0.114f * b);
        }
    }

    return gray;
}

bool saveImage(const std::string &fullPath, int width, int height, int channels, const unsigned char *data) {
    namespace fs = std::filesystem;
    try {
        fs::path p(fullPath);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());// создаём каталоги рекурсивно [web:254]
        std::string ext = p.extension().string();
        for (auto &c: ext) c = (char) std::tolower((unsigned char) c);// нормализуем расширение [web:254]

        if (ext == ".jpg" || ext == ".jpeg") {
            int quality = 90;                                                                   // 1..100, типичный диапазон качества для JPG [web:202]
            int ok = stbi_write_jpg(p.string().c_str(), width, height, channels, data, quality);// запись JPG [web:218]
            return ok != 0;                                                                     // 1 — успех, 0 — ошибка [web:218]
        } else {                                                                                // PNG по умолчанию
            int stride = width * channels;                                                      // stride — байт на строку, обязателен для PNG [web:235][web:202]
            int ok = stbi_write_png(p.string().c_str(), width, height, channels, data, stride); // запись PNG [web:218]
            return ok != 0;                                                                     // 1 — успех, 0 — ошибка [web:218]
        }
    } catch (const std::exception &e) {
        // можно залогировать e.what()
        return false;// исключение при работе с ФС или путями [web:254]
    }
}

void upscaleImage(int width, int height, int channels, const unsigned char *img, int scale) {
    int n_width = width * scale;
    int n_height = height * scale;

    unsigned char *upscaled = new unsigned char[n_width * n_height * channels];

    for (int y = 0; y < n_height; ++y) {
        for (int x = 0; x < n_width; ++x) {
            int src_x = x / scale;
            int src_y = y / scale;

            int src_idx = (src_y * width + src_x) * channels;
            int dst_idx = (y * n_width + x) * channels;

            for (int c = 0; c < channels; ++c)
                upscaled[dst_idx + c] = img[src_idx + c];
        }
    }
    // saveImage(n_width, n_height, channels, upscaled, "/upscaled");
    delete[] upscaled;
}

unsigned char *prewittFilter(int width, int height, const unsigned char *gray) {
    if (!gray) {
        std::cerr << "Ошибка: gray == nullptr" << std::endl;
        return nullptr;
    }
    if (width <= 2 || height <= 2) {
        std::cerr << "Ошибка: слишком маленькое изображение" << std::endl;
        return nullptr;
    }

    int Gx[3][3] = {
            {-1, 0, 1},
            {-1, 0, 1},
            {-1, 0, 1}};
    int Gy[3][3] = {
            {1, 1, 1},
            {0, 0, 0},
            {-1, -1, -1}};

    unsigned char *result = new unsigned char[width * height];
    std::fill(result, result + width * height, 0);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float sumX = 0.0f;
            float sumY = 0.0f;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixel = gray[(y + ky) * width + (x + kx)];
                    sumX += Gx[ky + 1][kx + 1] * pixel;
                    sumY += Gy[ky + 1][kx + 1] * pixel;
                }
            }

            float mag = std::sqrt(sumX * sumX + sumY * sumY);
            result[y * width + x] = static_cast<unsigned char>(
                    std::clamp(mag, 0.0f, 255.0f));
        }
    }

    return result;
}

unsigned char *bilateralFilter(const unsigned char *img, int width, int height, int channels,
                               int radius, float sigmaSpace, float sigmaColor) {
    unsigned char *out = new unsigned char[width * height * channels];

    float twoSigmaSpace2 = 2 * sigmaSpace * sigmaSpace;
    float twoSigmaColor2 = 2 * sigmaColor * sigmaColor;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {

                float sum = 0.0f;
                float wsum = 0.0f;
                unsigned char center = img[(y * width + x) * channels + c];

                for (int ky = -radius; ky <= radius; ++ky) {
                    for (int kx = -radius; kx <= radius; ++kx) {
                        int nx = std::clamp(x + kx, 0, width - 1);
                        int ny = std::clamp(y + ky, 0, height - 1);
                        unsigned char neighbor = img[(ny * width + nx) * channels + c];

                        float spatialWeight = std::exp(-(kx * kx + ky * ky) / twoSigmaSpace2);
                        float colorWeight = std::exp(-((neighbor - center) * (neighbor - center)) / twoSigmaColor2);
                        float w = spatialWeight * colorWeight;

                        sum += w * neighbor;
                        wsum += w;
                    }
                }
                out[(y * width + x) * channels + c] = static_cast<unsigned char>(sum / wsum);
            }
        }
    }
    return out;
}


unsigned char *rndPr(int width, int height, int channels, unsigned char *img) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            img[idx] = 87;
            img[idx + 1] = 245;
            img[idx + 2] = 66;
        }
    }
    return img;
}