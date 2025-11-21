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
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
        std::string ext = p.extension().string();
        for (auto &c: ext) c = (char) std::tolower((unsigned char) c);

        if (ext == ".jpg" || ext == ".jpeg") {
            int quality = 90;
            int ok = stbi_write_jpg(p.string().c_str(), width, height, channels, data, quality);
            return ok != 0;
        } else {
            int stride = width * channels;
            int ok = stbi_write_png(p.string().c_str(), width, height, channels, data, stride);
            return ok != 0;
        }
    } catch (const std::exception &e) {
        // можно залогировать e.what()
        return false;
    }
}