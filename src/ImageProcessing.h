#ifndef IMAGEFILTERING_IMAGEPROCESSING_H
#define IMAGEFILTERING_IMAGEPROCESSING_H

#include "libs.h"

#include <algorithm>
#include <cmath>

void negativeFilter(int width, int height, int channels, unsigned char *&img);
unsigned char *grayFilter(int width, int height, int channels, const unsigned char *img);
void upscaleImage(int width, int height, int channels, const unsigned char *img, int scale);
unsigned char *prewittFilter(int width, int height, const unsigned char *gray);
unsigned char *bilateralFilter(const unsigned char *img, int width, int height, int channels, int radius, float sigmaSpace, float sigmaColor);

unsigned char *rndPr(int width, int height, int channels, unsigned char *img);

bool saveImage(const std::string &fullPath, int width, int height, int channels, const unsigned char *data);
unsigned char *loadImage(const std::string &filepath, int &width, int &height, int &channels);
void freeImage(unsigned char *img);
void assertImage(const unsigned char *h_img, const unsigned char *d_img, int size);

#endif//IMAGEFILTERING_IMAGEPROCESSING_H
