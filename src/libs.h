#ifndef IMAGEFILTERING_LIBS_H
#define IMAGEFILTERING_LIBS_H

#include <iostream>
#include <string>

#include "ImageProcessing.h"
#include <chrono>

#define TIMER_START auto time_start = std::chrono::high_resolution_clock::now();

#define TIMER_STOP(name)                                                \
    auto time_end = std::chrono::high_resolution_clock::now();          \
    std::cout << "Elapsed Time for " name ": "                          \
              << std::chrono::duration_cast<std::chrono::milliseconds>( \
                         time_end - time_start)                         \
                         .count()                                       \
              << " ms" << std::endl;

std::string chooseImage(const std::string &directory);
extern "C" void prewittCUDA(unsigned char *img, unsigned char *result, int width, int height);

#endif//IMAGEFILTERING_LIBS_H
