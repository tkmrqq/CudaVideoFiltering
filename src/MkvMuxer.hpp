#pragma once
extern "C" {
#include <libavformat/avformat.h>
}
#include <cstdint>

class MkvMuxer {
public:
    bool open(const char *path, int w, int h, int fps);
    bool write(const uint8_t *data, int size, bool key, int64_t frameIndex);
    void close();
    ~MkvMuxer() { close(); }

private:
    AVFormatContext *oc = nullptr;
    AVStream *st = nullptr;
    AVRational tb{1, 1};
    bool header_written = false;
};
