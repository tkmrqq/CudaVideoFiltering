#pragma once
extern "C" {
#include <libavformat/avformat.h>
}
#include <cstdint>
#include <vector>

class MP4Muxer {
public:
    bool open(const char *path, int w, int h, int fps);
    bool write(const uint8_t *data, int size, bool key, int64_t frameIndex);
    void close();

    ~MP4Muxer() { close(); }

private:
    AVFormatContext *oc = nullptr;
    AVStream *st = nullptr;
    AVRational tb{1, 1};
    bool header_written = false;

    // avcC cache
    std::vector<uint8_t> avcc;

    std::vector<uint8_t> last_sps, last_pps;

    // helpers
    static void find_start_codes(const uint8_t *data, int size, std::vector<std::pair<int, int>> &ranges);
    static bool build_avcc_extradata_from_idr(const uint8_t *data, int size, std::vector<uint8_t> &out);// SPS/PPS -> avcC blob
    static bool annexb_to_avcc_packet(const uint8_t *data, int size, std::vector<uint8_t> &out);        // replace start codes with 4-byte lengths
    static void parse_sps_pps_from_annexb(const uint8_t *data, int size, std::vector<uint8_t> &sps, std::vector<uint8_t> &pps);
};
