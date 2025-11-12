#include "MP4Muxer.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

static inline bool is_start(const uint8_t *d, int sz, int k, int &sc_len) {
    if (k + 3 < sz && d[k] == 0 && d[k + 1] == 0 && d[k + 2] == 1) {
        sc_len = 3;
        return true;
    }
    if (k + 4 < sz && d[k] == 0 && d[k + 1] == 0 && d[k + 2] == 0 && d[k + 3] == 1) {
        sc_len = 4;
        return true;
    }
    return false;
}
static void be32(uint8_t *p, uint32_t v) {
    p[0] = (v >> 24) & 0xFF;
    p[1] = (v >> 16) & 0xFF;
    p[2] = (v >> 8) & 0xFF;
    p[3] = v & 0xFF;
}

void MP4Muxer::find_start_codes(const uint8_t *data, int size, std::vector<std::pair<int, int>> &ranges) {
    int i = 0, start = -1, sc = 0;
    while (i < size) {
        if (is_start(data, size, i, sc)) {
            if (start >= 0) ranges.emplace_back(start, i);
            start = i + sc;
            i += sc;
        } else {
            ++i;
        }
    }
    if (start >= 0) ranges.emplace_back(start, size);
}

void MP4Muxer::parse_sps_pps_from_annexb(const uint8_t *data, int size, std::vector<uint8_t> &sps, std::vector<uint8_t> &pps) {
    std::vector<std::pair<int, int>> rs;
    find_start_codes(data, size, rs);
    sps.clear();
    pps.clear();
    for (auto [s, e]: rs) {
        if (s < 0 || e <= s) continue;
        int nal_type = data[s] & 0x1F;
        if (nal_type == 7) sps.assign(data + s, data + e);
        else if (nal_type == 8)
            pps.assign(data + s, data + e);
    }
}

bool MP4Muxer::annexb_to_avcc_packet(const uint8_t *data, int size, std::vector<uint8_t> &out) {
    std::vector<std::pair<int, int>> rs;
    find_start_codes(data, size, rs);
    out.clear();
    out.reserve(size + 4 * 4);
    for (auto [s, e]: rs) {
        if (s < 0 || e <= s) continue;
        // Пропускаем "AUD" NAL (type 9), он не нужен в MP4
        int nal_type = data[s] & 0x1F;
        if (nal_type == 9) continue;
        uint32_t len = (uint32_t) (e - s);
        size_t base = out.size();
        out.resize(base + 4 + len);
        be32(out.data() + base, len);
        memcpy(out.data() + base + 4, data + s, len);
    }
    return !out.empty();
}

bool MP4Muxer::open(const char *path, int w, int h, int fps) {
    avformat_alloc_output_context2(&oc, nullptr, "mp4", path);
    if (!oc) {
        std::cerr << "[MP4] alloc_output_context failed\n";
        return false;
    }
    st = avformat_new_stream(oc, nullptr);
    if (!st) {
        std::cerr << "[MP4] new_stream failed\n";
        return false;
    }

    st->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    st->codecpar->codec_id = AV_CODEC_ID_H264;
    st->codecpar->width = w;
    st->codecpar->height = h;
    st->codecpar->codec_tag = 0;// auto
    st->time_base = tb = AVRational{1, fps};
    st->avg_frame_rate = AVRational{fps, 1};

    if (!(oc->oformat->flags & AVFMT_NOFILE)) {
        int r = avio_open(&oc->pb, path, AVIO_FLAG_WRITE);
        if (r < 0) {
            char err[128];
            av_strerror(r, err, sizeof(err));
            std::cerr << "[MP4] avio_open: " << err << "\n";
            return false;
        }
    }
    std::cout << "[MP4] open OK: " << path << " " << w << "x" << h << "@" << fps << "\n";
    return true;// header будет позже, когда появятся SPS/PPS
}

bool MP4Muxer::write(const uint8_t *data, int size, bool key, int64_t frameIndex) {
    if (!header_written && key) {
        std::vector<uint8_t> sps, pps;
        parse_sps_pps_from_annexb(data, size, sps, pps);
        if (!sps.empty()) last_sps = sps;
        if (!pps.empty()) last_pps = pps;

        std::cout << "[MP4] keyframe: sps=" << sps.size() << " pps=" << pps.size() << "\n";
        std::cout << "[MP4] have_sps=" << (!last_sps.empty()) << " have_pps=" << (!last_pps.empty()) << "\n";

        if (!last_sps.empty() && !last_pps.empty()) {
            // build avcC
            avcc.clear();
            int profile_idc = (last_sps.size() >= 2) ? last_sps[1] : 100;
            int compat = (last_sps.size() >= 3) ? last_sps[2] : 0;
            int level_idc = (last_sps.size() >= 4) ? last_sps[3] : 40;
            avcc.push_back(1);
            avcc.push_back((uint8_t) profile_idc);
            avcc.push_back((uint8_t) compat);
            avcc.push_back((uint8_t) level_idc);
            avcc.push_back(0xFF);// 4-byte len

            // SPS
            avcc.push_back(0xE1);
            uint16_t spsz = (uint16_t) last_sps.size();
            avcc.push_back((spsz >> 8) & 0xFF);
            avcc.push_back(spsz & 0xFF);
            avcc.insert(avcc.end(), last_sps.begin(), last_sps.end());
            // PPS
            avcc.push_back(1);
            uint16_t ppsz = (uint16_t) last_pps.size();
            avcc.push_back((ppsz >> 8) & 0xFF);
            avcc.push_back(ppsz & 0xFF);
            avcc.insert(avcc.end(), last_pps.begin(), last_pps.end());

            st->codecpar->extradata = (uint8_t *) av_mallocz(avcc.size() + AV_INPUT_BUFFER_PADDING_SIZE);
            st->codecpar->extradata_size = (int) avcc.size();
            memcpy(st->codecpar->extradata, avcc.data(), avcc.size());

            int rh = avformat_write_header(oc, nullptr);
            if (rh < 0) {
                char err[128];
                av_strerror(rh, err, sizeof(err));
                std::cerr << "[MP4] write_header: " << err << "\n";
                return false;
            }
            std::cout << "[MP4] write_header OK, avcC=" << st->codecpar->extradata_size << " bytes\n";

            // дамп начала avcC
            auto *ed = st->codecpar->extradata;
            int n = std::min(st->codecpar->extradata_size, 16);
            std::cout << "[MP4] avcC head:";
            for (int i = 0; i < n; i++) std::cout << " " << std::hex << (int) ed[i] << std::dec;
            std::cout << "\n";

            header_written = true;
        } else {
            return true;// ждём следующий ключ
        }
    }
    if (!header_written) return true;

    // Annex-B -> AVCC
    std::vector<uint8_t> pktbuf;
    if (!annexb_to_avcc_packet(data, size, pktbuf)) return true;

    auto *q = pktbuf.data();
    int m = std::min<int>(pktbuf.size(), 8);
    std::cout << "[MP4] pkt head:";
    for (int i = 0; i < m; i++) std::cout << " " << std::hex << (int) q[i] << std::dec;
    std::cout << "\n";

    AVPacket pkt;
    av_init_packet(&pkt);
    pkt.data = pktbuf.data();
    pkt.size = (int) pktbuf.size();
    pkt.stream_index = st->index;
    pkt.pts = frameIndex;
    pkt.dts = frameIndex;
    pkt.duration = 1;
    if (key) pkt.flags |= AV_PKT_FLAG_KEY;
    av_packet_rescale_ts(&pkt, tb, st->time_base);

    int r = av_interleaved_write_frame(oc, &pkt);
    if (r < 0) {
        char err[128];
        av_strerror(r, err, sizeof(err));
        std::cerr << "[MP4] write_frame: " << err << "\n";
        return false;
    }
    std::cout << "[MP4] write_frame OK size=" << pkt.size << " pts=" << (long long) frameIndex << " key=" << key << "\n";
    return true;
}

void MP4Muxer::close() {
    if (oc) {
        std::cout << "[MP4] close()\n";
        av_write_trailer(oc);
        if (!(oc->oformat->flags & AVFMT_NOFILE)) avio_closep(&oc->pb);
        avformat_free_context(oc);
        oc = nullptr;
        st = nullptr;
        header_written = false;
        last_sps.clear();
        last_pps.clear();
        avcc.clear();
    }
}
