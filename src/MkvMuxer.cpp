#include "MkvMuxer.hpp"
#include <iostream>
#include <vector>

static void find_start_codes(const uint8_t *data, int size, std::vector<std::pair<int, int>> &ranges) {
    // находим все NAL диапазоны [start,end)
    int i = 0, start = -1;
    auto is_start = [&](int k) {
        return k + 3 < size && data[k] == 0 && data[k + 1] == 0 && ((data[k + 2] == 1) || (data[k + 2] == 0 && k + 4 < size && data[k + 3] == 1));
    };
    while (i < size) {
        if (is_start(i)) {
            if (start >= 0) ranges.emplace_back(start, i);// закрываем предыдущий
            // пропускаем префикс
            if (data[i + 2] == 1) {
                start = i + 3;
                i += 3;
            } else {
                start = i + 4;
                i += 4;
            }
        } else {
            ++i;
        }
    }
    if (start >= 0) ranges.emplace_back(start, size);
}

static void build_h264_extradata_annexb(const uint8_t *data, int size, std::vector<uint8_t> &extradata) {
    std::vector<std::pair<int, int>> nal_ranges;
    find_start_codes(data, size, nal_ranges);
    extradata.clear();
    auto append_start = [&](int len) {
        static const uint8_t sc3[3] = {0, 0, 1};
        static const uint8_t sc4[4] = {0, 0, 0, 1};
        if (len == 3) extradata.insert(extradata.end(), sc3, sc3 + 3);
        else
            extradata.insert(extradata.end(), sc4, sc4 + 4);
    };
    // Собираем только SPS/PPS, оставляя Annex-B префиксы
    for (auto [s, e]: nal_ranges) {
        if (s >= e) continue;
        uint8_t nal_hdr = data[s];
        int nal_type = nal_hdr & 0x1F;// H.264
        if (nal_type == 7 || nal_type == 8) {
            append_start(4);
            extradata.insert(extradata.end(), data + s, data + e);
        }
    }
}

bool MkvMuxer::open(const char *path, int w, int h, int fps) {
    avformat_alloc_output_context2(&oc, nullptr, nullptr, path);
    if (!oc) return false;
    st = avformat_new_stream(oc, nullptr);
    if (!st) return false;
    st->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    st->codecpar->codec_id = AV_CODEC_ID_H264;
    st->codecpar->width = w;
    st->codecpar->height = h;
    tb = AVRational{1, fps};
    st->time_base = tb;
    st->avg_frame_rate = AVRational{fps, 1};
    if (!(oc->oformat->flags & AVFMT_NOFILE)) {
        int r = avio_open(&oc->pb, path, AVIO_FLAG_WRITE);
        if (r < 0) {
            char err[128];
            av_strerror(r, err, sizeof(err));
            std::cerr << "avio_open: " << err << "\n";
            return false;
        }
    }
    return true;// хедер пока не пишем
}


bool MkvMuxer::write(const uint8_t *data, int size, bool key, int64_t frameIndex) {
    if (!header_written && key) {
        // собрать extradata из SPS/PPS текущего IDR
        std::vector<uint8_t> ed;
        build_h264_extradata_annexb(data, size, ed);// см. выше
        if (!ed.empty()) {
            st->codecpar->extradata = (uint8_t *) av_mallocz(ed.size() + AV_INPUT_BUFFER_PADDING_SIZE);
            st->codecpar->extradata_size = (int) ed.size();
            memcpy(st->codecpar->extradata, ed.data(), ed.size());
        }
        int rh = avformat_write_header(oc, nullptr);
        if (rh < 0) {
            char err[128];
            av_strerror(rh, err, sizeof(err));
            std::cerr << "write_header: " << err << "\n";
            return false;
        }
        header_written = true;
    }
    if (!header_written) {
        // ещё не было ключа с SPS/PPS — отложим до первого keyframe
        return true;
    }
    AVPacket pkt;
    av_init_packet(&pkt);
    pkt.data = const_cast<uint8_t *>(data);
    pkt.size = size;
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
        std::cerr << "write_frame: " << err << "\n";
        return false;
    }
    return true;
}

void MkvMuxer::close() {
    if (oc) {
        av_write_trailer(oc);
        if (!(oc->oformat->flags & AVFMT_NOFILE)) avio_closep(&oc->pb);
        avformat_free_context(oc);
        oc = nullptr;
        st = nullptr;
    }
}
