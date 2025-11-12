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

    int sps_s = -1, sps_e = -1, pps_s = -1, pps_e = -1;
    for (auto [s, e]: nal_ranges) {
        if (s >= e) continue;
        int nal_type = data[s] & 0x1F;
        if (nal_type == 7) {
            sps_s = s;
            sps_e = e;
        }// запоминаем последний SPS
        else if (nal_type == 8) {
            pps_s = s;
            pps_e = e;
        }// последний PPS
    }
    if (sps_s >= 0 && sps_e > sps_s) {
        static const uint8_t sc4[4] = {0, 0, 0, 1};
        extradata.insert(extradata.end(), sc4, sc4 + 4);
        extradata.insert(extradata.end(), data + sps_s, data + sps_e);
    }
    if (pps_s >= 0 && pps_e > pps_s) {
        static const uint8_t sc4[4] = {0, 0, 0, 1};
        extradata.insert(extradata.end(), sc4, sc4 + 4);
        extradata.insert(extradata.end(), data + pps_s, data + pps_e);
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
        std::vector<uint8_t> ed;
        build_h264_extradata_annexb(data, size, ed);
        if (ed.empty()) {
            // ключевой пакет без SPS/PPS — ждём следующий ключ
            return true;
        }
        st->codecpar->extradata = (uint8_t *) av_mallocz(ed.size() + AV_INPUT_BUFFER_PADDING_SIZE);
        st->codecpar->extradata_size = (int) ed.size();
        memcpy(st->codecpar->extradata, ed.data(), ed.size());

        auto dump_nal_types = [](const uint8_t *d, int sz) {
            int i = 0;
            auto is_sc = [&](int k) { return k + 3 < sz && d[k] == 0 && d[k + 1] == 0 && ((d[k + 2] == 1) || (d[k + 2] == 0 && k + 4 < sz && d[k + 3] == 1)); };
            int start = -1;
            std::vector<int> types;
            while (i < sz) {
                if (is_sc(i)) {
                    if (start >= 0) {}
                    if (d[i + 2] == 1) {
                        start = i + 3;
                        i += 3;
                    } else {
                        start = i + 4;
                        i += 4;
                    }
                } else {
                    ++i;
                }
                if (start >= 0 && (i >= sz || is_sc(i))) {
                    int t = d[start] & 0x1F;
                    types.push_back(t);
                }
            }
#ifdef _DEBUG
            std::cout << "[MUX] extradata NAL types:";
            for (int t: types) std::cout << " " << t;
            std::cout << "\n";
#endif
        };
#ifdef _DEBUG
        std::cout << "[MUX] keyframe before header: ed_size=" << (int) ed.size() << "\n";
#endif
        dump_nal_types(ed.data(), (int) ed.size());


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
        // Ещё ждем «полезный» IDR с SPS/PPS
        return true;
    }
#ifdef _DEBUG
    std::cout << "[MUX] write pkt pts=" << (long long) frameIndex
              << " key=" << (key ? 1 : 0)
              << " sz=" << size << "\n";
#endif

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
