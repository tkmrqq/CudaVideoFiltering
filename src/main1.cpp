#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
}

#include "CUDecode.hpp"
#include "ImageProcessing.h"
#include "util.hpp"

void printMetaData(AVFormatContext *fmt_ctx, int vstream) {
    if (vstream >= 0) {
        AVCodecParameters *par = fmt_ctx->streams[vstream]->codecpar;
        std::cout << "width: " << par->width << "\n";
        std::cout << "height: " << par->height << "\n";
        std::cout << "codec id: " << par->codec_id << "\n";
        std::cout << "bitrate: " << par->bit_rate << "\n";
        AVRational fps = fmt_ctx->streams[vstream]->avg_frame_rate;
        if (fps.num && fps.den)
            std::cout << "fps: " << 1.0 * fps.num / fps.den << "\n";
        std::cout << "Extradata size: " << par->extradata_size << "\n";
    }

    // Общие метаданные файла
    if (fmt_ctx->metadata) {
        AVDictionaryEntry *tag = nullptr;
        while ((tag = av_dict_get(fmt_ctx->metadata, "", tag, AV_DICT_IGNORE_SUFFIX)))
            std::cout << tag->key << ": " << tag->value << "\n";
    }
}

void writeFrameToFile(AVPacket *op, AVStream *vs, int64_t count) {
    char name[64];
    snprintf(name, sizeof(name), "frame_%06lld%s.%s",
             (long long) ++count,
             (op->flags & AV_PKT_FLAG_KEY) ? "_key" : "",
             (vs->codecpar->codec_id == AV_CODEC_ID_H264) ? "h264" : "hevc");

    std::string filepath = std::string(OUTPUT_DIR) + name;
    FILE *f = fopen(filepath.c_str(), "wb");
    if (f) {
        fwrite(op->data, 1, op->size, f);
        fclose(f);
    }
}

int writeCodecAVPackets(std::string path) {
    AVFormatContext *ifmt = nullptr;
    if (avformat_open_input(&ifmt, path.c_str(), nullptr, nullptr) < 0) return 2;
    if (avformat_find_stream_info(ifmt, nullptr) < 0) return 3;

    int vstream = -1;
    for (unsigned i = 0; i < ifmt->nb_streams; ++i) {
        if (ifmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            vstream = (int) i;
            break;
        }
    }
    if (vstream < 0) return 4;

    printMetaData(ifmt, vstream);

    AVStream *vs = ifmt->streams[vstream];

    //CUDA init
    CUdevice dev;
    CUcontext ctx;
    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuDevicePrimaryCtxRetain(&ctx, dev);

    int w = vs->codecpar->width, h = vs->codecpar->height;
    AVRational fpsr = vs->avg_frame_rate.num ? vs->avg_frame_rate : vs->r_frame_rate;
    int fps = fpsr.num && fpsr.den ? (int) lrint((double) fpsr.num / fpsr.den) : 25;

    NVENCEncoder encoder;
    if (!encoder.init(ctx, w, h, fps, 1)) {
        std::cerr << "NVENC init failed\n";
        return 8;
    }
    encoder.reconfigureBitrate(8'000'000, 12'000'000, fps);

    MkvMuxer muxer;
    // MP4Muxer mp4;
    if (!muxer.open("C:/Users/user/Desktop/VideoFiltering/videos/out.mkv", w, h, fps)) {
        std::cerr << "Muxer open failed\n";
        return 9;
    }
    // mp4.open("C:/Users/user/Desktop/VideoFiltering/videos/out.mp4", w, h, fps);
    CuDecode nvdec(ctx, vs->codecpar->width, vs->codecpar->height);
    // nvdec.attachMP4(&mp4);
    nvdec.attachEncoderMux(&encoder, &muxer);

    nvdec.setFrameCallback([&](const std::vector<uint8_t> &rgb, int w, int h, int64_t pts) {
        char fname[256];
        snprintf(fname, sizeof(fname), "%sframe_%06lld.jpg", OUTPUT_DIR, pts);
        if (!saveImage(fname, w, h, 3, rgb.data())) std::cerr << "Failed to save: " << fname << "\n";
        std::cout << "Saved: " << fname << std::endl;
    });

    // Optional: set up bitstream filter for Annex B if H.264/H.265 in MP4/MKV
    const AVBitStreamFilter *bsf_def = nullptr;
    if (vs->codecpar->codec_id == AV_CODEC_ID_H264) bsf_def = av_bsf_get_by_name("h264_mp4toannexb");
    else if (vs->codecpar->codec_id == AV_CODEC_ID_HEVC)
        bsf_def = av_bsf_get_by_name("hevc_mp4toannexb");


    AVBSFContext *bsf_ctx = nullptr;
    if (bsf_def) {
        if (av_bsf_alloc(bsf_def, &bsf_ctx) < 0) return 5;
        if (avcodec_parameters_copy(bsf_ctx->par_in, vs->codecpar) < 0) return 6;
        bsf_ctx->time_base_in = vs->time_base;
        if (av_bsf_init(bsf_ctx) < 0) return 7;
    }

    std::cout << "Extradata size (init): " << vs->codecpar->extradata_size << "\n";
    if (bsf_ctx)
        std::cout << "Extradata size (after bsf): " << bsf_ctx->par_out->extradata_size << "\n";


    AVPacket *pkt = av_packet_alloc();
    int64_t count = 0;
    //NVDEC INIT
    nvdec.init((vs->codecpar->codec_id == AV_CODEC_ID_H264) ? cudaVideoCodec_H264 : cudaVideoCodec_HEVC);
    CUcontext cur;
    cuCtxSetCurrent(ctx);
    cuCtxGetCurrent(&cur);
    std::cout << "CUDA ctx active: " << (cur != nullptr) << std::endl;

    while (av_read_frame(ifmt, pkt) >= 0) {
        if (pkt->stream_index != vstream) {
            av_packet_unref(pkt);
            continue;
        }

        if (bsf_ctx) {
            if (av_bsf_send_packet(bsf_ctx, pkt) < 0) {
                av_packet_unref(pkt);
                break;
            }
            av_packet_unref(pkt);
            while (true) {
                AVPacket *op = av_packet_alloc();
                int r = av_bsf_receive_packet(bsf_ctx, op);
                if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) {
                    av_packet_free(&op);
                    break;
                }

                //writeFrameToFile(op, vs, count);
                nvdec.decodePacket(op);

                av_packet_free(&op);
            }
        }
    }
    if (bsf_ctx) {
        // signal EOF to BSF
        av_bsf_send_packet(bsf_ctx, nullptr);
        while (true) {
            AVPacket *op = av_packet_alloc();
            int r = av_bsf_receive_packet(bsf_ctx, op);
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) {
                av_packet_free(&op);
                break;
            }
            nvdec.decodePacket(op);
            av_packet_free(&op);
        }
    }
    nvdec.decodePacket(nullptr);

    av_packet_free(&pkt);
    avformat_close_input(&ifmt);
    cuDevicePrimaryCtxRelease(dev);
    muxer.close();
    // mp4.close();
    encoder.destroy();
    return 0;
}

int main() {
    //helper funcs
    std::string filepath = std::string(OUTPUT_DIR);
    std::cout << "output dir: " << filepath << std::endl;
    createDirectoryIfNotExists(filepath);

    int rc = writeCodecAVPackets("C:/Users/user/Desktop/VideoFiltering/videos/dr.mp4");

    return EXIT_SUCCESS;
}
