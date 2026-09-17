// test-server.cpp: one conversation over the Realtime WebSocket
//
// Plays a WAV into a running s2s-server the way a browser would: 24 kHz
// PCM16, base64, 20 ms per frame, in real time so the turn detection sees the
// silences it needs. Prints one line per server event and the audio it got
// back, then reports the timings.
//
// The server side is expected in loopback mode, which keeps the language
// model out of the measurement.

#include "audio-resample.h"
#include "httplib.h"
#include "realtime-proto.h"
#include "timer.h"
#include "version.h"
#include "wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define FRAME_MS 20

static void print_usage(const char * prog) {
    fprintf(stderr, "s2s.cpp %s\n\n", S2S_VERSION);
    fprintf(stderr,
            "Usage: %s <url> <wav> [seconds]\n"
            "\n"
            "Streams the WAV to a Realtime endpoint and prints the events.\n"
            "seconds bounds how much audio is sent, 0 sends the whole file.\n",
            prog);
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string url      = argv[1];
    const char *      wav_path = argv[2];
    const double      limit    = argc == 4 ? atof(argv[3]) : 0.0;

    FILE * fp = fopen(wav_path, "rb");
    if (!fp) {
        fprintf(stderr, "[Client] FATAL: cannot open %s\n", wav_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    const long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> raw((size_t) size);
    if (fread(raw.data(), 1, raw.size(), fp) != raw.size()) {
        fprintf(stderr, "[Client] FATAL: short read on %s\n", wav_path);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    int     n_frames = 0;
    int     rate     = 0;
    float * stereo   = read_wav_buf(raw.data(), raw.size(), &n_frames, &rate);
    if (!stereo) {
        return 1;
    }
    std::vector<float> mono((size_t) n_frames);
    for (int i = 0; i < n_frames; i++) {
        mono[(size_t) i] = 0.5f * (stereo[i * 2] + stereo[i * 2 + 1]);
    }
    free(stereo);

    // The protocol carries 24 kHz, whatever the file holds.
    if (rate != SAMPLE_RATE_24K) {
        int     n_out = 0;
        float * r     = audio_resample(mono.data(), n_frames, rate, SAMPLE_RATE_24K, 1, &n_out);
        if (!r) {
            fprintf(stderr, "[Client] FATAL: resample %d -> %d failed\n", rate, SAMPLE_RATE_24K);
            return 1;
        }
        mono.assign(r, r + n_out);
        free(r);
        n_frames = n_out;
    }

    if (limit > 0.0) {
        const int keep = (int) (limit * SAMPLE_RATE_24K);
        if (keep < n_frames) {
            mono.resize((size_t) keep);
            n_frames = keep;
        }
    }

    httplib::ws::WebSocketClient client(url);
    if (!client.is_valid() || !client.connect()) {
        fprintf(stderr, "[Client] FATAL: cannot connect to %s\n", url.c_str());
        return 1;
    }

    Timer  timer;
    size_t audio_bytes = 0;
    int    n_events    = 0;
    bool   running     = true;

    std::thread reader([&]() {
        std::string frame;
        while (running) {
            if (client.read(frame) == httplib::ws::ReadResult::Fail) {
                break;
            }

            yyjson_doc * doc = yyjson_read(frame.c_str(), frame.size(), 0);
            if (!doc) {
                continue;
            }
            yyjson_val *      root = yyjson_doc_get_root(doc);
            const std::string type = rt_json_str(root, "type");

            n_events++;

            if (type == "response.output_audio.delta") {
                audio_bytes += rt_json_str(root, "delta").size();
            } else if (type == "conversation.item.input_audio_transcription.completed") {
                printf("[Event] %7.2fs  %-46s \"%s\"\n", timer.ms() / 1000.0, type.c_str(),
                       rt_json_str(root, "transcript").c_str());
            } else if (type == "response.output_audio_transcript.delta") {
                printf("[Event] %7.2fs  %-46s \"%s\"\n", timer.ms() / 1000.0, type.c_str(),
                       rt_json_str(root, "delta").c_str());
            } else if (type == "error") {
                yyjson_val * error = yyjson_obj_get(root, "error");
                printf("[Event] %7.2fs  %-46s %s\n", timer.ms() / 1000.0, type.c_str(),
                       error ? rt_json_str(error, "message").c_str() : "");
            } else {
                printf("[Event] %7.2fs  %s\n", timer.ms() / 1000.0, type.c_str());
            }
            fflush(stdout);
            yyjson_doc_free(doc);
        }
    });

    client.send(std::string("{\"type\":\"session.update\",\"session\":{\"mode\":\"loopback\"}}"));

    const int frame_samples = FRAME_MS * SAMPLE_RATE_24K / 1000;
    for (int off = 0; off < n_frames; off += frame_samples) {
        const int n = std::min(frame_samples, n_frames - off);

        const std::string message = "{\"type\":\"input_audio_buffer.append\",\"audio\":\"" +
                                    rt_float_to_pcm16_base64(mono.data() + off, (size_t) n) + "\"}";
        if (!client.send(message)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(FRAME_MS));
    }

    // Let the last turn finish: the server is still recognizing and speaking
    // when the audio ends.
    client.send(std::string("{\"type\":\"input_audio_buffer.commit\"}"));
    std::this_thread::sleep_for(std::chrono::seconds(8));

    running = false;
    client.close();
    reader.join();

    const double audio_sec = (double) audio_bytes * 3.0 / 4.0 / 2.0 / SAMPLE_RATE_24K;
    printf("[Client] %d events, %.2fs of audio received over %.2fs of streaming\n", n_events, audio_sec,
           timer.ms() / 1000.0);
    return 0;
}
