// test-server.cpp: one conversation over the Realtime WebSocket
//
// Plays a WAV into a running s2s-server the way a browser would: 24 kHz
// PCM16, base64, 20 ms per frame, in real time so the turn detection sees the
// silences it needs. Prints one line per server event and the audio it got
// back, then reports the timings.
//
// The session runs in loopback, which keeps the language model out of the
// measurement.
//
// With --room the client is a laptop on speakers. It plays what it receives
// in real time, and its microphone hears that playback back through a room:
// delayed, smeared by a decaying response, added to the speaker's voice. It
// sends the played audio as the reference and asks for the server echo
// canceller. The first seconds of the file start a conversation, then the
// speaker talks over an answer with a later passage of the file, one second
// into its playback and with two more seconds of it left to hear. Like the
// browser, it drops its playback on speech_started.
//
// With --no-endpoint the session is a conversation with an endpoint that
// cannot be reached: every response it opens has to end anyway.

#include "audio-resample.h"
#include "httplib.h"
#include "realtime-proto.h"
#include "timer.h"
#include "version.h"
#include "wav.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define FRAME_MS 20

#define ROOM_DELAY_MS       120   // loudspeaker to microphone
#define ROOM_TAIL_MS        50    // decaying room response
#define ROOM_ECHO_GAIN      0.8f  // the echo about as loud as the voice
#define ROOM_TALK_OVER_MS   1000  // playback of the answer heard before the speaker talks over it
#define ROOM_TALK_OVER_LEFT 2000  // playback of the answer still to come, in ms
#define ROOM_TALK_OVER_FROM 12.0  // seconds into the file the interruption starts
#define ROOM_TALK_OVER_SEC  3.0
#define ROOM_SETTLE_SEC     6.0   // silence streamed after the interruption
#define ROOM_WAIT_SEC       20.0  // longest wait for an answer long enough to talk over

static void print_usage(const char * prog) {
    fprintf(stderr, "s2s.cpp %s\n\n", S2S_VERSION);
    fprintf(stderr,
            "Usage: %s <url> <wav> [seconds] [--room | --no-endpoint]\n"
            "\n"
            "Streams the WAV to a Realtime endpoint and prints the events.\n"
            "seconds bounds how much audio is sent, 0 sends the whole file.\n"
            "--room plays the answers into the microphone through a simulated\n"
            "room and talks over one of them, with the server echo canceller.\n"
            "--no-endpoint asks for a conversation with an unreachable endpoint.\n",
            prog);
}

// The loudspeaker: what the server sent and has not been played yet, and
// everything it played, which the room turns into echo.
struct Speaker {
    std::mutex         mutex;
    std::deque<float>  queue;
    std::vector<float> played;
};

int main(int argc, char ** argv) {
    bool                      room        = false;
    bool                      no_endpoint = false;
    std::vector<const char *> args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--room") == 0) {
            room = true;
        } else if (strcmp(argv[i], "--no-endpoint") == 0) {
            no_endpoint = true;
        } else {
            args.push_back(argv[i]);
        }
    }
    if (args.size() < 2 || args.size() > 3) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string url      = args[0];
    const char *      wav_path = args[1];
    const double      limit    = args.size() == 3 ? atof(args[2]) : 0.0;

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

    // The passage the speaker talks over the answer with, taken before the
    // file is cut to the opening.
    std::vector<float> talk_over;
    if (room) {
        const size_t from = (size_t) (ROOM_TALK_OVER_FROM * SAMPLE_RATE_24K);
        const size_t to   = from + (size_t) (ROOM_TALK_OVER_SEC * SAMPLE_RATE_24K);
        if (to > mono.size()) {
            fprintf(stderr, "[Client] FATAL: %s is too short for the room scenario\n", wav_path);
            return 1;
        }
        talk_over.assign(mono.begin() + (ptrdiff_t) from, mono.begin() + (ptrdiff_t) to);
    }

    if (limit > 0.0) {
        const int keep = (int) (limit * SAMPLE_RATE_24K);
        if (keep < n_frames) {
            mono.resize((size_t) keep);
            n_frames = keep;
        }
    }

    // A fixed decaying room response, normalized so the echo level only
    // depends on the gain.
    std::vector<float> room_ir((size_t) (ROOM_TAIL_MS * SAMPLE_RATE_24K / 1000));
    {
        uint32_t seed = 1;
        double   sum  = 0.0;
        for (size_t k = 0; k < room_ir.size(); k++) {
            seed = seed * 1664525u + 1013904223u;
            const double tap =
                ((double) (seed >> 8) / (double) (1u << 24) - 0.5) * std::exp(-(double) k / (0.01 * SAMPLE_RATE_24K));
            room_ir[k] = (float) tap;
            sum += std::fabs(tap);
        }
        for (float & tap : room_ir) {
            tap = (float) (tap / sum);
        }
    }
    const size_t room_delay = (size_t) (ROOM_DELAY_MS * SAMPLE_RATE_24K / 1000);
    Speaker      speaker;

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
                const std::string delta = rt_json_str(root, "delta");
                audio_bytes += delta.size();
                if (room) {
                    std::vector<float> pcm;
                    rt_pcm16_to_float(rt_base64_decode(delta.c_str(), delta.size()), pcm);
                    std::lock_guard<std::mutex> lock(speaker.mutex);
                    speaker.queue.insert(speaker.queue.end(), pcm.begin(), pcm.end());
                }
            } else if (type == "conversation.item.input_audio_transcription.completed") {
                printf("[Event] %7.2fs  %-46s %s \"%s\"\n", timer.ms() / 1000.0, type.c_str(),
                       rt_json_str(root, "item_id").c_str(), rt_json_str(root, "transcript").c_str());
            } else if (type == "response.output_audio_transcript.delta") {
                printf("[Event] %7.2fs  %-46s \"%s\"\n", timer.ms() / 1000.0, type.c_str(),
                       rt_json_str(root, "delta").c_str());
            } else if (type == "error") {
                yyjson_val * error = yyjson_obj_get(root, "error");
                printf("[Event] %7.2fs  %-46s %s\n", timer.ms() / 1000.0, type.c_str(),
                       error ? rt_json_str(error, "message").c_str() : "");
            } else {
                printf("[Event] %7.2fs  %s\n", timer.ms() / 1000.0, type.c_str());
                // The browser drops what it has not played yet once the user
                // takes the floor.
                if (room && type == "input_audio_buffer.speech_started") {
                    std::lock_guard<std::mutex> lock(speaker.mutex);
                    if (!speaker.queue.empty()) {
                        printf("[Client] %7.2fs  playback flushed\n", timer.ms() / 1000.0);
                        speaker.queue.clear();
                    }
                }
            }
            fflush(stdout);
            yyjson_doc_free(doc);
        }
    });

    const char * session =
        room        ? "{\"type\":\"session.update\",\"session\":{\"mode\":\"loopback\",\"echo\":\"server\"}}" :
        no_endpoint ? "{\"type\":\"session.update\",\"session\":{\"mode\":\"conversation\",\"llm_url\":\"nowhere\"}}" :
                      "{\"type\":\"session.update\",\"session\":{\"mode\":\"loopback\"}}";
    client.send(std::string(session));

    const int frame_samples = FRAME_MS * SAMPLE_RATE_24K / 1000;
    if (!room) {
        for (int off = 0; off < n_frames; off += frame_samples) {
            const int n = std::min(frame_samples, n_frames - off);

            const std::string message = "{\"type\":\"input_audio_buffer.append\",\"audio\":\"" +
                                        rt_float_to_pcm16_base64(mono.data() + off, (size_t) n) + "\"}";
            if (!client.send(message)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(FRAME_MS));
        }
    } else {
        // One frame at a time: the voice for this frame, what the speaker
        // plays during it, and the microphone that hears both.
        const size_t       frame = (size_t) frame_samples;
        std::vector<float> voice(frame), playback(frame), mic(frame);
        size_t             opening      = 0;     // samples of the opening sent
        size_t             talked       = 0;     // samples of the interruption sent
        int                played_ms    = 0;     // playback of the current answer heard so far
        size_t             left         = 0;     // samples of playback still queued
        bool               playing      = false;
        double             settle_until = -1.0;  // seconds, once the interruption is over
        double             opened       = -1.0;  // seconds, once the opening is sent

        for (;;) {
            const double now = timer.ms() / 1000.0;

            std::fill(voice.begin(), voice.end(), 0.0f);
            if (opening < mono.size()) {
                const size_t n = std::min(frame, mono.size() - opening);
                std::copy(mono.begin() + (ptrdiff_t) opening, mono.begin() + (ptrdiff_t) (opening + n), voice.begin());
                opening += n;
            } else if (talked < talk_over.size() &&
                       (talked > 0 || (played_ms >= ROOM_TALK_OVER_MS &&
                                       left >= (size_t) (ROOM_TALK_OVER_LEFT * SAMPLE_RATE_24K / 1000)))) {
                if (talked == 0) {
                    printf("[Client] %7.2fs  talking over the answer\n", now);
                }
                const size_t n = std::min(frame, talk_over.size() - talked);
                std::copy(talk_over.begin() + (ptrdiff_t) talked, talk_over.begin() + (ptrdiff_t) (talked + n),
                          voice.begin());
                talked += n;
                if (talked == talk_over.size()) {
                    printf("[Client] %7.2fs  done talking over\n", now);
                    settle_until = now + ROOM_SETTLE_SEC;
                }
            }

            bool any = false;
            {
                std::lock_guard<std::mutex> lock(speaker.mutex);
                for (size_t i = 0; i < frame; i++) {
                    playback[i] = 0.0f;
                    if (!speaker.queue.empty()) {
                        playback[i] = speaker.queue.front();
                        speaker.queue.pop_front();
                        any = true;
                    }
                }
                speaker.played.insert(speaker.played.end(), playback.begin(), playback.end());
                left = speaker.queue.size();

                const size_t end = speaker.played.size();
                for (size_t i = 0; i < frame; i++) {
                    const size_t t    = end - frame + i;
                    float        echo = 0.0f;
                    for (size_t k = 0; k < room_ir.size() && t >= room_delay + k; k++) {
                        echo += room_ir[k] * speaker.played[t - room_delay - k];
                    }
                    mic[i] = voice[i] + ROOM_ECHO_GAIN * echo;
                }
            }

            if (any && !playing) {
                printf("[Client] %7.2fs  playback started\n", now);
            }
            playing   = any;
            played_ms = any && opening >= mono.size() ? played_ms + FRAME_MS : 0;

            std::string message = "{\"type\":\"input_audio_buffer.append\",\"audio\":\"" +
                                  rt_float_to_pcm16_base64(mic.data(), frame) + "\"";
            if (any) {
                message += ",\"reference\":\"" + rt_float_to_pcm16_base64(playback.data(), frame) + "\"";
            }
            message += "}";
            if (!client.send(message)) {
                break;
            }
            if (settle_until > 0.0 && now >= settle_until) {
                break;
            }
            if (opened < 0.0 && opening >= mono.size()) {
                opened = now;
            }
            if (talked == 0 && opened >= 0.0 && now - opened > ROOM_WAIT_SEC) {
                printf("[Client] %7.2fs  no answer long enough to talk over\n", now);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(FRAME_MS));
        }
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
