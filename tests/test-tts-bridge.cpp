// test-tts-bridge.cpp: streaming synthesis and barge-in over the bridge
//
// Speaks two units. The first runs to the end and is written to a WAV, which
// measures the time to first audio and the real time factor. The second is
// cancelled after a few chunks, the way a barge-in does, and the harness
// reports how much audio was produced before the floor was released.

#include "audio-io.h"
#include "timer.h"
#include "tts-bridge.h"
#include "version.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "s2s.cpp %s\n\n", S2S_VERSION);
    fprintf(stderr,
            "Usage: %s <talker-gguf> <codec-gguf> <voices-dir> <out-prefix>\n"
            "\n"
            "Speaks with the first voice of <voices-dir>, writes <out-prefix>-speech.wav\n"
            "and prints the streaming timings.\n",
            prog);
}

struct Capture {
    std::vector<float>  pcm;
    Timer               timer;
    double              first_chunk_ms = -1.0;
    int                 n_chunks       = 0;
    int                 cancel_after   = 0;  // 0 keeps every chunk
    std::atomic<bool> * cancel         = nullptr;
};

static bool on_chunk(const float * pcm, size_t n_samples, void * user) {
    Capture * capture = (Capture *) user;

    if (capture->first_chunk_ms < 0.0) {
        capture->first_chunk_ms = capture->timer.ms();
    }
    capture->pcm.insert(capture->pcm.end(), pcm, pcm + n_samples);
    capture->n_chunks++;

    if (capture->cancel_after > 0 && capture->n_chunks >= capture->cancel_after) {
        capture->cancel->store(true);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        print_usage(argv[0]);
        return 1;
    }

    tts_request       request;
    tts_bridge_params params;
    params.talker_path    = argv[1];
    params.codec_path     = argv[2];
    params.voices_dir     = argv[3];
    // Fixed seed: a harness that samples a different voice path on every run
    // cannot tell a regression from a draw.
    params.sampling.seed  = 42;
    request.sampling.seed = 42;

    const std::string prefix = argv[4];

    tts_bridge * bridge = tts_bridge_load(params);
    if (!bridge) {
        fprintf(stderr, "[TTS] FATAL: %s\n", tts_bridge_last_error());
        return 1;
    }

    const int rate = tts_bridge_sample_rate(bridge);

    // Full unit.
    std::atomic<bool> cancel(false);
    Capture           full;
    full.cancel = &cancel;

    Timer        t_full;
    const bool   ok = tts_bridge_speak(bridge, "Different cultures have their own creation story.", request, on_chunk,
                                       &full, &cancel);
    const double full_ms = t_full.ms();
    if (!ok) {
        fprintf(stderr, "[TTS] FATAL: %s\n", tts_bridge_last_error());
        tts_bridge_free(bridge);
        return 1;
    }

    const double audio_sec = (double) full.pcm.size() / rate;
    fprintf(stderr, "[TTS] Spoke %.2fs in %d chunks\n", audio_sec, full.n_chunks);
    fprintf(stderr, "[Perf] TTFA %.1f ms, total %.1f ms, RTF %.4f\n", full.first_chunk_ms, full_ms,
            (full_ms / 1000.0) / audio_sec);

    // The encoder comes from the submodule, which already writes the WAV
    // flavor the codec output deserves.
    const std::string wav_path = prefix + "-speech.wav";
    const std::string encoded  = audio_encode_wav_s16(full.pcm.data(), (int) full.pcm.size(), rate);
    FILE *            wav      = fopen(wav_path.c_str(), "wb");
    if (!wav || fwrite(encoded.data(), 1, encoded.size(), wav) != encoded.size()) {
        fprintf(stderr, "[TTS] FATAL: cannot write %s\n", wav_path.c_str());
        tts_bridge_free(bridge);
        return 1;
    }
    fclose(wav);

    // Barge-in: the flag goes up after the third chunk.
    cancel.store(false);
    Capture interrupted;
    interrupted.cancel       = &cancel;
    interrupted.cancel_after = 3;

    Timer      t_cancel;
    const bool spoke = tts_bridge_speak(bridge, "This sentence is interrupted long before it ends, mid flow.", request,
                                        on_chunk, &interrupted, &cancel);
    const double cancel_ms = t_cancel.ms();

    fprintf(stderr, "[TTS] Barge-in after %d chunks, %.2fs of audio, returned %s\n", interrupted.n_chunks,
            (double) interrupted.pcm.size() / rate, spoke ? "true" : "false");
    fprintf(stderr, "[Perf] Cancel %.1f ms\n", cancel_ms);

    tts_bridge_free(bridge);
    return spoke ? 1 : 0;
}
