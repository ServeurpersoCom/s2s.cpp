// test-silero.cpp: speech probability dump for the parity harness
//
// Reads a WAV, downmixes to mono, resamples to the model rate, then walks
// the file window by window and writes one probability per line. The
// resampled signal is dumped as raw float32 next to the probabilities, so
// test-silero.py feeds the ONNX reference the exact same samples and the
// comparison measures the model, not two resamplers.

#include "audio-resample.h"
#include "silero.h"
#include "timer.h"
#include "version.h"
#include "wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "s2s.cpp %s\n\n", S2S_VERSION);
    fprintf(stderr,
            "Usage: %s <gguf> <wav> <out-prefix>\n"
            "\n"
            "Writes <out-prefix>-probs.txt, one speech probability per window,\n"
            "and <out-prefix>-pcm.f32, the resampled mono signal the model saw.\n",
            prog);
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }
    const char *      model_path = argv[1];
    const char *      wav_path   = argv[2];
    const std::string prefix     = argv[3];
    const std::string out_path   = prefix + "-probs.txt";
    const std::string pcm_path   = prefix + "-pcm.f32";

    FILE * fp = fopen(wav_path, "rb");
    if (!fp) {
        fprintf(stderr, "[Silero] FATAL: cannot open %s\n", wav_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    const long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> raw((size_t) size);
    if (fread(raw.data(), 1, raw.size(), fp) != raw.size()) {
        fprintf(stderr, "[Silero] FATAL: short read on %s\n", wav_path);
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

    sv_context * vad = sv_init(model_path, 1);
    if (!vad) {
        fprintf(stderr, "[Silero] FATAL: %s\n", sv_last_error());
        return 1;
    }

    const int target = sv_sample_rate(vad);
    if (rate != target) {
        int     n_out = 0;
        float * r     = audio_resample(mono.data(), n_frames, rate, target, 1, &n_out);
        if (!r) {
            fprintf(stderr, "[Silero] FATAL: resample %d -> %d failed\n", rate, target);
            sv_free(vad);
            return 1;
        }
        mono.assign(r, r + n_out);
        free(r);
        n_frames = n_out;
    }

    {
        FILE * dump = fopen(pcm_path.c_str(), "wb");
        if (!dump) {
            fprintf(stderr, "[Silero] FATAL: cannot write %s\n", pcm_path.c_str());
            sv_free(vad);
            return 1;
        }
        fwrite(mono.data(), sizeof(float), (size_t) n_frames, dump);
        fclose(dump);
    }

    sv_state * state = sv_state_new(vad);
    if (!state) {
        fprintf(stderr, "[Silero] FATAL: %s\n", sv_last_error());
        sv_free(vad);
        return 1;
    }

    const int window = sv_window(vad);
    FILE *    out    = fopen(out_path.c_str(), "w");
    if (!out) {
        fprintf(stderr, "[Silero] FATAL: cannot write %s\n", out_path.c_str());
        sv_state_free(state);
        sv_free(vad);
        return 1;
    }

    int   n_windows = 0;
    Timer t_scan;
    for (int off = 0; off + window <= n_frames; off += window) {
        const float p = sv_prob(state, mono.data() + off, window);
        if (p < 0.0f) {
            fprintf(stderr, "[Silero] FATAL: %s\n", sv_last_error());
            break;
        }
        fprintf(out, "%.9f\n", (double) p);
        n_windows++;
    }

    const double scan_ms   = t_scan.ms();
    const double audio_sec = (double) n_frames / target;

    fclose(out);
    fprintf(stderr, "[Silero] Scored %d windows over %.2fs of audio\n", n_windows, audio_sec);
    fprintf(stderr, "[Perf] VAD %.1f ms (%.3f ms/window, RTF %.5f)\n", scan_ms,
            n_windows > 0 ? scan_ms / n_windows : 0.0, (scan_ms / 1000.0) / audio_sec);

    sv_state_free(state);
    sv_free(vad);
    return 0;
}
