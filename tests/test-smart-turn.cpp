// test-smart-turn.cpp: completion probability dump for the parity harness
//
// Reads a WAV, downmixes to mono, resamples to the model rate, then scores
// a sliding series of turn candidates: prefixes of the file that grow one
// second at a time. Every candidate is one line of the dump, and the
// resampled signal goes out as raw float32 so test-smart-turn.py feeds the
// ONNX reference the exact same samples.

#include "audio-resample.h"
#include "smart-turn.h"
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
            "Scores growing one second prefixes of the file as turn candidates.\n"
            "Writes <out-prefix>-probs.txt, one completion probability per\n"
            "candidate, and <out-prefix>-pcm.f32, the signal the model saw.\n",
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
        fprintf(stderr, "[Turn] FATAL: cannot open %s\n", wav_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    const long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> raw((size_t) size);
    if (fread(raw.data(), 1, raw.size(), fp) != raw.size()) {
        fprintf(stderr, "[Turn] FATAL: short read on %s\n", wav_path);
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

    st_context * turn = st_init(model_path, 0);
    if (!turn) {
        fprintf(stderr, "[Turn] FATAL: %s\n", st_last_error());
        return 1;
    }

    const int target = st_sample_rate(turn);
    if (rate != target) {
        int     n_out = 0;
        float * r     = audio_resample(mono.data(), n_frames, rate, target, 1, &n_out);
        if (!r) {
            fprintf(stderr, "[Turn] FATAL: resample %d -> %d failed\n", rate, target);
            st_free(turn);
            return 1;
        }
        mono.assign(r, r + n_out);
        free(r);
        n_frames = n_out;
    }

    {
        FILE * dump = fopen(pcm_path.c_str(), "wb");
        if (!dump) {
            fprintf(stderr, "[Turn] FATAL: cannot write %s\n", pcm_path.c_str());
            st_free(turn);
            return 1;
        }
        fwrite(mono.data(), sizeof(float), (size_t) n_frames, dump);
        fclose(dump);
    }

    FILE * out = fopen(out_path.c_str(), "w");
    if (!out) {
        fprintf(stderr, "[Turn] FATAL: cannot write %s\n", out_path.c_str());
        st_free(turn);
        return 1;
    }

    int   n_candidates = 0;
    Timer t_scan;
    for (int end = target; end <= n_frames; end += target) {
        const float p = st_predict(turn, mono.data(), end);
        if (p < 0.0f) {
            fprintf(stderr, "[Turn] FATAL: %s\n", st_last_error());
            break;
        }
        fprintf(out, "%.9f\n", (double) p);
        n_candidates++;
    }

    const double scan_ms = t_scan.ms();

    fclose(out);
    fprintf(stderr, "[Turn] Scored %d candidates over %.2fs of audio\n", n_candidates, (double) n_frames / target);
    fprintf(stderr, "[Perf] Turn %.1f ms (%.1f ms/candidate)\n", scan_ms,
            n_candidates > 0 ? scan_ms / n_candidates : 0.0);

    st_free(turn);
    return 0;
}
