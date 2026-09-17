// parakeet-transcribe.cpp: file to text on the command line
//
// Reads a WAV of any rate and channel count, downmixes, resamples, and prints
// the transcript. With --dump it writes the stage tensors the parity harness
// compares against the transformers reference.

#include "audio-resample.h"
#include "parakeet.h"
#include "pipeline-asr.h"
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
            "Usage: %s --model <gguf> --file <wav> [options]\n"
            "\n"
            "Required:\n"
            "  --model <gguf>         Parakeet TDT model file\n"
            "  --file <wav>           Input audio, any rate, mono or stereo\n"
            "\n"
            "Optional:\n"
            "  --out <path>           Write the transcript to a file as well\n"
            "  --threads <N>          CPU thread count (default: physical cores)\n"
            "  --cpu                  Force the CPU backend\n"
            "  --stream               Print every piece as the transducer emits it\n"
            "\n"
            "Debug:\n"
            "  --dump <dir>           Dump the stage tensors as raw f32\n",
            prog);
}

static bool on_token(const char * chunk, void * user) {
    (void) user;
    fputs(chunk, stdout);
    fflush(stdout);
    return true;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * wav_path   = nullptr;
    const char * out_path   = nullptr;
    const char * dump_dir   = nullptr;
    int          n_threads  = 0;
    bool         use_gpu    = true;
    bool         stream     = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
            wav_path = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu") == 0) {
            use_gpu = false;
        } else if (strcmp(argv[i], "--stream") == 0) {
            stream = true;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (!model_path || !wav_path) {
        print_usage(argv[0]);
        return 1;
    }

    FILE * fp = fopen(wav_path, "rb");
    if (!fp) {
        fprintf(stderr, "[ASR] FATAL: cannot open %s\n", wav_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    const long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> raw((size_t) size);
    if (fread(raw.data(), 1, raw.size(), fp) != raw.size()) {
        fprintf(stderr, "[ASR] FATAL: short read on %s\n", wav_path);
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

    // The dump path needs the pipeline directly, the plain path goes through
    // the public ABI like any consumer would.
    std::string text;
    if (dump_dir) {
        pipeline_asr_params params;
        params.model_path = model_path;
        params.n_threads  = n_threads;
        params.use_gpu    = use_gpu;
        params.dump_dir   = dump_dir;

        pipeline_asr * pipeline = pipeline_asr_load(params);
        if (!pipeline) {
            fprintf(stderr, "[ASR] FATAL: %s\n", pk_last_error());
            return 1;
        }

        pk_transcribe_params run = pk_transcribe_default_params();
        run.on_token             = stream ? on_token : nullptr;

        const pk_status status = pipeline_asr_run(pipeline, mono.data(), mono.size(), rate, &run, text);
        pipeline_asr_free(pipeline);
        if (status != PK_STATUS_OK) {
            fprintf(stderr, "[ASR] FATAL: %s\n", pk_last_error());
            return 1;
        }
    } else {
        pk_init_params init = pk_init_default_params();
        init.model_path     = model_path;
        init.n_threads      = n_threads;
        init.use_gpu        = use_gpu;

        pk_context * ctx = pk_init(&init);
        if (!ctx) {
            fprintf(stderr, "[ASR] FATAL: %s\n", pk_last_error());
            return 1;
        }

        pk_transcribe_params run = pk_transcribe_default_params();
        run.on_token             = stream ? on_token : nullptr;

        char *          out    = nullptr;
        const pk_status status = pk_transcribe(ctx, mono.data(), mono.size(), rate, &run, &out);
        if (status != PK_STATUS_OK) {
            fprintf(stderr, "[ASR] FATAL: %s\n", pk_last_error());
            pk_free(ctx);
            return 1;
        }
        text = out;
        pk_free_text(out);
        pk_free(ctx);
    }

    if (stream) {
        fputc('\n', stdout);
    } else {
        printf("%s\n", text.c_str());
    }

    if (out_path) {
        FILE * out = fopen(out_path, "w");
        if (!out) {
            fprintf(stderr, "[ASR] FATAL: cannot write %s\n", out_path);
            return 1;
        }
        fprintf(out, "%s\n", text.c_str());
        fclose(out);
    }
    return 0;
}
