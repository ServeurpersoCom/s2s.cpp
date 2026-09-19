// test-session.cpp: turn timeline of the state machine over a file
//
// Feeds a WAV to the session in 20 ms chunks, the way a client would, and
// prints one line per event. Every committed turn goes straight to the
// recognizer, so the output is the conversation as the server would see it:
// when speech opened, when the classifier reopened a turn, and what each
// committed turn said.

#include "audio-resample.h"
#include "parakeet.h"
#include "s2s-session.h"
#include "silero.h"
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
            "Usage: %s --vad <gguf> --turn <gguf> --asr <gguf> --file <wav> [options]\n"
            "\n"
            "Required:\n"
            "  --vad <gguf>           Silero VAD model file\n"
            "  --turn <gguf>          Smart Turn model file\n"
            "  --asr <gguf>           Parakeet TDT model file\n"
            "  --file <wav>           Input audio, any rate, mono or stereo\n"
            "\n"
            "Optional:\n"
            "  --out <path>           Write the event timeline to a file as well\n"
            "  --cpu                  Force the CPU backend for the recognizer\n"
            "  --chunk <ms>           Client chunk size (default: 20)\n",
            prog);
}

struct SessionTap {
    pk_context * asr     = nullptr;
    FILE *       out     = nullptr;
    int          n_turns = 0;
    double       asr_ms  = 0.0;
};

static const char * event_name(s2s_session_event event) {
    switch (event) {
        case S2S_EVENT_SPEECH_STARTED:
            return "speech_started";
        case S2S_EVENT_SPEECH_STOPPED:
            return "speech_stopped";
        case S2S_EVENT_TURN_REOPENED:
            return "turn_reopened";
        case S2S_EVENT_TURN_COMMITTED:
            return "turn_committed";
        case S2S_EVENT_TURN_FINAL:
            return "turn_final";
        case S2S_EVENT_BARGE_IN:
            return "barge_in";
    }
    return "unknown";
}

static void emit(const s2s_session_report * report, void * user) {
    SessionTap * tap = (SessionTap *) user;

    char line[4096];
    int  n = snprintf(line, sizeof(line), "[Turn] %7.2fs  turn %d rev %d  %-14s", report->time_sec, report->turn_id,
                      report->revision, event_name(report->event));

    if (report->event == S2S_EVENT_TURN_REOPENED) {
        n += snprintf(line + n, sizeof(line) - (size_t) n, "  score %.3f", (double) report->turn_score);
    }

    if (report->event == S2S_EVENT_TURN_COMMITTED) {
        Timer                t_asr;
        pk_transcribe_params params = pk_transcribe_default_params();
        char *               text   = nullptr;

        const pk_status status = pk_transcribe(tap->asr, report->pcm, report->n_samples, 16000, &params, &text);
        tap->asr_ms += t_asr.ms();
        tap->n_turns++;

        if (status != PK_STATUS_OK) {
            n += snprintf(line + n, sizeof(line) - (size_t) n, "  FATAL: %s", pk_last_error());
        } else {
            n += snprintf(line + n, sizeof(line) - (size_t) n, "  score %.3f  %.2fs  \"%s\"",
                          (double) report->turn_score, (double) report->n_samples / 16000.0, text);
            pk_free_text(text);
        }
    }

    printf("%s\n", line);
    if (tap->out) {
        fprintf(tap->out, "%s\n", line);
    }
}

int main(int argc, char ** argv) {
    const char * vad_path  = nullptr;
    const char * turn_path = nullptr;
    const char * asr_path  = nullptr;
    const char * wav_path  = nullptr;
    const char * out_path  = nullptr;
    int          chunk_ms  = 20;
    bool         use_gpu   = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--vad") == 0 && i + 1 < argc) {
            vad_path = argv[++i];
        } else if (strcmp(argv[i], "--turn") == 0 && i + 1 < argc) {
            turn_path = argv[++i];
        } else if (strcmp(argv[i], "--asr") == 0 && i + 1 < argc) {
            asr_path = argv[++i];
        } else if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
            wav_path = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) {
            chunk_ms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu") == 0) {
            use_gpu = false;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (!vad_path || !turn_path || !asr_path || !wav_path) {
        print_usage(argv[0]);
        return 1;
    }

    FILE * fp = fopen(wav_path, "rb");
    if (!fp) {
        fprintf(stderr, "[Session] FATAL: cannot open %s\n", wav_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    const long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> raw((size_t) size);
    if (fread(raw.data(), 1, raw.size(), fp) != raw.size()) {
        fprintf(stderr, "[Session] FATAL: short read on %s\n", wav_path);
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

    sv_context * vad = sv_init(vad_path, 1);
    if (!vad) {
        fprintf(stderr, "[Session] FATAL: %s\n", sv_last_error());
        return 1;
    }

    if (rate != sv_sample_rate(vad)) {
        int     n_out = 0;
        float * r     = audio_resample(mono.data(), n_frames, rate, sv_sample_rate(vad), 1, &n_out);
        if (!r) {
            fprintf(stderr, "[Session] FATAL: resample %d -> %d failed\n", rate, sv_sample_rate(vad));
            return 1;
        }
        mono.assign(r, r + n_out);
        free(r);
        n_frames = n_out;
    }

    st_context * turn = st_init(turn_path, 0);
    if (!turn) {
        fprintf(stderr, "[Session] FATAL: %s\n", st_last_error());
        return 1;
    }

    pk_init_params init = pk_init_default_params();
    init.model_path     = asr_path;
    init.use_gpu        = use_gpu;

    pk_context * asr = pk_init(&init);
    if (!asr) {
        fprintf(stderr, "[Session] FATAL: %s\n", pk_last_error());
        return 1;
    }

    SessionTap tap;
    tap.asr = asr;
    tap.out = out_path ? fopen(out_path, "w") : nullptr;
    if (out_path && !tap.out) {
        fprintf(stderr, "[Session] FATAL: cannot write %s\n", out_path);
        return 1;
    }

    const s2s_session_params params;
    s2s_session *            session = s2s_session_new(vad, turn, params, emit, &tap);
    if (!session) {
        fprintf(stderr, "[Session] FATAL: %s\n", sv_last_error());
        return 1;
    }

    const int chunk = chunk_ms * sv_sample_rate(vad) / 1000;

    Timer t_stream;
    for (int off = 0; off < n_frames; off += chunk) {
        const int n = std::min(chunk, n_frames - off);
        s2s_session_push(session, mono.data() + off, (size_t) n);
    }
    // The stream ends mid turn on a file, so the last turn is committed the
    // way a push to talk release or a disconnect would.
    s2s_session_commit_now(session);
    const double stream_ms = t_stream.ms();

    const double audio_sec = (double) n_frames / sv_sample_rate(vad);
    fprintf(stderr, "[Session] %d turns over %.2fs of audio\n", tap.n_turns, audio_sec);
    fprintf(stderr, "[Perf] Stream %.1f ms (%.1f ms listening, %.1f ms recognizing, RTF %.4f)\n", stream_ms,
            stream_ms - tap.asr_ms, tap.asr_ms, (stream_ms / 1000.0) / audio_sec);

    if (tap.out) {
        fclose(tap.out);
    }
    s2s_session_free(session);
    pk_free(asr);
    st_free(turn);
    sv_free(vad);
    return 0;
}
