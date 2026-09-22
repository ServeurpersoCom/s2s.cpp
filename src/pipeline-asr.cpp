// pipeline-asr.cpp: the pcm -> text path
//
// Two graph families live here. The decoder pair, prediction and joint, has
// fixed shapes and is built once into its arena. The frontend and the encoder
// depend on the utterance length, so they are rebuilt per call into a scratch
// context and allocated through a persistent gallocr that grows with the
// longest utterance seen so far.
//
// Host work sits where GGML has no reduction to offer: the preemphasis and the
// zero padding before the transform, the per mel bin normalization that needs
// two passes over the spectrogram, and the sinusoid table, which is pure
// geometry of the frame count.

#include "pipeline-asr.h"

#include "audio-mel.h"
#include "audio-resample.h"
#include "backend.h"
#include "fastconformer-forward.h"
#include "ggml-alloc.h"
#include "ggml.h"
#include "graph-arena.h"
#include "parakeet-mel.h"
#include "sp-detok.h"
#include "tdt-decoder.h"
#include "timer.h"
#include "utf8.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>

#define ASR_GRAPH_NODES   16384
#define ASR_DECODER_NODES 512

struct pipeline_asr {
    ggml_backend_t backend = nullptr;

    ParakeetHParams        hp;
    ParakeetEncoderWeights encoder;
    ParakeetDecoderWeights decoder;
    SpDetok                sp;
    WeightCtx              wctx = {};

    AudioMelConfig    mel_cfg = {};
    AudioMelConstants mel_c;
    WeightCtx         const_wctx = {};

    struct ggml_tensor * hann      = nullptr;
    struct ggml_tensor * dft_real  = nullptr;
    struct ggml_tensor * dft_imag  = nullptr;
    struct ggml_tensor * mel_basis = nullptr;
    struct ggml_tensor * log_guard = nullptr;

    // The two decoder graphs live side by side for the whole session and each
    // keeps its own allocator: an allocator serves one graph at a time, and
    // allocating a second one through it would move the first one's memory.
    GraphArena           dec_arena     = {};
    ggml_gallocr_t       predict_alloc = nullptr;
    ggml_gallocr_t       joint_alloc   = nullptr;
    ParakeetPredictGraph predict;
    ParakeetJointGraph   joint;

    ggml_gallocr_t run_alloc = nullptr;

    std::string dump_dir;
    std::mutex  mutex;
};

// Uploads the Hann window, the two DFT matrices, the filterbank and the log
// guard. They never change, so they live next to the weights.
static bool pipeline_asr_load_constants(pipeline_asr * p) {
    audio_mel_compute_constants(p->mel_cfg, p->mel_c);

    wctx_init(&p->const_wctx, 5);
    const int n_fft  = p->mel_cfg.n_fft;
    const int n_freq = p->mel_c.n_freq;

    p->hann      = ggml_new_tensor_1d(p->const_wctx.ctx, GGML_TYPE_F32, n_fft);
    p->dft_real  = ggml_new_tensor_2d(p->const_wctx.ctx, GGML_TYPE_F32, n_fft, n_freq);
    p->dft_imag  = ggml_new_tensor_2d(p->const_wctx.ctx, GGML_TYPE_F32, n_fft, n_freq);
    p->mel_basis = ggml_new_tensor_2d(p->const_wctx.ctx, GGML_TYPE_F32, n_freq, p->mel_cfg.n_mels);
    p->log_guard = ggml_new_tensor_1d(p->const_wctx.ctx, GGML_TYPE_F32, 1);

    if (!wctx_alloc(&p->const_wctx, p->backend)) {
        return false;
    }

    const float guard = PARAKEET_LOG_GUARD;
    ggml_backend_tensor_set(p->hann, p->mel_c.hann.data(), 0, p->mel_c.hann.size() * sizeof(float));
    ggml_backend_tensor_set(p->dft_real, p->mel_c.dft_real.data(), 0, p->mel_c.dft_real.size() * sizeof(float));
    ggml_backend_tensor_set(p->dft_imag, p->mel_c.dft_imag.data(), 0, p->mel_c.dft_imag.size() * sizeof(float));
    ggml_backend_tensor_set(p->mel_basis, p->mel_c.mel_basis.data(), 0, p->mel_c.mel_basis.size() * sizeof(float));
    ggml_backend_tensor_set(p->log_guard, &guard, 0, sizeof(float));
    return true;
}

pipeline_asr * pipeline_asr_load(const pipeline_asr_params & params) {
    Timer          t_load;
    pipeline_asr * p = new pipeline_asr();
    p->dump_dir      = params.dump_dir;

    try {
        GGUFModel gf = {};
        if (!gf_load(&gf, params.model_path.c_str())) {
            s2s_set_error("[ASR] Failed to open %s", params.model_path.c_str());
            pipeline_asr_free(p);
            return nullptr;
        }

        parakeet_read_hparams(gf, p->hp);
        p->mel_cfg = parakeet_mel_config(p->hp);

        p->backend = backend_init("ASR");
        if (!p->backend) {
            s2s_set_error("[ASR] No usable backend");
            gf_close(&gf);
            pipeline_asr_free(p);
            return nullptr;
        }

        wctx_init(&p->wctx, 26 * p->hp.n_layers + 32);
        parakeet_load_encoder(&p->encoder, &p->wctx, gf, p->hp);
        parakeet_load_decoder(&p->decoder, &p->wctx, gf);
        sp_detok_load(&p->sp, gf, "asr.vocab");

        const bool loaded = wctx_alloc(&p->wctx, p->backend);
        gf_close(&gf);
        if (!loaded) {
            s2s_set_error("[ASR] Failed to upload the weights");
            pipeline_asr_free(p);
            return nullptr;
        }

        if (!pipeline_asr_load_constants(p)) {
            s2s_set_error("[ASR] Failed to upload the mel constants");
            pipeline_asr_free(p);
            return nullptr;
        }

        if (!graph_arena_init(&p->dec_arena, 2 * ASR_DECODER_NODES)) {
            s2s_set_error("[ASR] Failed to allocate the decoder arena");
            pipeline_asr_free(p);
            return nullptr;
        }

        struct ggml_context * dctx = graph_arena_begin(&p->dec_arena);
        p->predict                 = parakeet_predict_build(dctx, p->decoder, ASR_DECODER_NODES);
        p->joint                   = parakeet_joint_build(dctx, p->decoder, ASR_DECODER_NODES);

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(p->backend);
        p->predict_alloc                = ggml_gallocr_new(buft);
        p->joint_alloc                  = ggml_gallocr_new(buft);
        p->run_alloc                    = ggml_gallocr_new(buft);
        if (!p->predict_alloc || !p->joint_alloc || !p->run_alloc ||
            !ggml_gallocr_alloc_graph(p->predict_alloc, p->predict.graph) ||
            !ggml_gallocr_alloc_graph(p->joint_alloc, p->joint.graph)) {
            s2s_set_error("[ASR] Failed to allocate the decoder graphs");
            pipeline_asr_free(p);
            return nullptr;
        }

        s2s_log(S2S_LOG_INFO, "[ASR] %d Hz, %d layers, d_model %d, vocab %d, durations %zu", p->hp.sample_rate,
                p->hp.n_layers, p->hp.d_model, p->decoder.vocab_size, p->decoder.durations.size());
        s2s_log(S2S_LOG_INFO, "[Perf] Load %.1f ms", t_load.ms());
        return p;
    } catch (const std::exception & e) {
        s2s_set_error("%s", e.what());
        pipeline_asr_free(p);
        return nullptr;
    }
}

void pipeline_asr_free(pipeline_asr * p) {
    if (!p) {
        return;
    }
    if (p->predict_alloc) {
        ggml_gallocr_free(p->predict_alloc);
    }
    if (p->joint_alloc) {
        ggml_gallocr_free(p->joint_alloc);
    }
    if (p->run_alloc) {
        ggml_gallocr_free(p->run_alloc);
    }
    graph_arena_free(&p->dec_arena);
    wctx_free(&p->wctx);
    wctx_free(&p->const_wctx);
    if (p->backend) {
        ggml_backend_free(p->backend);
    }
    delete p;
}

int pipeline_asr_sample_rate(const pipeline_asr * p) {
    return p ? p->hp.sample_rate : 0;
}

static void pipeline_asr_dump(const std::string & dir, const char * name, const std::vector<float> & data) {
    if (dir.empty()) {
        return;
    }
    const std::string path = dir + "/parakeet-" + name + ".f32";
    FILE *            fp   = utf8_fopen(path.c_str(), "wb");
    if (!fp) {
        s2s_log(S2S_LOG_WARN, "[ASR] Cannot write %s", path.c_str());
        return;
    }
    fwrite(data.data(), sizeof(float), data.size(), fp);
    fclose(fp);
}

// Every stage timer stops after the readback that follows it, so the GPU work
// is inside the measurement.
struct AsrPerf {
    double resample_ms = 0.0;  // host resample to the model rate, zero when it already matches
    double mel_ms      = 0.0;  // frontend graph plus the host normalization
    double encoder_ms  = 0.0;  // subsampling, FastConformer blocks and the projection
    double decode_ms   = 0.0;  // transducer loop, prediction and joint steps
    double total_ms    = 0.0;  // entry to return
    int    n_frames    = 0;    // mel frames
    int    n_tokens    = 0;    // emitted pieces
    int    n_steps     = 0;    // joint evaluations, blanks included
    double audio_sec   = 0.0;  // input audio duration
};

// Route the per stage timings through the log at info level.
static void asr_log_perf(const AsrPerf & p) {
    const double rtf     = p.audio_sec > 0.0 ? (p.total_ms / 1000.0) / p.audio_sec : 0.0;
    const double ms_step = p.n_steps > 0 ? p.decode_ms / (double) p.n_steps : 0.0;

    s2s_log(S2S_LOG_INFO, "[Perf] Resample %.1f ms", p.resample_ms);
    s2s_log(S2S_LOG_INFO, "[Perf] Mel %.1f ms (%d frames)", p.mel_ms, p.n_frames);
    s2s_log(S2S_LOG_INFO, "[Perf] Encoder %.1f ms (%d tokens)", p.encoder_ms, (p.n_frames + 7) / 8);
    s2s_log(S2S_LOG_INFO, "[Perf] Decode %.1f ms (%d steps, %d pieces, %.2f ms/step)", p.decode_ms, p.n_steps,
            p.n_tokens, ms_step);
    s2s_log(S2S_LOG_INFO, "[Perf] Total %.1f ms (audio %.2f s, RTF %.4f)", p.total_ms, p.audio_sec, rtf);
}

pk_status pipeline_asr_run(pipeline_asr *               p,
                           const float *                pcm,
                           size_t                       n_samples,
                           int                          sample_rate,
                           const pk_transcribe_params * params,
                           std::string &                text) {
    if (!p || !pcm || n_samples == 0) {
        s2s_set_error("[ASR] Empty input");
        return PK_STATUS_INVALID_PARAMS;
    }

    std::lock_guard<std::mutex> lock(p->mutex);

    AsrPerf perf;
    Timer   t_total;

    try {
        Timer              t_resample;
        std::vector<float> mono(pcm, pcm + n_samples);
        if (sample_rate != p->hp.sample_rate) {
            int     n_out = 0;
            float * r     = audio_resample(mono.data(), (int) mono.size(), sample_rate, p->hp.sample_rate, 1, &n_out);
            if (!r) {
                s2s_set_error("[ASR] Resample %d -> %d failed", sample_rate, p->hp.sample_rate);
                return PK_STATUS_DECODE_FAILED;
            }
            mono.assign(r, r + n_out);
            free(r);
        }
        perf.resample_ms = t_resample.ms();
        perf.audio_sec   = (double) mono.size() / (double) p->hp.sample_rate;
        pipeline_asr_dump(p->dump_dir, "pcm", mono);

        std::vector<float> audio;
        const size_t n_frames = parakeet_mel_prepare(mono.data(), mono.size(), p->mel_cfg, p->hp.preemphasis, audio);
        if (n_frames < 2) {
            s2s_set_error("[ASR] Audio shorter than one frame");
            return PK_STATUS_INVALID_PARAMS;
        }
        const size_t n_tokens = (n_frames + (size_t) p->hp.sub_factor - 1) / (size_t) p->hp.sub_factor;

        // Per call inputs: the audio, the mel and the sinusoid table.
        WeightCtx in_wctx = {};
        wctx_init(&in_wctx, 3);
        struct ggml_tensor * t_audio = ggml_new_tensor_1d(in_wctx.ctx, GGML_TYPE_F32, (int64_t) audio.size());
        struct ggml_tensor * t_mel   = ggml_new_tensor_2d(in_wctx.ctx, GGML_TYPE_F32, p->hp.n_mels, (int64_t) n_frames);
        struct ggml_tensor * t_pos =
            ggml_new_tensor_2d(in_wctx.ctx, GGML_TYPE_F32, p->hp.d_model, (int64_t) (2 * n_tokens - 1));
        if (!wctx_alloc(&in_wctx, p->backend)) {
            s2s_set_error("[ASR] Failed to allocate the per call inputs");
            return PK_STATUS_DECODE_FAILED;
        }
        ggml_backend_tensor_set(t_audio, audio.data(), 0, audio.size() * sizeof(float));

        Timer                   t_frontend;
        struct ggml_init_params gparams = {
            ggml_tensor_overhead() * ASR_GRAPH_NODES + ggml_graph_overhead_custom(ASR_GRAPH_NODES, false), nullptr, true
        };
        struct ggml_context * gctx = ggml_init(gparams);

        // Frontend.
        struct ggml_cgraph * mel_graph = ggml_new_graph_custom(gctx, ASR_GRAPH_NODES, false);
        struct ggml_tensor * mel_out   = parakeet_mel_build_graph(gctx, t_audio, p->hann, p->dft_real, p->dft_imag,
                                                                  p->mel_basis, p->log_guard, p->mel_cfg);
        ggml_build_forward_expand(mel_graph, mel_out);

        if (!ggml_gallocr_alloc_graph(p->run_alloc, mel_graph) ||
            ggml_backend_graph_compute(p->backend, mel_graph) != GGML_STATUS_SUCCESS) {
            ggml_free(gctx);
            wctx_free(&in_wctx);
            s2s_set_error("[ASR] Mel graph failed");
            return PK_STATUS_DECODE_FAILED;
        }

        std::vector<float> mel_full((size_t) ggml_nelements(mel_out));
        ggml_backend_tensor_get(mel_out, mel_full.data(), 0, mel_full.size() * sizeof(float));

        // The transform emits one frame past the audio, the extractor masks it
        // out, so the pipeline stops at the valid frames.
        std::vector<float> mel(mel_full.begin(), mel_full.begin() + (ptrdiff_t) (n_frames * (size_t) p->hp.n_mels));
        parakeet_mel_normalize(mel, p->hp.n_mels, n_frames);
        perf.mel_ms   = t_frontend.ms();
        perf.n_frames = (int) n_frames;
        pipeline_asr_dump(p->dump_dir, "mel", mel);

        std::vector<float> positions;
        parakeet_positions((int) n_tokens, p->hp.d_model, positions);

        ggml_backend_tensor_set(t_mel, mel.data(), 0, mel.size() * sizeof(float));
        ggml_backend_tensor_set(t_pos, positions.data(), 0, positions.size() * sizeof(float));

        // Encoder.
        Timer t_encoder;
        ggml_free(gctx);
        gctx = ggml_init(gparams);

        struct ggml_cgraph * enc_graph = ggml_new_graph_custom(gctx, ASR_GRAPH_NODES, false);
        struct ggml_tensor * stem      = nullptr;
        struct ggml_tensor * block0    = nullptr;
        struct ggml_tensor * states    = parakeet_encoder_build(gctx, p->encoder, p->hp, t_mel, t_pos, &stem, &block0);
        struct ggml_tensor * projected = parakeet_project(gctx, p->encoder, states);
        ggml_build_forward_expand(enc_graph, states);
        ggml_build_forward_expand(enc_graph, projected);
        if (!p->dump_dir.empty()) {
            // Without this the allocator reuses their memory for later nodes
            // and the dump would hold whatever ran last.
            ggml_set_output(stem);
            ggml_set_output(block0);
            ggml_build_forward_expand(enc_graph, stem);
            ggml_build_forward_expand(enc_graph, block0);
        }

        if (!ggml_gallocr_alloc_graph(p->run_alloc, enc_graph) ||
            ggml_backend_graph_compute(p->backend, enc_graph) != GGML_STATUS_SUCCESS) {
            ggml_free(gctx);
            wctx_free(&in_wctx);
            s2s_set_error("[ASR] Encoder graph failed");
            return PK_STATUS_DECODE_FAILED;
        }

        if (!p->dump_dir.empty()) {
            std::vector<float> buffer((size_t) ggml_nelements(stem));
            ggml_backend_tensor_get(stem, buffer.data(), 0, buffer.size() * sizeof(float));
            pipeline_asr_dump(p->dump_dir, "stem", buffer);

            buffer.resize((size_t) ggml_nelements(block0));
            ggml_backend_tensor_get(block0, buffer.data(), 0, buffer.size() * sizeof(float));
            pipeline_asr_dump(p->dump_dir, "block0", buffer);

            buffer.resize((size_t) ggml_nelements(states));
            ggml_backend_tensor_get(states, buffer.data(), 0, buffer.size() * sizeof(float));
            pipeline_asr_dump(p->dump_dir, "states", buffer);
        }

        std::vector<float> frames((size_t) ggml_nelements(projected));
        ggml_backend_tensor_get(projected, frames.data(), 0, frames.size() * sizeof(float));
        pipeline_asr_dump(p->dump_dir, "projected", frames);

        ggml_free(gctx);
        wctx_free(&in_wctx);
        perf.encoder_ms = t_encoder.ms();

        // Transducer.
        Timer    t_decode;
        TdtStats stats;
        text.clear();
        const pk_status status =
            parakeet_tdt_decode(p->backend, p->decoder, p->sp, p->predict, p->joint, frames.data(), (int) n_tokens,
                                params ? params->on_token : nullptr, params ? params->user : nullptr, text, stats);
        perf.decode_ms = t_decode.ms();
        pipeline_asr_dump(p->dump_dir, "prediction", stats.first_state);
        pipeline_asr_dump(p->dump_dir, "joint", stats.first_logits);
        perf.n_tokens = stats.n_tokens;
        perf.n_steps  = stats.n_steps;
        perf.total_ms = t_total.ms();
        asr_log_perf(perf);
        return status;
    } catch (const std::exception & e) {
        s2s_set_error("%s", e.what());
        return PK_STATUS_DECODE_FAILED;
    }
}
