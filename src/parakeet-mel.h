#pragma once
// parakeet-mel.h: NeMo style log mel, the ParakeetFeatureExtractor equivalent
//
// Differences from the Whisper frontend in audio-mel.h:
//
//   preemphasis 0.97 on the host, y[0] = x[0], y[t] = x[t] - 0.97 x[t-1]
//   the stft is centered on zeros, not on a reflection
//   the window is a 400 point symmetric Hann centered in a 512 transform
//   log(mel + 2^-24), no clamp and no decibel scale
//   normalization is per mel bin over the utterance, with the unbiased
//   variance, instead of the Whisper global maximum
//
// The graph emits ne = [n_mels, n_frames], one frame being a contiguous run
// of mel bins, which is the image layout the conv2d stem reads.

#include "audio-mel.h"
#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <vector>

#define PARAKEET_MEL_EPS 1e-5f

// Preemphasis then zero padding by n_fft / 2 on both sides. Returns the
// frame count the encoder will see.
static size_t parakeet_mel_prepare(const float *          pcm,
                                   size_t                 n_samples,
                                   const AudioMelConfig & cfg,
                                   float                  preemphasis,
                                   std::vector<float> &   out) {
    const size_t half = (size_t) cfg.n_fft / 2;
    out.assign(n_samples + 2 * half, 0.0f);
    if (n_samples > 0) {
        out[half] = pcm[0];
    }
    for (size_t i = 1; i < n_samples; i++) {
        out[half + i] = pcm[i] - preemphasis * pcm[i - 1];
    }
    return n_samples / (size_t) cfg.hop;
}

// Power spectrogram to log mel. Returns ne = [n_mels, n_frames].
static struct ggml_tensor * parakeet_mel_build_graph(struct ggml_context *  ctx,
                                                     struct ggml_tensor *   audio_padded,
                                                     struct ggml_tensor *   hann,
                                                     struct ggml_tensor *   dft_real,
                                                     struct ggml_tensor *   dft_imag,
                                                     struct ggml_tensor *   mel_basis,
                                                     struct ggml_tensor *   log_guard,
                                                     const AudioMelConfig & cfg) {
    struct ggml_tensor * power = audio_spectrogram_build_graph(ctx, audio_padded, hann, dft_real, dft_imag, cfg);

    struct ggml_tensor * mel = ggml_mul_mat(ctx, mel_basis, power);  // [n_mels, n_frames]
    ggml_prec_set_acc(mel, GGML_PREC_F32);

    mel = ggml_log(ctx, ggml_add(ctx, mel, log_guard));
    ggml_set_name(mel, "mel.log");
    return mel;
}

// Per mel bin normalization over the utterance: subtract the mean, divide by
// the unbiased standard deviation plus epsilon. Runs on the host because both
// passes need the whole spectrogram.
static void parakeet_mel_normalize(std::vector<float> & mel, int n_mels, size_t n_frames) {
    for (int bin = 0; bin < n_mels; bin++) {
        double sum = 0.0;
        for (size_t f = 0; f < n_frames; f++) {
            sum += mel[f * (size_t) n_mels + (size_t) bin];
        }
        const double mean = sum / (double) n_frames;

        double sq = 0.0;
        for (size_t f = 0; f < n_frames; f++) {
            const double d = mel[f * (size_t) n_mels + (size_t) bin] - mean;
            sq += d * d;
        }
        const double std_dev = std::sqrt(sq / (double) (n_frames - 1));

        for (size_t f = 0; f < n_frames; f++) {
            float & value = mel[f * (size_t) n_mels + (size_t) bin];
            value         = (float) ((value - mean) / (std_dev + PARAKEET_MEL_EPS));
        }
    }
}
