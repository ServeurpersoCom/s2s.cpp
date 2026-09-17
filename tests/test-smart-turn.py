#!/usr/bin/env python3
# test-smart-turn.py: parity of the Smart Turn classifier against the ONNX reference.
#
# Owns both sides: runs the GGML harness on the example wav, rebuilds the
# Whisper features for the same growing prefixes, feeds them to the upstream
# ONNX graph, and compares the completion probabilities plus the decisions at
# the operating threshold. Run from the tests/ directory.
#
# Usage:
#     ./test-smart-turn.py

import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort

BIN = "../build/test-smart-turn"
GGUF = "../models/smart-turn-v3.2-F32.gguf"
ONNX = "../checkpoints/smart-turn/smart-turn-v3.2-gpu.onnx"
WAV = "../examples/freeman.wav"
TMP = "tmp"

RATE = 16000
WINDOW = 8 * RATE
N_FFT = 400
HOP = 160
N_MELS = 80
N_FRAMES = 800
MAX_ABS = 1e-4


def mel_filters():
    # Slaney filterbank, librosa.filters.mel(htk=False) with fmin 0, fmax 8000.
    def hz_to_mel(hz):
        f_sp, min_log_hz = 200.0 / 3.0, 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        return np.where(hz < min_log_hz, hz / f_sp, min_log_mel + np.log(np.maximum(hz, 1e-9) / min_log_hz) / logstep)

    def mel_to_hz(mel):
        f_sp, min_log_hz = 200.0 / 3.0, 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        return np.where(mel < min_log_mel, f_sp * mel, min_log_hz * np.exp(logstep * (mel - min_log_mel)))

    n_freq = N_FFT // 2 + 1
    hz = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(RATE / 2), N_MELS + 2))
    freqs = np.arange(n_freq) * RATE / N_FFT
    filters = np.zeros((N_MELS, n_freq), dtype=np.float64)
    for m in range(N_MELS):
        lo, mid, hi = hz[m], hz[m + 1], hz[m + 2]
        up, down = (freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)
        filters[m] = np.maximum(0.0, np.minimum(up, down)) * (2.0 / (hi - lo))
    return filters


def features(pcm, filters):
    audio = np.zeros(WINDOW, dtype=np.float64)
    kept = min(len(pcm), WINDOW)
    audio[:kept] = pcm[len(pcm) - kept :]

    padded = np.pad(audio, N_FFT // 2, mode="reflect")
    window = np.hanning(N_FFT + 1)[:N_FFT]
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP] * window
    spec = np.fft.rfft(frames, axis=-1)
    power = (spec.real**2 + spec.imag**2).T[:, :-1]

    log_spec = np.log10(np.maximum(filters @ power, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def reference(pcm):
    session = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    filters = mel_filters()

    probs = []
    for end in range(RATE, len(pcm) + 1, RATE):
        mel = features(pcm[:end].astype(np.float64), filters)
        out = session.run(["logits"], {"input_features": mel[None, :, :]})[0]
        probs.append(float(out[0, 0]))
    return np.array(probs, dtype=np.float64)


def report(label, ref, got, max_abs):
    if ref.size != got.size:
        print("[Parity] %s FAIL size %d vs %d" % (label, ref.size, got.size))
        return False
    cos = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got)))
    mx = float(np.abs(ref - got).max())
    flips = int(np.sum((ref > 0.5) != (got > 0.5)))
    ok = mx <= max_abs and flips == 0
    print(
        "[Parity] %s: cossim %.9f max abs %.3e decisions %d/%d %s"
        % (label, cos, mx, ref.size - flips, ref.size, "OK" if ok else "FAIL")
    )
    return ok


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(TMP, exist_ok=True)

    subprocess.run([BIN, GGUF, WAV, TMP + "/turn"], check=True)

    pcm = np.fromfile(TMP + "/turn-pcm.f32", dtype=np.float32)
    got = np.loadtxt(TMP + "/turn-probs.txt", dtype=np.float64, ndmin=1)

    ok = report("turn", reference(pcm), got, MAX_ABS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
