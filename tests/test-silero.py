#!/usr/bin/env python3
# test-silero.py: parity of the Silero VAD against the ONNX reference.
#
# Owns both sides: runs the GGML harness on the example wav, then the upstream
# ONNX graph on the very samples the harness dumped, and compares the two
# probability series plus the speech decisions at the operating threshold.
# Run from the tests/ directory.
#
# Usage:
#     ./test-silero.py

import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort

BIN = "../build/test-silero"
GGUF = "../models/silero-vad-F32.gguf"
ONNX = "../checkpoints/silero-vad/onnx/model.onnx"
WAV = "../examples/freeman.wav"
TMP = "tmp"

WINDOW = 512
CONTEXT = 64
RATE = 16000
MAX_ABS = 1e-4


def reference(pcm):
    session = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, CONTEXT), dtype=np.float32)

    probs = []
    for off in range(0, len(pcm) - WINDOW + 1, WINDOW):
        window = pcm[off : off + WINDOW].reshape(1, WINDOW)
        frame = np.concatenate([context, window], axis=1)
        out, state = session.run(
            ["output", "stateN"],
            {"input": frame, "state": state, "sr": np.array(RATE, dtype=np.int64)},
        )
        probs.append(float(out[0, 0]))
        context = window[:, -CONTEXT:]
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

    subprocess.run([BIN, GGUF, WAV, TMP + "/silero"], check=True)

    pcm = np.fromfile(TMP + "/silero-pcm.f32", dtype=np.float32)
    got = np.loadtxt(TMP + "/silero-probs.txt", dtype=np.float64, ndmin=1)

    ok = report("vad", reference(pcm), got, MAX_ABS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
