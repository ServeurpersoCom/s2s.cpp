#!/usr/bin/env python3
# test-localvqe.py: parity of the echo canceller against the PyTorch reference.
#
# Owns both sides. Builds a synthetic call from the example wav: the first
# half plays through a loudspeaker, delayed and smeared by a decaying room
# response, the second half is the local talker, who also speaks over the
# echo for a while. Runs the GGML harness hop by hop on it, runs the upstream
# LocalVQE model on the same signals in one pass, and compares the outputs,
# the harness being one hop late. Then checks that the echo is really gone
# where only the far end plays. Last, runs the call again among other streams
# sharing the batch, which come and go and grow it halfway, and checks that
# the output of the call does not move. Run from the tests/ directory.
#
# Usage:
#     ./test-localvqe.py
#     GGML_BACKEND=CPU ./test-localvqe.py

import os
import subprocess
import sys
import wave

sys.dont_write_bytecode = True

import numpy as np
import torch

sys.path.insert(0, "../../LocalVQE/pytorch")
from localvqe.model import LocalVQE  # noqa: E402

BIN = "../build/test-localvqe"
GGUF = "../models/localvqe-v1.3-F32.gguf"
CKPT = "../checkpoints/localvqe/localvqe-v1.3-4.8M.pt"
WAV = "../qwentts.cpp/examples/freeman.wav"
TMP = "tmp"

RATE = 16000
HOP = 256
SECONDS = 8
DELAY_S = 0.12
MIN_COS = 0.9999
MIN_ERLE_DB = 20.0
STREAMS = 4
MIN_BATCH_COS = 0.99999  # the same stream, alone or in a batch: kernels may differ, the result may not


def load_wav(path):
    with wave.open(path) as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
        pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
    t = np.arange(int(len(pcm) * RATE / rate)) * rate / RATE
    return np.interp(t, np.arange(len(pcm)), pcm)


def scenario():
    rng = np.random.default_rng(0)
    speech = load_wav(WAV)
    n = SECONDS * RATE
    far = speech[:n]
    near = np.zeros(n)
    near[n // 2 :] = speech[n : n + n - n // 2] if len(speech) >= 2 * n else speech[: n - n // 2]

    # the far end plays alone for the first 3/8, the talker joins at 1/2
    far[3 * n // 4 :] = 0.0
    taps = int(0.05 * RATE)
    room = rng.standard_normal(taps) * np.exp(-np.arange(taps) / (0.01 * RATE))
    room /= np.abs(room).sum()
    echo = np.convolve(far, room)[:n]
    echo = np.concatenate([np.zeros(int(DELAY_S * RATE)), echo])[:n] * 4.0

    mic = echo + near + 1e-3 * rng.standard_normal(n)
    return mic.astype(np.float32), far.astype(np.float32), n


def reference(mic, ref):
    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = LocalVQE(
        mic_channels=config["mic_channels"],
        far_channels=config["far_channels"],
        align_hidden=32,
        dmax=64,
        power_law_c=0.3,
        n_freqs=config["n_freqs"],
        kernel_size=config["kernel_size"],
        bottleneck_hidden=config["bottleneck_hidden"],
        arch_version=3,
    ).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.align.fold_temperature()

    with torch.no_grad():
        enhanced = model(torch.from_numpy(mic)[None], torch.from_numpy(ref)[None])  # (1, F, T, 2)
        frames = model.decoder.linear(enhanced.permute(0, 1, 3, 2).reshape(1, -1, enhanced.shape[2]).transpose(1, 2))
    frames = frames[0].numpy().astype(np.float64)  # (T, 512)

    # sqrt-Hann analysis and synthesis sum to one at half overlap: plain
    # overlap-add, then the first half frame of padding goes
    out = np.zeros((frames.shape[0] + 1) * HOP)
    for t, frame in enumerate(frames):
        out[t * HOP : t * HOP + 2 * HOP] += frame
    return out[HOP:]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(TMP, exist_ok=True)

    mic, far, n = scenario()
    mic.tofile(TMP + "/localvqe-mic.f32")
    far.tofile(TMP + "/localvqe-ref.f32")
    subprocess.run([BIN, GGUF, TMP + "/localvqe-mic.f32", TMP + "/localvqe-ref.f32", TMP + "/localvqe-out.f32"],
                   check=True)

    got = np.fromfile(TMP + "/localvqe-out.f32", dtype=np.float32).astype(np.float64)[HOP:]
    ref = reference(mic, far)[: got.size]

    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref)))
    mx = float(np.abs(got - ref).max())
    ok = cos >= MIN_COS
    print("[Parity] output: cossim %.9f max abs %.3e %s" % (cos, mx, "OK" if ok else "FAIL"))

    # far end alone, past the first second the delay search needs
    lo, hi = RATE, 3 * n // 8
    erle = 10 * np.log10(np.mean(mic[lo:hi].astype(np.float64) ** 2) / np.mean(got[lo:hi] ** 2))
    erle_ok = erle >= MIN_ERLE_DB
    print("[Check] echo removed where the far end plays alone: %.1f dB %s" % (erle, "OK" if erle_ok else "FAIL"))

    subprocess.run([BIN, GGUF, TMP + "/localvqe-mic.f32", TMP + "/localvqe-ref.f32", TMP + "/localvqe-batch.f32",
                    str(STREAMS)], check=True)
    batch = np.fromfile(TMP + "/localvqe-batch.f32", dtype=np.float32).astype(np.float64)[HOP:]
    bcos = float(batch @ got / (np.linalg.norm(batch) * np.linalg.norm(got)))
    batch_ok = bcos >= MIN_BATCH_COS and np.isfinite(batch).all()
    print("[Check] same output among %d streams sharing the batch: cossim %.9f max abs %.3e %s" %
          (STREAMS, bcos, float(np.abs(batch - got).max()), "OK" if batch_ok else "FAIL"))

    return 0 if ok and erle_ok and batch_ok else 1


if __name__ == "__main__":
    sys.exit(main())
