#!/usr/bin/env python3
# test-parakeet.py: parity of the Parakeet frontend, encoder and transcript.
#
# Owns both sides: runs parakeet-transcribe --dump on the example wav for the
# stage tensors and the transcript, and ParakeetForTDT on the very samples the
# tool dumped for the reference, then compares the mel, the encoder states, the
# projection the joint network reads and the decoded text.
# Run from the tests/ directory.
#
# Reference runs CPU fp32 in this interpreter, needs transformers 5.6.
#
# Usage:
#     ./test-parakeet.py
#     GGML_BACKEND=CPU ./test-parakeet.py --model ../models/parakeet-tdt-0.6b-v3-Q8_0.gguf

import os
import subprocess
import sys

import difflib
import warnings

import numpy as np
import torch
from transformers import ParakeetForTDT, ParakeetProcessor
from transformers.utils import logging as hf_logging

BIN = "../build/parakeet-transcribe"


def _arg(name, default):
    if name in sys.argv and sys.argv.index(name) + 1 < len(sys.argv):
        return sys.argv[sys.argv.index(name) + 1]
    return default


GGUF = _arg("--model", "../models/parakeet-tdt-0.6b-v3-F32.gguf")
CKPT = "../checkpoints/parakeet"
WAV = "../qwentts.cpp/examples/freeman.wav"
TMP = "tmp"

RATE = 16000

# Cosine floor per weight format. The f32 file is compared against the same
# arithmetic the reference runs, so it sits at the numerical noise; a quantized
# file genuinely moves the activations, and the transcript is what has to stay
# identical.
MIN_COS = {"F32": 0.9999, "Q8_0": 0.999, "Q6_K": 0.999, "Q5_K_M": 0.999, "Q4_K_M": 0.98}

# Transcript floor per weight format. The f32 file has to match the reference
# character for character; a quantized file is allowed to differ on casing or
# a hesitation, and the ratio is what catches a real regression.
MIN_TEXT = {"F32": 1.0, "Q8_0": 1.0, "Q6_K": 0.999, "Q5_K_M": 0.995, "Q4_K_M": 0.98}

# Stages the tool dumps, in forward order: the normalized mel, the subsampling
# output, the first conformer block, the encoder states, the 640 wide
# projection, the prediction the start of sequence gives, and the joint logits
# on the first encoder frame.
STAGES = ["mel", "stem", "block0", "states", "projected", "prediction", "joint"]


def report(label, ref, got, min_cos):
    ref = ref.reshape(-1).astype(np.float64)
    got = got.reshape(-1).astype(np.float64)
    if ref.size != got.size:
        print("[Parity] %s FAIL size %d vs %d" % (label, ref.size, got.size))
        return False
    cos = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got)))
    mx = float(np.abs(ref - got).max())
    ok = cos >= min_cos
    print("[Parity] %s: cossim %.9f max abs %.3e peak %.3f %s" % (label, cos, mx, np.abs(ref).max(), "OK" if ok else "FAIL"))
    return ok


def report_text(ref, got, min_ratio):
    ratio = difflib.SequenceMatcher(None, ref, got).ratio()
    ok = ratio >= min_ratio
    print("[Parity] text: ratio %.6f %s" % (ratio, "OK" if ok else "FAIL"))
    if ratio < 1.0:
        for line in difflib.unified_diff([ref], [got], "reference", "ggml", lineterm=""):
            print(line)
    return ok


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(TMP, exist_ok=True)
    hf_logging.disable_progress_bar()
    warnings.filterwarnings("ignore")

    subprocess.run(
        [BIN, "--model", GGUF, "--file", WAV, "--dump", TMP, "--out", TMP + "/parakeet-text.txt"],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    pcm = np.fromfile(TMP + "/parakeet-pcm.f32", dtype=np.float32)

    processor = ParakeetProcessor.from_pretrained(CKPT)
    model = ParakeetForTDT.from_pretrained(CKPT, dtype=torch.float32).eval()

    inputs = processor(pcm, sampling_rate=RATE, return_tensors="pt")

    # Hooks on the two encoder stages that have no entry point of their own.
    taps = {}
    handles = [
        model.encoder.subsampling.register_forward_hook(lambda m, i, o: taps.__setitem__("stem", o)),
        model.encoder.layers[0].register_forward_hook(lambda m, i, o: taps.__setitem__("block0", o)),
    ]

    with torch.no_grad():
        encoder_out = model.encoder(**inputs)
        projected = model.encoder_projector(encoder_out.last_hidden_state)
        prediction = model.decoder(torch.tensor([[model.config.blank_token_id]]))
        joint = model.joint(decoder_hidden_states=prediction[:, 0], encoder_hidden_states=projected[:, 0])
        generated = model.generate(**inputs)

    for handle in handles:
        handle.remove()

    # The extractor emits one transform frame past the audio and masks it out,
    # the GGML frontend stops at the valid frames.
    n_valid = int(inputs["attention_mask"][0].sum())
    ref = {
        "mel": inputs["input_features"][0].numpy()[:n_valid],
        "stem": taps["stem"][0].numpy(),
        "block0": taps["block0"][0].numpy(),
        "states": encoder_out.last_hidden_state[0].numpy(),
        "projected": projected[0].numpy(),
        "prediction": prediction[0, 0].numpy(),
        "joint": joint[0].numpy(),
    }

    quant = os.path.basename(GGUF).rsplit("-", 1)[-1].replace(".gguf", "")
    min_cos = MIN_COS.get(quant, 0.99)

    ok = True
    for stage in STAGES:
        got = np.fromfile("%s/parakeet-%s.f32" % (TMP, stage), dtype=np.float32)
        ok = report(stage, ref[stage], got, min_cos) and ok

    text_ref = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
    text_got = open(TMP + "/parakeet-text.txt").read().strip()
    ok = report_text(text_ref, text_got, MIN_TEXT.get(quant, 0.98)) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
