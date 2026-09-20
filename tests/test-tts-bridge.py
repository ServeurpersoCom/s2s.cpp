#!/usr/bin/env python3
# test-tts-bridge.py: streaming and cancellation properties of the TTS bridge.
#
# The synthesis itself is qwentts.cpp territory and has its own parity
# harnesses over there. What this checks is the contract the session loop
# depends on: audio arrives in chunks rather than in one block, the first
# chunk arrives early enough to hide the latency, and a raised cancel flag
# stops the synthesis instead of running it to the end.
# Run from the tests/ directory.
#
# Usage:
#     ./test-tts-bridge.py
#     GGML_BACKEND=CPU ./test-tts-bridge.py

import os
import re
import subprocess
import sys
import wave

BIN = "../build/test-tts-bridge"
TALKER = "../models/qwen-talker-1.7b-base-Q8_0.gguf"
CODEC = "../models/qwen-tokenizer-12hz-Q8_0.gguf"
VOICES = "../voices"
TMP = "tmp"

MAX_TTFA_MS = 1500.0
MAX_CANCEL_MS = 1000.0

# The harness speaks one short sentence. Anything past this means the talker
# never emitted its end of speech and ran to the frame cap.
MAX_AUDIO_SEC = 20.0


def check(label, ok, detail):
    print("[Check] %s: %s %s" % (label, detail, "OK" if ok else "FAIL"))
    return ok


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(TMP, exist_ok=True)

    run = subprocess.run(
        [BIN, TALKER, CODEC, VOICES, TMP + "/tts"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    log = run.stderr

    spoke = re.search(r"\[TTS\] Spoke ([\d.]+)s in (\d+) chunks", log)
    perf = re.search(r"\[Perf\] TTFA ([\d.]+) ms, total ([\d.]+) ms, RTF ([\d.]+)", log)
    barge = re.search(r"\[TTS\] Barge-in after (\d+) chunks, ([\d.]+)s of audio, returned (\w+)", log)
    cancel = re.search(r"\[Perf\] Cancel ([\d.]+) ms", log)

    ok = check("parsed", bool(spoke and perf and barge and cancel), "harness reported every stage")
    if not ok:
        return 1

    audio_sec, n_chunks = float(spoke.group(1)), int(spoke.group(2))
    ttfa_ms, total_ms, rtf = (float(perf.group(i)) for i in (1, 2, 3))
    cut_chunks, cut_sec, returned = int(barge.group(1)), float(barge.group(2)), barge.group(3)
    cancel_ms = float(cancel.group(1))

    ok = check("streaming", n_chunks > 1, "%d chunks over %.2fs of audio" % (n_chunks, audio_sec)) and ok
    ok = check("end of speech", audio_sec <= MAX_AUDIO_SEC, "%.2fs for one sentence" % audio_sec) and ok
    ok = check("ttfa", ttfa_ms <= MAX_TTFA_MS, "%.1f ms to first audio" % ttfa_ms) and ok
    ok = check("faster than real time", rtf < 1.0, "RTF %.4f over %.1f ms" % (rtf, total_ms)) and ok
    ok = check("cancel honored", returned == "false", "synthesis reported cancelled") and ok
    ok = check("cancel latency", cancel_ms <= MAX_CANCEL_MS, "%.1f ms to release the floor" % cancel_ms) and ok
    ok = check("cancel truncates", cut_sec < audio_sec, "%.2fs after %d chunks" % (cut_sec, cut_chunks)) and ok

    with wave.open(TMP + "/tts-speech.wav", "rb") as wav:
        written = wav.getnframes() / wav.getframerate()
    ok = check("wav", abs(written - audio_sec) < 0.05, "%.2fs written at %d Hz" % (written, 24000)) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
