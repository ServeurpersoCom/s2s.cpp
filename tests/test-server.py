#!/usr/bin/env python3
# test-server.py: one conversation through s2s-server, end to end.
#
# Starts the server on a test port, the client asks for loopback itself, waits for it to answer
# /health, streams a WAV through the Realtime WebSocket in real time, then
# checks that the loop did what a conversation needs: speech was detected,
# turns were committed and recognized, and audio came back.
#
# Loopback keeps the language model out of the path, so the check never
# depends on an endpoint being up.
# Run from the tests/ directory.
#
# Usage:
#     ./test-server.py
#     GGML_BACKEND=CPU ./test-server.py

import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

SERVER = "../build/s2s-server"
CLIENT = "../build/test-server"
MODELS = "../models"
WAV = "../examples/freeman.wav"
PORT = 18088
SECONDS = "7"
TMP = "tmp"

BOOT_TIMEOUT_S = 120


def check(label, ok, detail):
    print("[Check] %s: %s %s" % (label, detail, "OK" if ok else "FAIL"))
    return ok


def wait_for_health(process):
    deadline = time.time() + BOOT_TIMEOUT_S
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=1) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.5)
    return False


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(TMP, exist_ok=True)

    log = open(TMP + "/server.log", "w")
    server = subprocess.Popen(
        [SERVER, "--models", MODELS, "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=log,
        stderr=subprocess.STDOUT,
    )

    try:
        if not wait_for_health(server):
            print("[Check] boot: server never answered /health FAIL")
            return 1

        run = subprocess.run(
            [CLIENT, "ws://127.0.0.1:%d/v1/realtime" % PORT, WAV, SECONDS],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        out = run.stdout
        print(out, end="")
    finally:
        server.terminate()
        server.wait(timeout=30)
        log.close()

    events = re.findall(r"\[Event\]\s+[\d.]+s\s+(\S+)", out)
    transcripts = re.findall(r'input_audio_transcription.completed "(.*)"', out)
    spoken = re.findall(r'response.output_audio_transcript.delta\s+"(.*)"', out)
    summary = re.search(r"(\d+) events, ([\d.]+)s of audio received", out)

    ok = check("boot", bool(summary), "server answered and the client ran")
    if not ok:
        return 1

    audio_sec = float(summary.group(2))

    ok = check("session", "session.created" in events, "session.created received") and ok
    ok = check("speech", events.count("input_audio_buffer.speech_started") > 0,
               "%d speech starts" % events.count("input_audio_buffer.speech_started")) and ok
    ok = check("turns", len(transcripts) > 0, "%d turns recognized" % len(transcripts)) and ok
    ok = check("transcripts", all(t.strip() for t in transcripts), "no empty transcript") and ok
    ok = check("loopback", spoken == transcripts, "%d units spoken back verbatim" % len(spoken)) and ok
    ok = check("audio", audio_sec > 1.0, "%.2fs of synthesized audio" % audio_sec) and ok
    ok = check("responses", events.count("response.done") >= len(transcripts),
               "%d responses completed" % events.count("response.done")) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
