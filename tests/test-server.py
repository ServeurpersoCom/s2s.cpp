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
#
# A second run puts the client in a simulated room with the server echo
# canceller: its answers come back into the microphone as echo, and the
# speaker talks over one of them. The loop has to hear the speaker through
# the echo, stop the playback, and transcribe the interruption without a word
# of the assistant.
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
ROOM_SECONDS = "12"
TALK_OVER_WORDS = "where you go"
ECHO_WORDS = ("different", "cultures", "creation", "afterlife")
REACTION_S = 1.5
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

        url = "ws://127.0.0.1:%d/v1/realtime" % PORT
        out = subprocess.run([CLIENT, url, WAV, SECONDS], check=True, stdout=subprocess.PIPE, text=True).stdout
        print(out, end="")
        room = subprocess.run([CLIENT, url, WAV, ROOM_SECONDS, "--room"], check=True, stdout=subprocess.PIPE,
                              text=True).stdout
        print(room, end="")
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

    return 0 if check_room(room) and ok else 1


def check_room(out):
    at = lambda what: [float(t) for t in re.findall(r"\[Client\]\s+([\d.]+)s\s+%s" % what, out)]
    started = at("talking over the answer")
    ok = check("room talk over", bool(started), "the speaker talked over an answer")
    if not ok:
        return False
    t0 = started[0]

    flushed = [t for t in at("playback flushed") if t >= t0]
    ok = check("room barge-in", bool(flushed) and flushed[0] - t0 <= REACTION_S,
               "playback stopped %.2fs after the speaker started" % (flushed[0] - t0 if flushed else -1.0)) and ok

    heard = [(float(t), text) for t, text in
             re.findall(r'\[Event\]\s+([\d.]+)s\s+conversation.item.input_audio_transcription.completed.*"(.*)"', out)
             if float(t) >= t0]
    words = [text for _, text in heard if TALK_OVER_WORDS in text.lower()]
    ok = check("room transcript", bool(words), "the interruption heard: %s" % (words[0] if words else "nothing")) and ok
    echo = [text for _, text in heard if any(w in text.lower() for w in ECHO_WORDS)]
    ok = check("room echo", not echo, "no word of the assistant in what was heard") and ok
    return ok


if __name__ == "__main__":
    sys.exit(main())
