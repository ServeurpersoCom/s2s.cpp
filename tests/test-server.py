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
#
# A server that owns its endpoint runs last: /props says nothing of it, the
# model list is closed, and a client naming another endpoint is refused.
#
# A third run asks for a conversation on a server without an endpoint,
# naming none either. Across all three, every response.created is closed by exactly one
# response.done or response.cancelled.
# Run from the tests/ directory.
#
# Usage:
#     ./test-server.py
#     GGML_BACKEND=CPU ./test-server.py

import json
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
OWNED_PORT = 18087
OWNED_URL = "http://127.0.0.1:18086/v1"
OWNED_MODEL = "owned-model"
OWNED_KEY = "s2s-test-key"
GRACE_S = 0.8  # reopen_grace_ms at its default
TMP = "tmp"

BOOT_TIMEOUT_S = 120


def check(label, ok, detail):
    print("[Check] %s: %s %s" % (label, detail, "OK" if ok else "FAIL"))
    return ok


def wait_for_health(process, port=PORT):
    deadline = time.time() + BOOT_TIMEOUT_S
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=1) as response:
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
        broken = subprocess.run([CLIENT, url, WAV, SECONDS, "--no-endpoint"], check=True, stdout=subprocess.PIPE,
                                text=True).stdout
        print(broken, end="")
    finally:
        server.terminate()
        server.wait(timeout=30)
        log.close()

    events = re.findall(r"\[Event\]\s+[\d.]+s\s+(\S+)", out)
    items = re.findall(r'input_audio_transcription.completed\s+(\S*) "(.*)"', out)
    transcripts = [text for _, text in items]
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
    # Each transcript names its turn, and a turn is transcribed once.
    named = [item for item, _ in items if item.startswith("turn_")]
    ok = check("items", len(named) == len(items) and len(set(named)) == len(named),
               "%d transcripts, each with its own turn item" % len(items)) and ok
    # The example is a monologue: the next words often arrive during the
    # grace, and the answer to a fragment is dropped before anyone hears it.
    # Every answer that is heard speaks back the turn that opened it.
    heard, verbatim = spoken_back(out)
    ok = check("loopback", heard > 0 and verbatim, "%d answers heard, each its turn verbatim" % heard) and ok
    ok = check("audio", audio_sec > 1.0, "%.2fs of synthesized audio" % audio_sec) and ok
    ok = check_grace(TMP + "/server.log") and ok

    ok = check_room(room) and ok
    for label, run in (("plain", out), ("room", room), ("no endpoint", broken)):
        ok = check_terminals(label, run) and ok
    errors = re.findall(r"\[Event\]\s+[\d.]+s\s+error\s+(.*)", broken)
    ok = check("no endpoint", bool(errors) and all("No endpoint" in e for e in errors),
               "%d answers told there is no endpoint" % len(errors)) and ok
    return 0 if check_owned_endpoint() and ok else 1


def check_owned_endpoint():
    key_file = TMP + "/llm.key"
    with open(key_file, "w") as f:
        f.write(OWNED_KEY + "\n")

    log = open(TMP + "/server-owned.log", "w")
    server = subprocess.Popen(
        [SERVER, "--models", MODELS, "--host", "127.0.0.1", "--port", str(OWNED_PORT),
         "--llm-url", OWNED_URL, "--llm-model", OWNED_MODEL, "--llm-key-file", key_file],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        if not wait_for_health(server, OWNED_PORT):
            return check("owned endpoint", False, "server never answered /health")
        base = "http://127.0.0.1:%d" % OWNED_PORT
        props = urllib.request.urlopen(base + "/props", timeout=5).read().decode()
        try:
            urllib.request.urlopen(urllib.request.Request(base + "/v1/models", data=b"{}", method="POST"), timeout=5)
            models = 200
        except urllib.error.HTTPError as e:
            models = e.code
        out = subprocess.run([CLIENT, "ws://127.0.0.1:%d/v1/realtime" % OWNED_PORT, WAV, SECONDS, "--other-endpoint"],
                             check=True, stdout=subprocess.PIPE, text=True).stdout
        print(out, end="")
    finally:
        server.terminate()
        server.wait(timeout=30)
        log.close()
        os.remove(key_file)

    fixed = json.loads(props)["defaults"].get("llm_fixed") is True
    hidden = not any(secret in props for secret in (OWNED_URL, OWNED_MODEL, OWNED_KEY))
    ok = check("owned props", fixed and hidden, "the endpoint is the server's, its URL, model and key unpublished")
    ok = check("owned models", models == 403, "model list closed with %d" % models) and ok
    refused = re.findall(r"\[Event\]\s+[\d.]+s\s+error\s+(.*)", out)
    ok = check("owned endpoint", any("set by the server" in e for e in refused),
               "another endpoint refused: %s" % (refused[0] if refused else "no error")) and ok
    return check_terminals("owned", out) and ok

# The completed responses that spoke, and whether each one spoke back the
# transcript that opened it, word for word.
def spoken_back(out):
    heard, verbatim, question, answer = 0, True, None, []
    for name, rest in re.findall(r"\[Event\]\s+[\d.]+s\s+(\S+)(.*)", out):
        if name == "conversation.item.input_audio_transcription.completed":
            question = re.search(r'"(.*)"', rest).group(1)
        elif name == "response.created":
            answer = []
        elif name == "response.output_audio_transcript.delta":
            answer.append(re.search(r'"(.*)"', rest).group(1))
        elif name == "response.done" and answer:
            heard += 1
            verbatim = verbatim and " ".join(answer) == question
    return heard, verbatim


# From the server log: a turn committed as complete turns final no sooner
# than the grace, a forced commit turns final at once.
def check_grace(path):
    commits, graced, forced, ok = {}, 0, 0, True
    for line in open(path, errors="replace"):
        if "[Server] Connection from" in line:
            commits = {}
        m = re.search(r"Turn committed at ([\d.]+)s \(turn (\d+) .*completion ([\d.]+)", line)
        if m:
            commits[m.group(2)] = (float(m.group(1)), float(m.group(3)))
        m = re.search(r"Turn final at ([\d.]+)s \(turn (\d+) ", line)
        if m and m.group(2) in commits:
            at, score = commits.pop(m.group(2))
            gap = float(m.group(1)) - at
            if score == 0.0:
                forced += 1
                ok = ok and gap == 0.0
            else:
                graced += 1
                ok = ok and gap >= GRACE_S - 0.001
    return check("grace", ok and graced > 0, "%d answers held for the grace, %d forced commits final at once" % (graced, forced))


# Each response.created is closed by exactly one response.done or
# response.cancelled before the next one opens.
def check_terminals(label, out):
    events = re.findall(r"\[Event\]\s+[\d.]+s\s+(\S+)", out)
    open_response, closed = False, True
    for name in events:
        if name == "response.created":
            closed = closed and not open_response
            open_response = True
        elif name in ("response.done", "response.cancelled"):
            closed = closed and open_response
            open_response = False
    closed = closed and not open_response
    return check("%s terminals" % label, closed,
                 "%d responses, each closed exactly once" % events.count("response.created"))


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
