#!/usr/bin/env python3
# test-server.py: the behavior of s2s-server, one scenario at a time.
#
# Each scenario runs the scripted client, test-server, against one server: a
# script of passages of the example file, silences, commits and a microphone
# cut, and for a conversation a mock endpoint whose latency the scenario sets.
# The client prints a timeline, and the checks judge it in time windows
# relative to its own events, so they hold whatever the machine.
#
# The passages of examples/freeman.wav are three sentences:
#     A  0.0 to 6.0 s   "If you go into different cultures, ... concepts of creation."
#     B  6.0 to 11.6 s  "They have their own creation story ... afterlife is."
#     C  11.6 to 16.0 s "Um where you go, what you do, who you're gonna be with, you know."
#
# Every script ends on 3 s of silence, enough for an unfinished sentence to be
# committed anyway, then the client settles until its answer is over.
#
# Every scenario also checks two invariants: each response.created is closed
# by exactly one response.done or response.cancelled, and the transcripts of
# a turn come in a row, never after another turn.
#
# A server that owns its endpoint runs last: /props and the log say nothing
# of it, the model list is closed, and a client naming another endpoint is
# refused.
# Run from the tests/ directory; the timelines land in tmp/.
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
VOICES = "../voices"
WAV = "../examples/freeman.wav"
PORT = 18088
OWNED_PORT = 18087
OWNED_URL = "http://127.0.0.1:18086/v1"
OWNED_MODEL = "owned-model"
OWNED_KEY = "s2s-test-key"
TMP = "tmp"

A, B, C, AB = "0-6", "6-11.6", "11.6-16", "0-11.6"
GRACE_S = 0.8     # reopen_grace_ms at its default
REACTION_S = 1.5  # longest a talk over may take to stop the playback
ECHO_WORDS = ("different", "cultures", "creation", "afterlife")

BOOT_TIMEOUT_S = 120

LINE = re.compile(r"\[(Event|Client|Mock)\]\s+([\d.]+)s\s+(\S+)(.*)")


def check(label, ok, detail):
    print("[Check] %s: %s %s" % (label, detail, "OK" if ok else "FAIL"))
    return ok


class Timeline:
    def __init__(self, out):
        self.lines = [(m.group(1), float(m.group(2)), m.group(3), m.group(4).strip())
                      for m in map(LINE.match, out.splitlines()) if m]
        summary = re.search(r"([\d.]+)s of audio received", out)
        self.audio_sec = float(summary.group(1)) if summary else 0.0

    def at(self, source, name, after=0.0):
        return [(t, rest) for s, t, n, rest in self.lines if s == source and n == name and t >= after]

    def first(self, source, name, after=0.0):
        found = self.at(source, name, after)
        return found[0][0] if found else None

    # When the client started saying a passage, and when it was done with it.
    def said(self, passage):
        start, end = map(float, passage.split("-"))
        return next((t for t, rest in self.at("Client", "say") if rest == "%.2f-%.2f" % (start, end)), None)

    def said_end(self, passage):
        start, end = map(float, passage.split("-"))
        t = self.said(passage)
        return None if t is None else t + end - start

    def items(self):
        found = []
        for t, rest in self.at("Event", "conversation.item.input_audio_transcription.completed"):
            m = re.match(r'(\S*) "(.*)"', rest)
            found.append((t, m.group(1), m.group(2)))
        return found

    def spoken(self):
        return [re.match(r'"(.*)"', rest).group(1) for _, rest in self.at("Event", "response.output_audio_transcript.delta")]

    def errors(self):
        return [rest for _, rest in self.at("Event", "error")]


# Each response.created is closed by exactly one response.done or
# response.cancelled before the next one opens, and the transcripts of a
# turn come in a row.
def invariants(label, run):
    open_response, closed = False, True
    for source, _, name, _ in run.lines:
        if source != "Event":
            continue
        if name == "response.created":
            closed = closed and not open_response
            open_response = True
        elif name in ("response.done", "response.cancelled"):
            closed = closed and open_response
            open_response = False
    closed = closed and not open_response
    ok = check("%s terminals" % label, closed,
               "%d responses, each closed exactly once" % len(run.at("Event", "response.created")))
    named = [item for _, item, _ in run.items()]
    grouped = all(named[i] == named[i - 1] or named[i] not in named[:i] for i in range(1, len(named)))
    return check("%s items" % label, grouped, "%d transcripts of %d turns, revisions in a row" %
                 (len(named), len(set(named)))) and ok


# One sentence in loopback: the answer is its transcript, word for word, and
# it starts once the grace has run out.
def check_loopback(run):
    items = run.items()
    stopped = run.at("Event", "input_audio_buffer.speech_stopped")
    audio = run.first("Event", "response.output_audio.delta")
    ok = check("loopback turn", len(set(i for _, i, _ in items)) == 1 and bool(items), "%d transcripts" % len(items))
    if not ok or not stopped or audio is None:
        return check("loopback audio", False, "no answer")
    ok = check("loopback verbatim", " ".join(run.spoken()) == items[-1][2], "the answer is the transcript") and ok
    gap = audio - stopped[-1][0]
    ok = check("loopback latency", GRACE_S - 0.1 <= gap <= GRACE_S + 1.5,
               "first audio %.2fs after the speech stopped" % gap) and ok
    return check("loopback length", run.audio_sec > 1.0, "%.2fs of audio" % run.audio_sec) and ok


# A monologue: the speaker goes on during the grace, the turn is resumed and
# recognized again whole, and nothing is spoken before the speaker is done.
def check_grace(run):
    items = run.items()
    ended = run.said_end(AB)
    audio = run.first("Event", "response.output_audio.delta")
    ok = check("grace one turn", bool(items) and len(set(i for _, i, _ in items)) == 1 and len(items) >= 2,
               "%d transcripts of one turn" % len(items))
    whole = items[-1][2].lower() if items else ""
    ok = check("grace whole", "creation" in whole and "afterlife" in whole, "last transcript: %s" % whole) and ok
    return check("grace silent", audio is not None and ended is not None and audio >= ended,
                 "first audio %.2fs after the monologue ended" % ((audio or 0.0) - (ended or 0.0))) and ok


# The speaker goes on after the grace but before the slow endpoint answered:
# nothing was heard, so the turn is the same one, the pending request is
# aborted, and the model reads both sentences as one message.
def check_before_answer(run):
    items = run.items()
    ended = run.said_end(B)
    audio = run.first("Event", "response.output_audio.delta")
    requests = run.at("Mock", "request")
    ok = check("before answer one turn", bool(items) and len(set(i for _, i, _ in items)) == 1,
               "%d transcripts, turns %s" % (len(items), sorted(set(i for _, i, _ in items))))
    ok = check("before answer nothing heard", audio is not None and ended is not None and audio >= ended,
               "first audio %.2fs after the second sentence ended" % ((audio or 0.0) - (ended or 0.0))) and ok
    last = requests[-1][1] if requests else ""
    ok = check("before answer one message", last.startswith("1 user messages") and "creation" in last and
               "afterlife" in last, "last request: %s" % last) and ok
    return check("before answer aborted", bool(run.at("Mock", "aborted")), "the request of the first sentence aborted") and ok


# The speaker talks over an answer already playing while the endpoint still
# writes it: a new turn, the playback, the response and the request stopped
# at once.
def check_barge_in(run):
    t0 = run.said(C)
    if t0 is None:
        return check("barge-in", False, "no talk over, no answer was heard")
    items = run.items()
    flushed = run.first("Client", "playback", t0)
    cancelled = run.first("Event", "response.cancelled", t0)
    ok = check("barge-in new turn", len(set(i for _, i, _ in items)) == 2, "turns %s" % sorted(set(i for _, i, _ in items)))
    ok = check("barge-in playback", flushed is not None and flushed - t0 <= REACTION_S,
               "playback flushed %.2fs after the talk over" % ((flushed or t0) - t0)) and ok
    ok = check("barge-in response", cancelled is not None and cancelled - t0 <= REACTION_S,
               "response cancelled %.2fs after the talk over" % ((cancelled or t0) - t0)) and ok
    ok = check("barge-in request", bool(run.at("Mock", "aborted", t0)), "the endpoint request aborted") and ok
    heard = [text for t, _, text in items if t >= t0]
    return check("barge-in transcript", any("where you go" in text.lower() for text in heard),
                 "the interruption heard: %s" % (heard[0] if heard else "nothing")) and ok


# A push to talk: the client commits mid sentence and cuts its microphone,
# and the answer still comes.
def check_push_to_talk(run):
    committed = run.first("Client", "commit")
    audio = run.first("Event", "response.output_audio.delta", committed or 0.0)
    ok = check("push to talk turn", bool(run.items()), "%d transcripts" % len(run.items()))
    return check("push to talk answer", committed is not None and audio is not None and audio - committed < GRACE_S,
                 "first audio %.2fs after the commit, no grace" % ((audio or 0.0) - (committed or 0.0))) and ok


# A laptop on speakers: the answer comes back into the microphone, the speaker
# talks over it, and the server echo canceller keeps the assistant out of
# what is heard.
def check_room(run):
    t0 = run.said(C)
    if t0 is None:
        return check("room", False, "no talk over, no answer was heard")
    flushed = run.first("Client", "playback", t0)
    ok = check("room barge-in", flushed is not None and flushed - t0 <= REACTION_S,
               "playback flushed %.2fs after the talk over" % ((flushed or t0) - t0))
    heard = [text for t, _, text in run.items() if t >= t0]
    ok = check("room transcript", any("where you go" in text.lower() for text in heard),
               "the interruption heard: %s" % (heard[0] if heard else "nothing")) and ok
    echo = [text for text in heard if any(w in text.lower() for w in ECHO_WORDS)]
    return check("room echo", not echo, "no word of the assistant in what was heard") and ok


def check_error(expected):
    def judge(run):
        errors = run.errors()
        return check("error", bool(errors) and all(expected in e for e in errors),
                     "%d answers told: %s" % (len(errors), errors[0] if errors else "nothing"))
    return judge


CONVERSATION = ["--mode", "conversation", "--llm", "v1"]

SCENARIOS = [
    ("loopback", ["--mode", "loopback", "say:" + A, "pause:3"], check_loopback),
    ("grace", ["--mode", "loopback", "say:" + AB, "pause:3"], check_grace),
    ("before answer", CONVERSATION + ["--llm-first-ms", "3000", "say:" + A, "pause:1.5", "say:" + B, "pause:3"],
     check_before_answer),
    ("barge-in", CONVERSATION + ["--llm-token-ms", "150", "say:" + A, "heard:1000", "say:" + C, "pause:3"],
     check_barge_in),
    ("push to talk", CONVERSATION + ["say:0-3", "commit", "mute:3"], check_push_to_talk),
    ("room", ["--mode", "loopback", "--echo", "server", "--room", "say:" + A, "heard:1000", "say:" + C, "pause:3"],
     check_room),
    ("no endpoint", ["--mode", "conversation", "say:" + A, "pause:3"], check_error("No endpoint")),
    ("broken endpoint", ["--mode", "conversation", "--llm", "broken", "say:" + A, "pause:3"],
     check_error("model not loaded")),
    ("midstream", ["--mode", "conversation", "--llm", "midstream", "say:" + A, "pause:3"],
     check_error("context size exceeded")),
]


def wait_for_health(process, port):
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


def run_client(port, name, args):
    out = subprocess.run([CLIENT, "ws://127.0.0.1:%d/v1/realtime" % port, WAV] + args, check=True,
                         stdout=subprocess.PIPE, text=True).stdout
    with open("%s/%s.txt" % (TMP, name.replace(" ", "-")), "w") as f:
        f.write(out)
    return Timeline(out)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(TMP, exist_ok=True)

    log = open(TMP + "/server.log", "w")
    server = subprocess.Popen(
        [SERVER, "--models", MODELS, "--voices", VOICES, "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    ok = True
    try:
        if not wait_for_health(server, PORT):
            check("boot", False, "server never answered /health")
            return 1
        for name, args, judge in SCENARIOS:
            run = run_client(PORT, name, args)
            ok = invariants(name, run) and ok
            ok = judge(run) and ok
    finally:
        server.terminate()
        server.wait(timeout=30)
        log.close()

    ok = check_grace_log(TMP + "/server.log") and ok
    ok = check_owned_endpoint() and ok
    return 0 if ok else 1


# From the server log, over every scenario: a turn committed as complete
# turns final no sooner than the grace. A forced commit has none, which the
# push to talk shows with an answer sooner than the grace.
def check_grace_log(path):
    commits, graced, ok = {}, 0, True
    for line in open(path, errors="replace"):
        if "Server] Connection from" in line:
            commits = {}
        m = re.search(r"Turn committed at ([\d.]+)s \(turn (\d+) .*completion ([\d.]+)", line)
        if m:
            commits[m.group(2)] = (float(m.group(1)), float(m.group(3)))
        m = re.search(r"Turn final at ([\d.]+)s \(turn (\d+) ", line)
        if m and m.group(2) in commits:
            at, score = commits.pop(m.group(2))
            if score > 0.0:
                graced += 1
                ok = ok and float(m.group(1)) - at >= GRACE_S - 0.001
    return check("grace log", ok and graced > 0, "%d answers to a complete commit held for the grace" % graced)


def check_owned_endpoint():
    key_file = TMP + "/llm.key"
    with open(key_file, "w") as f:
        f.write(OWNED_KEY + "\n")

    log = open(TMP + "/server-owned.log", "w")
    server = subprocess.Popen(
        [SERVER, "--models", MODELS, "--voices", VOICES, "--host", "127.0.0.1", "--port", str(OWNED_PORT),
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
        run = run_client(OWNED_PORT, "owned", ["--mode", "conversation", "--other-endpoint", "say:" + A, "pause:3"])
    finally:
        server.terminate()
        server.wait(timeout=30)
        log.close()
        os.remove(key_file)

    fixed = json.loads(props)["defaults"].get("llm_fixed") is True
    hidden = not any(secret in props for secret in (OWNED_URL, OWNED_MODEL, OWNED_KEY))
    ok = check("owned props", fixed and hidden, "the endpoint is the server's, its URL, model and key unpublished")
    ok = check("owned models", models == 403, "model list closed with %d" % models) and ok
    # /logs streams the log to every page: the endpoint never appears in it.
    with open(TMP + "/server-owned.log") as f:
        logged = f.read()
    ok = check("owned log", not any(secret in logged for secret in (OWNED_URL, OWNED_MODEL, OWNED_KEY)),
               "the endpoint URL, model and key never logged") and ok
    refused = run.errors()
    ok = check("owned endpoint", any("set by the server" in e for e in refused),
               "another endpoint refused: %s" % (refused[0] if refused else "no error")) and ok
    return invariants("owned", run) and ok


if __name__ == "__main__":
    sys.exit(main())
