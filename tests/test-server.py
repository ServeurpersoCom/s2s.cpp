#!/usr/bin/env python3
# test-server.py: the behavior of s2s-server, every case at once.
#
# Each case is one scripted client, test-server, with a mock endpoint in
# process. All cases run in parallel, one connection each, against a server
# that batches them, and the checks judge each timeline in windows relative
# to its own events.
#
# The core is one pattern: say A, pause P, say B, with an endpoint slow to
# start. P falls in one of three brackets of the time since the commit of A:
#
#     grace          before the grace ran out: the same turn
#     before answer  after the grace, before any answer: still the same turn
#     answer heard   after the answer started playing: a new turn, a barge-in
#
# Around it: a push to talk, a broken endpoint, an endpoint that answers
# nothing, an endpoint nothing answers at all, a room whose echo the server
# cancels, and a server that owns its endpoint.
#
# The passages of examples/freeman.wav, cut on their speech:
#     A  0.0 to 5.9 s    "If you go into different cultures, ... concepts of creation."
#     B  6.0 to 11.5 s   "They have their own creation story ... afterlife is."
#     C  11.9 to 16.0 s  "Um where you go, what you do, who you're gonna be with, you know."
# A resumption lands about P + 0.13 s after the commit of A.
#
# Every case also checks two invariants: each response.created is closed by
# exactly one response.done or response.cancelled, and the transcripts of a
# turn come in a row. The timelines land in tmp/.
# Run from the tests/ directory.
#
# Usage:
#     ./test-server.py
#     GGML_BACKEND=Vulkan0 ./test-server.py

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
MAX_BATCH = "8"
TMP = "tmp"

A, B, C = "0-5.9", "6-11.5", "11.9-16"
UNREACHABLE = "http://10.255.255.1:9/v1"  # a private address with nothing behind it: the connection hangs
GRACE_S = 0.8     # reopen_grace_ms at its default
REACTION_S = 1.5  # longest a talk over may take to stop the playback and the response
ECHO_WORDS = ("different", "cultures", "creation", "afterlife")

# The endpoint of the brackets: its first sentence is ready about 4.5 s after
# the commit, and it is still writing when a talk over comes.
SLOW = ["--mode", "conversation", "--llm", "v1", "--llm-first-ms", "3000", "--llm-token-ms", "250"]
FAST = ["--mode", "conversation", "--llm", "v1"]

BOOT_TIMEOUT_S = 120

LINE = re.compile(r"\[(Event|Client|Mock)\]\s+([\d.]+)s\s+(\S+)(.*)")


def check(label, ok, detail):
    print("[Check] %s: %s %s" % (label, detail, "OK" if ok else "FAIL"))
    return ok


class Timeline:
    def __init__(self, out):
        self.lines = [(m.group(1), float(m.group(2)), m.group(3), m.group(4).strip())
                      for m in map(LINE.match, out.splitlines()) if m]

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

    def turns(self):
        return sorted(set(item for _, item, _ in self.items()))

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


# Nothing was heard when B came: one turn, recognized whole, its first
# request aborted, one message with both sentences reaching the model, and
# no sound before the speaker is done.
def check_same_turn(label, run):
    items = run.items()
    ended = run.said_end(B)
    audio = run.first("Event", "response.output_audio.delta")
    requests = run.at("Mock", "request")
    last = requests[-1][1] if requests else ""
    ok = check("%s one turn" % label, len(run.turns()) == 1 and len(items) >= 2,
               "%d transcripts, turns %s" % (len(items), run.turns()))
    ok = check("%s one message" % label, last.startswith("1 user messages") and "creation" in last and
               "afterlife" in last, "last request: %s" % last) and ok
    ok = check("%s aborted" % label, bool(run.at("Mock", "aborted")), "the request of A aborted") and ok
    return check("%s silent" % label, audio is not None and ended is not None and audio >= ended,
                 "first audio %.2fs after B ended" % ((audio or 0.0) - (ended or 0.0))) and ok


# A talk over an answer playing opens a new turn. An answer still being
# written has its response and its endpoint request stopped at once; its
# playback may run dry between two sentences, so the flush is judged on a
# loopback answer, sent whole before anyone talks over.
def check_barge_in(label, run, passage=B, words="afterlife", writing=True):
    t0 = run.said(passage)
    if t0 is None:
        return check(label, False, "no talk over, no answer was heard")
    flushed = run.first("Client", "playback", t0)
    cancelled = run.first("Event", "response.cancelled", t0)
    heard = [text for t, _, text in run.items() if t >= t0]
    ok = check("%s new turn" % label, len(run.turns()) == 2, "turns %s" % run.turns())
    if not writing:
        ok = check("%s playback" % label, flushed is not None and flushed - t0 <= REACTION_S,
                   "playback flushed %.2fs after the talk over" % ((flushed or t0) - t0)) and ok
    else:
        ok = check("%s response" % label, cancelled is not None and cancelled - t0 <= REACTION_S,
                   "response cancelled %.2fs after the talk over" % ((cancelled or t0) - t0)) and ok
        ok = check("%s request" % label, bool(run.at("Mock", "aborted", t0)), "the endpoint request aborted") and ok
    return check("%s transcript" % label, any(words in text.lower() for text in heard),
                 "the talk over heard: %s" % (heard[0] if heard else "nothing")) and ok


# A commit mid sentence and the microphone cut: the answer still comes, with
# no grace.
def check_push_to_talk(label, run):
    committed = run.first("Client", "commit")
    audio = run.first("Event", "response.output_audio.delta", committed or 0.0)
    return check("%s answer" % label, committed is not None and audio is not None and audio - committed < GRACE_S,
                 "first audio %.2fs after the commit, no grace" % ((audio or 0.0) - (committed or 0.0)))


# The model thinks and ends without a word: the answer closes as done, with
# no sound, the client is told the model answered nothing, and the log says
# what came back.
def check_empty(label, run):
    done = run.at("Event", "response.done")
    ok = check("%s done" % label, bool(done) and not run.at("Event", "response.cancelled"),
               "%d answers closed as done, none cancelled" % len(done))
    ok = check("%s error" % label, any("answered nothing" in e for e in run.errors()),
               "the error names the empty answer") and ok
    ok = check("%s silent" % label, run.first("Event", "response.output_audio.delta") is None, "no audio") and ok
    with open(TMP + "/server.log", errors="replace") as f:
        logged = re.findall(r"-LLM\] Answer of 0 bytes, (\d+) bytes of reasoning dropped, finish_reason stop", f.read())
    return check("%s log" % label, any(int(n) > 0 for n in logged),
                 "the log says the answer was empty after its reasoning") and ok


# The endpoint is an address nothing answers: the request stays stuck in its
# connection, which no cancel can reach. The words of the user are recognized
# all the same, as fast as ever, and the answer fails within the timeout.
def check_unreachable(label, run):
    items = run.items()
    ended = run.said_end(B)
    after = [t for t, _, _ in items if ended is not None and t >= ended - 0.5]
    ok = check("%s live" % label, bool(after) and after[0] - ended <= 1.0,
               "revision %.2fs after B ended, the endpoint stuck" % ((after[0] - ended) if after else -1.0))
    whole = items[-1][2].lower() if items else ""
    ok = check("%s whole" % label, "creation" in whole and "afterlife" in whole, "last transcript: %s" % whole) and ok
    errors = run.errors()
    return check("%s error" % label, bool(errors), "the answer failed: %s" % (errors[0] if errors else "no error")) and ok


def check_broken(label, run):
    errors = run.errors()
    return check("%s error" % label, bool(errors) and all("model not loaded" in e for e in errors),
                 "%d answers told: %s" % (len(errors), errors[0] if errors else "nothing"))


# A laptop on speakers: the answer comes back into the microphone, C talks
# over it, and the echo canceller keeps the assistant out of what is heard.
def check_room(label, run):
    ok = check_barge_in(label, run, C, "where you go", writing=False)
    t0 = run.said(C) or 0.0
    echo = [text for t, _, text in run.items() if t >= t0 and any(w in text.lower() for w in ECHO_WORDS)]
    return check("%s echo" % label, not echo, "no word of the assistant in what was heard") and ok


CASES = [
    ("grace 0.2", SLOW + ["say:" + A, "pause:0.2", "say:" + B, "pause:3"], check_same_turn),
    ("grace 0.4", SLOW + ["say:" + A, "pause:0.4", "say:" + B, "pause:3"], check_same_turn),
    ("before answer 1.0", SLOW + ["say:" + A, "pause:1.0", "say:" + B, "pause:3"], check_same_turn),
    ("before answer 2.5", SLOW + ["say:" + A, "pause:2.5", "say:" + B, "pause:3"], check_same_turn),
    ("heard 1000", SLOW + ["say:" + A, "heard:1000", "say:" + B, "pause:3"], check_barge_in),
    ("heard 2000", SLOW + ["say:" + A, "heard:2000", "say:" + B, "pause:3"], check_barge_in),
    ("push to talk", FAST + ["say:0-3", "commit", "mute:3"], check_push_to_talk),
    ("broken endpoint", ["--mode", "conversation", "--llm", "broken", "say:" + A, "pause:3"], check_broken),
    ("empty answer", ["--mode", "conversation", "--llm", "empty", "say:" + A, "pause:3"], check_empty),
    ("unreachable endpoint", ["--mode", "conversation", "--llm-url", UNREACHABLE, "--llm-timeout", "3", "say:" + A,
                              "pause:1.5", "say:" + B, "pause:3"], check_unreachable),
    ("room", ["--mode", "loopback", "--echo", "server", "--room", "say:" + A, "heard:1000", "say:" + C, "pause:3"],
     check_room),
]


def start_server(port, log_name, extra):
    log = open("%s/%s" % (TMP, log_name), "w")
    process = subprocess.Popen(
        [SERVER, "--models", MODELS, "--voices", VOICES, "--host", "127.0.0.1", "--port", str(port),
         "--max-batch", MAX_BATCH] + extra,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return process, log


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


def start_client(port, args):
    return subprocess.Popen([CLIENT, "ws://127.0.0.1:%d/v1/realtime" % port, WAV] + args, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)


def finish_client(process, name):
    out = process.communicate()[0]
    with open("%s/%s.txt" % (TMP, name.replace(" ", "-")), "w") as f:
        f.write(out)
    return Timeline(out)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(TMP, exist_ok=True)

    key_file = TMP + "/llm.key"
    with open(key_file, "w") as f:
        f.write(OWNED_KEY + "\n")

    servers = [
        start_server(PORT, "server.log", []),
        start_server(OWNED_PORT, "server-owned.log",
                     ["--llm-url", OWNED_URL, "--llm-model", OWNED_MODEL, "--llm-key-file", key_file]),
    ]
    ok = True
    try:
        if not all(wait_for_health(process, port) for (process, _), port in zip(servers, (PORT, OWNED_PORT))):
            check("boot", False, "a server never answered /health")
            return 1

        clients = [(name, start_client(PORT, args), judge) for name, args, judge in CASES]
        owned = start_client(OWNED_PORT, ["--mode", "conversation", "--other-endpoint", "say:" + A, "pause:3"])
        owned_props, owned_models = owned_routes()

        for name, process, judge in clients:
            run = finish_client(process, name)
            ok = invariants(name, run) and ok
            ok = judge(name, run) and ok
        run = finish_client(owned, "owned")
        ok = invariants("owned", run) and ok
        ok = check_owned(run, owned_props, owned_models) and ok
    finally:
        for process, log in servers:
            process.terminate()
            process.wait(timeout=30)
            log.close()
        os.remove(key_file)

    ok = check_grace_log(TMP + "/server.log") and ok
    return 0 if ok else 1


# From the server log, over every case: a turn committed as complete turns
# final no sooner than the grace. A forced commit has none, which the push to
# talk shows with an answer sooner than the grace.
def check_grace_log(path):
    commits, graced, ok = {}, 0, True
    for line in open(path, errors="replace"):
        m = re.search(r"\[Reader-(\d+)-Session\] Turn committed at ([\d.]+)s \(turn (\d+) .*completion ([\d.]+)", line)
        if m:
            commits[(m.group(1), m.group(3))] = (float(m.group(2)), float(m.group(4)))
        m = re.search(r"-(\d+)-Session\] Turn final at ([\d.]+)s \(turn (\d+) ", line)
        if m and (m.group(1), m.group(3)) in commits:
            at, score = commits.pop((m.group(1), m.group(3)))
            if score > 0.0:
                graced += 1
                ok = ok and float(m.group(2)) - at >= GRACE_S - 0.001
    return check("grace log", ok and graced > 0, "%d answers to a complete commit held for the grace" % graced)


def owned_routes():
    base = "http://127.0.0.1:%d" % OWNED_PORT
    props = urllib.request.urlopen(base + "/props", timeout=5).read().decode()
    try:
        urllib.request.urlopen(urllib.request.Request(base + "/v1/models", data=b"{}", method="POST"), timeout=5)
        models = 200
    except urllib.error.HTTPError as e:
        models = e.code
    return props, models


# A server that owns its endpoint: nothing of it published or logged, the
# model list closed, another endpoint refused.
def check_owned(run, props, models):
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
    return check("owned endpoint", any("set by the server" in e for e in refused),
                 "another endpoint refused: %s" % (refused[0] if refused else "no error")) and ok


if __name__ == "__main__":
    sys.exit(main())
