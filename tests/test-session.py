#!/usr/bin/env python3
# test-session.py: invariants of the turn state machine.
#
# There is no upstream reference for this layer: the state machine is ours.
# So the harness runs the timeline and checks the properties a conversation
# has to hold, the ones a regression in the thresholds or in the buffering
# would break first. Run from the tests/ directory.
#
# Usage:
#     ./test-session.py
#     GGML_BACKEND=CPU ./test-session.py

import os
import re
import subprocess
import sys

BIN = "../build/test-session"
VAD = "../models/silero-vad-F32.gguf"
TURN = "../models/smart-turn-v3.2-F32.gguf"
ASR = "../models/parakeet-tdt-0.6b-v3-Q8_0.gguf"
WAV = "../examples/freeman.wav"
TMP = "tmp"
GRACE_S = 0.8  # reopen_grace_ms at its default
WINDOW_S = 0.032

LINE = re.compile(
    r"\[Turn\]\s+([\d.]+)s\s+turn (\d+) rev (\d+)\s+(\w+)"
    r"(?:\s+score ([\d.]+))?(?:\s+([\d.]+)s\s+\"(.*)\")?"
)


def check(label, ok, detail):
    print("[Check] %s: %s %s" % (label, detail, "OK" if ok else "FAIL"))
    return ok


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(TMP, exist_ok=True)

    subprocess.run(
        [BIN, "--vad", VAD, "--turn", TURN, "--asr", ASR, "--file", WAV, "--out", TMP + "/session.txt"],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    events = []
    for line in open(TMP + "/session.txt"):
        match = LINE.match(line.strip())
        if match:
            time, turn, rev, name, score, seconds, text = match.groups()
            events.append(
                {
                    "time": float(time),
                    "turn": int(turn),
                    "rev": int(rev),
                    "event": name,
                    "score": float(score) if score else None,
                    "seconds": float(seconds) if seconds else None,
                    "text": text,
                }
            )

    commits = [e for e in events if e["event"] == "turn_committed"]
    starts = [e for e in events if e["event"] == "speech_started"]

    ok = check("parsed", len(events) > 0, "%d events" % len(events))
    ok = check("turns", len(commits) > 0, "%d committed, %d speech starts" % (len(commits), len(starts))) and ok

    # A commit always carries audio and a transcript, otherwise the recognizer
    # would be handed silence.
    empty = [c for c in commits if not c["text"] or c["seconds"] is None or c["seconds"] <= 0.0]
    ok = check("payload", not empty, "%d commits with audio and text" % (len(commits) - len(empty))) and ok

    # Time never goes backwards and turn ids only grow.
    times = [e["time"] for e in events]
    turns = [e["turn"] for e in events]
    ok = check("ordering", times == sorted(times), "timeline monotonic") and ok
    ok = check("turn ids", turns == sorted(turns), "turn ids monotonic") and ok

    # Every turn opens before it closes, and a reopen raises the revision.
    opened = set()
    sequence = True
    for e in events:
        if e["event"] == "speech_started":
            opened.add(e["turn"])
        elif e["turn"] not in opened:
            sequence = False
    ok = check("sequence", sequence, "no event before its turn opened") and ok

    reopened = [e for e in events if e["event"] == "turn_reopened"]
    revisions = all(
        any(s["turn"] == r["turn"] and s["rev"] == r["rev"] + 1 for s in starts) for r in reopened
    )
    ok = check("revisions", revisions, "%d reopenings each followed by a revision" % len(reopened)) and ok

    # A commit the classifier judged complete stays silent for the grace, then
    # turns final, unless the next turn opens first; a forced commit is final
    # at once.
    grace_ok, n_graced, n_forced = True, 0, 0
    for c in commits:
        final = next((e for e in events if e["event"] == "turn_final" and e["turn"] == c["turn"]), None)
        later = next((e for e in events if e["event"] == "speech_started" and e["turn"] > c["turn"]), None)
        if c["score"] == 0.0:
            n_forced += 1
            grace_ok = grace_ok and final is not None and final["time"] == c["time"]
        elif final is not None:
            n_graced += 1
            grace_ok = grace_ok and final["time"] - c["time"] >= GRACE_S - WINDOW_S
        else:
            grace_ok = grace_ok and (later is not None or c is commits[-1])
    ok = check("grace", grace_ok, "%d commits final after the grace, %d forced final at once" % (n_graced, n_forced)) and ok

    for c in commits:
        print('[Turn] %.2fs  turn %d  %.2fs  "%s"' % (c["time"], c["turn"], c["seconds"], c["text"]))

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
