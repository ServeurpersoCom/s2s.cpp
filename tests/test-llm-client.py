#!/usr/bin/env python3
# test-llm-client.py: streaming, splitting and cancellation of the LLM client.
#
# The harness carries its own chat completions endpoint, so this checks the
# client and the sentence splitter and never depends on a model being loaded
# somewhere. What matters for the loop: the first synthesis unit leaves well
# before the answer ends, a raised cancel flag stops the request, a failing
# endpoint is reported instead of swallowed, and the sampling of a
# session.update reaches the server.
# Run from the tests/ directory.
#
# Usage:
#     ./test-llm-client.py

import os
import re
import subprocess
import sys

BIN = "../build/test-llm-client"
PORT = "18080"


def check(label, ok, detail):
    print("[Check] %s: %s %s" % (label, detail, "OK" if ok else "FAIL"))
    return ok


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    run = subprocess.run([BIN, PORT], check=True, stdout=subprocess.PIPE, text=True)
    log = run.stdout
    print(log, end="")

    streamed = re.search(r"Streamed (\d+) deltas, (\d+) characters, (\d+) units in ([\d.]+) ms", log)
    first = re.search(r"First unit after ([\d.]+) ms", log)
    cancelled = re.search(r"Cancelled after (\d+) deltas, (\d+) characters, returned (\w+)", log)
    broken = re.search(r"Broken endpoint returned (\w+)", log)
    units = re.findall(r'\[Unit\] \d+: end \d+ "(.*)"', log)
    ends = [int(e) for e in re.findall(r'\[Unit\] \d+: end (\d+) ', log)]
    written = re.search(r"Written (\d+) UTF-16 units, the last unit ends at (\d+)", log)
    parsed = re.search(r"\[Parse\] (.*)", log)

    ok = check("parsed", bool(streamed and first and cancelled and broken and written and parsed),
               "harness reported every stage")
    if not ok:
        return 1

    n_deltas, n_chars, n_units, total_ms = (float(streamed.group(i)) for i in (1, 2, 3, 4))
    first_ms = float(first.group(1))
    cut_chars, returned = int(cancelled.group(2)), cancelled.group(3)

    ok = check("streaming", n_deltas > n_units, "%d deltas for %d units" % (n_deltas, n_units)) and ok
    ok = check("splitting", n_units >= 2, "%d synthesis units" % n_units) and ok
    ok = check("head start", first_ms < total_ms / 2, "first unit at %.1f of %.1f ms" % (first_ms, total_ms)) and ok

    # No unit may carry a markdown marker or a line break into the TTS.
    dirty = [u for u in units if any(c in u for c in "*_`#\n")]
    ok = check("clean units", not dirty, "%d units free of markup" % len(units)) and ok

    # Each unit says where it ends in the written text, further than the one
    # before, and the flush reaches the very end: past it, nothing was left
    # unspoken.
    rising = all(b > a for a, b in zip(ends, ends[1:]))
    ok = check("ends", rising and int(written.group(2)) == int(written.group(1)),
               "unit ends rising, the last one at %s of %s" % (written.group(2), written.group(1))) and ok

    # The same ends a browser computes on the written text, where an accent is
    # one code unit and the emoji two, with every character cut across deltas.
    mixed = "Déjà vu ? Oui 😄. Fin"
    utf16 = lambda text: len(text.encode("utf-16-le")) // 2
    expected = [utf16(mixed[: mixed.index("?") + 1]), utf16(mixed[: mixed.index(".") + 1]), utf16(mixed)]
    split = [int(e) for e in re.findall(r"\[Split\] end (\d+) ", log)]
    ok = check("utf-16", split == expected, "ends %s, expected %s" % (split, expected)) and ok

    # Every sampling field the panel edits reaches the patch, with its value.
    sent = "temperature 0.7 top_p 0.9 top_k 40 min_p 0.05 max_tokens 256 presence 0.5 frequency 0.25 seed 42 timeout 30"
    ok = check("session sampling", parsed.group(1) == sent, parsed.group(1)) and ok

    ok = check("cancel honored", returned == "false", "request reported cancelled") and ok
    ok = check("cancel truncates", cut_chars < n_chars, "%d of %d characters" % (cut_chars, n_chars)) and ok
    ok = check("error surfaced", broken.group(1) == "false", "HTTP 500 reported") and ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
