#!/bin/bash

set -eo pipefail

# The GPU backends only: on the CPU the synthesis is slower than real time,
# so the time windows mean nothing there. qwentts.cpp checks its CPU output.
for backend in CUDA0 Vulkan0; do
    env GGML_BACKEND=$backend ./test-server.py 2>&1 | tee ${backend}-server.log
done
