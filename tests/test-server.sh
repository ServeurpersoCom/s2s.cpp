#!/bin/bash

set -eo pipefail

for backend in CUDA0 Vulkan0 CPU; do
    env GGML_BACKEND=$backend ./test-server.py 2>&1 | tee ${backend}-server.log
done
