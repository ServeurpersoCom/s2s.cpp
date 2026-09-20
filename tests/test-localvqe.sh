#!/bin/bash

set -eo pipefail

for backend in CUDA0 Vulkan0 CPU; do
    env GGML_BACKEND=$backend ./test-localvqe.py 2>&1 | tee ${backend}-localvqe.log
done
