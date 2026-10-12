#!/bin/bash

set -eo pipefail

for backend in CUDA0 Vulkan0 CPU; do
    ./test-localvqe.py --device $backend 2>&1 | tee ${backend}-localvqe.log
done
