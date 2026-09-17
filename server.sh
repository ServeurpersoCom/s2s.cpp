#!/bin/bash

set -eu

# Multi-GPU: set GGML_BACKEND to pick a device (CUDA0, CUDA1, Vulkan0...)
#export GGML_BACKEND=CUDA0
#export GGML_BACKEND=Vulkan0

# Everything else belongs to the client: mode, endpoint, prompt, voice,
# sampling and turn detection travel in session.update, and their defaults
# are published on /props.

./build/s2s-server \
    --host 0.0.0.0 \
    --port 8088 \
    --models ./models \
    --origin http://localhost:8088 \
    --llm-host 127.0.0.1:8080
