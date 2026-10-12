#!/bin/bash

set -eu

# Multi-GPU: add --device <name> below (CUDA0, CUDA1, Vulkan0, CPU)

# Everything else belongs to the client: mode, endpoint, prompt, voice,
# sampling and turn detection travel in session.update, and their defaults
# are published on /props.

./build/s2s-server \
    --host 0.0.0.0 \
    --port 8088 \
    --models ./models \
    --voices ./voices \
    --origin http://localhost:8088 \
    --origin http://127.0.0.1:8088
