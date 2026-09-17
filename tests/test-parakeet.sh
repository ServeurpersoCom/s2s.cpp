#!/bin/bash

set -eo pipefail

for backend in CUDA0 Vulkan0 CPU; do
    for quant in F32 Q8_0 Q4_K_M; do
        env GGML_BACKEND=$backend ./test-parakeet.py \
            --model ../models/parakeet-tdt-0.6b-v3-${quant}.gguf \
            2>&1 | tee ${backend}-${quant}-parakeet.log
    done
done
