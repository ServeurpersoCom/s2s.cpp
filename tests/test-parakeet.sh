#!/bin/bash

set -eo pipefail

for model in parakeet-tdt-0.6b-v3:parakeet parakeet-ultra:parakeet-ultra; do
    gguf=${model%:*}
    ckpt=${model#*:}
    for backend in CUDA0 Vulkan0 CPU; do
        for quant in F32 Q8_0 Q4_K_M; do
            env GGML_BACKEND=$backend ./test-parakeet.py \
                --model ../models/${gguf}-${quant}.gguf \
                --checkpoint ../checkpoints/${ckpt} \
                2>&1 | tee ${backend}-${quant}-${ckpt}.log
        done
    done
done
