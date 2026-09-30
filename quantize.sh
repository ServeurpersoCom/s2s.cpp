#!/bin/bash

set -eu

Q="./build/quantize"

quantize() {
    local native="$1" type="$2"
    local out="${native%-*.gguf}-${type}.gguf"
    if [ -f "$out" ]; then
        echo "[Skip] $out"
    else
        $Q "$native" "$out" "$type"
    fi
}

# Parakeet TDT 0.6B v3 and Parakeet Ultra (native F32): every 2D weight
# quantizes, the convolution kernels, the sinusoid table and the relative
# attention biases stay F32.
for native in models/parakeet-tdt-0.6b-v3-F32.gguf models/parakeet-ultra-F32.gguf; do
    for type in Q4_K_M Q5_K_M Q6_K Q8_0; do
        quantize "$native" "$type"
    done
done

# Silero VAD (2 MB) and Smart Turn (8M params) ship as F32 only:
# quantizing them saves nothing and costs turn accuracy.
