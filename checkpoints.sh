#!/bin/bash
# Download the source checkpoints converted by ./convert.py
# Usage: ./checkpoints.sh
#   silero-vad    ONNX from onnx-community/silero-vad
#   smart-turn    ONNX from pipecat-ai/smart-turn-v3 (v3.2 GPU export, FP32)
#   parakeet      safetensors and tokenizer from nvidia/parakeet-tdt-0.6b-v3
#   localvqe      v1.3 PyTorch checkpoint from LocalAI-io/LocalVQE
#
# The TTS checkpoints belong to the qwentts submodule: run qwentts.cpp/checkpoints.sh
# there, or fetch the prebuilt GGUF with ./models.sh.

set -eu

DIR="checkpoints"
HF="hf download --quiet"

mkdir -p "$DIR"

dl_file() {
    local repo="$1" file="$2" target="$3"
    if [ -f "$DIR/$target/$file" ]; then
        echo "[OK] $target/$file"
        return
    fi
    echo "[Download] $target/$file <- $repo"
    $HF "$repo" "$file" --local-dir "$DIR/$target"
}

dl_file "onnx-community/silero-vad" "onnx/model.onnx" "silero-vad"
dl_file "pipecat-ai/smart-turn-v3" "smart-turn-v3.2-gpu.onnx" "smart-turn"

for file in config.json generation_config.json processor_config.json model.safetensors tokenizer.json tokenizer_config.json; do
    dl_file "nvidia/parakeet-tdt-0.6b-v3" "$file" "parakeet"
done

dl_file "LocalAI-io/LocalVQE" "localvqe-v1.3-4.8M.pt" "localvqe"

find "$DIR" -name '.cache' -type d -exec rm -rf {} + 2>/dev/null
echo "[Done] Checkpoints ready in $DIR"
echo "[Done] Run: ./convert.py"
