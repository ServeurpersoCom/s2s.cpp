#!/bin/bash
# Download the GGUF models used at runtime
#
# Usage: ./models.sh [options]
#   default:   Q8_0 everywhere
#   --asr X:   ASR model, ultra (Parakeet Ultra) or v3 (Parakeet TDT 0.6B v3).
#              The server takes Parakeet Ultra when present, v3 otherwise.
#   --quant X: ASR quant (Q4_K_M, Q5_K_M, Q6_K, Q8_0, F32)
#   --tts X:   TTS quant (Q4_K_M, Q8_0, BF16). Q4_K_M breaks the talker end of
#              speech, so the default stays Q8_0.
#
# The talker is the 1.7b Base one: the voices in voices/ are references
# extracted for its hidden size. The server loads the best quant up to Q8_0.
# The VAD, the turn detector and the echo canceller are small and always ship
# as F32.

set -eu

DIR="models"
VAD_REPO="Serveurperso/Silero-VAD-GGUF"
TURN_REPO="Serveurperso/Smart-Turn-v3-GGUF"
TTS_REPO="Serveurperso/Qwen3-TTS-GGUF"
AEC_REPO="Serveurperso/LocalVQE-GGUF"
ASR="ultra"
QUANT="Q8_0"
TTS_QUANT="Q8_0"

while [ $# -gt 0 ]; do
    case "$1" in
        --asr)    ASR="$2"; shift ;;
        --quant)  QUANT="$2"; shift ;;
        --tts)    TTS_QUANT="$2"; shift ;;
        *)        echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

case "$ASR" in
    ultra) ASR_REPO="Serveurperso/Parakeet-Ultra-GGUF";      ASR_FILE="parakeet-ultra" ;;
    v3)    ASR_REPO="Serveurperso/Parakeet-TDT-0.6b-v3-GGUF"; ASR_FILE="parakeet-tdt-0.6b-v3" ;;
    *)     echo "Unknown ASR: $ASR"; exit 1 ;;
esac

mkdir -p "$DIR"

dl() {
    local repo="$1" file="$2"
    if [ -f "$DIR/$file" ]; then
        echo "[OK] $file"
        return
    fi
    echo "[Download] $file <- $repo"
    hf download --quiet "$repo" "$file" --local-dir "$DIR"
}

dl "$VAD_REPO"  "silero-vad-F32.gguf"
dl "$TURN_REPO" "smart-turn-v3.2-F32.gguf"
dl "$ASR_REPO"  "${ASR_FILE}-${QUANT}.gguf"
dl "$TTS_REPO"  "qwen-talker-1.7b-base-${TTS_QUANT}.gguf"
dl "$TTS_REPO"  "qwen-tokenizer-12hz-Q8_0.gguf"
dl "$AEC_REPO"  "localvqe-v1.3-F32.gguf"
