#!/bin/bash
# Download the GGUF models used at runtime
#
# Usage: ./models.sh [options]
#   default:   the 1.7b talker, Q8_0 everywhere
#   --talker X: TTS talker size (0.6b, 1.7b). 0.6b fits a small card next to a
#              LLM, 1.7b speaks better.
#   --quant X: ASR quant (Q4_K_M, Q5_K_M, Q6_K, Q8_0, F32)
#   --tts X:   TTS quant (Q4_K_M, Q8_0, BF16). Q4_K_M breaks the talker end of
#              speech, so the default stays Q8_0.
#
# The server loads the largest talker present, at the best quant up to Q8_0.
# The VAD, the turn detector and the echo canceller are small and always ship
# as F32.

set -eu

DIR="models"
ASR_REPO="Serveurperso/Parakeet-TDT-0.6b-v3-GGUF"
VAD_REPO="Serveurperso/Silero-VAD-GGUF"
TURN_REPO="Serveurperso/Smart-Turn-v3-GGUF"
TTS_REPO="Serveurperso/Qwen3-TTS-GGUF"
AEC_REPO="Serveurperso/LocalVQE-GGUF"
QUANT="Q8_0"
TTS_QUANT="Q8_0"
TALKER="1.7b"

while [ $# -gt 0 ]; do
    case "$1" in
        --talker) TALKER="$2"; shift ;;
        --quant)  QUANT="$2"; shift ;;
        --tts)    TTS_QUANT="$2"; shift ;;
        *)        echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

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
dl "$ASR_REPO"  "parakeet-tdt-0.6b-v3-${QUANT}.gguf"
dl "$TTS_REPO"  "qwen-talker-${TALKER}-customvoice-${TTS_QUANT}.gguf"
dl "$TTS_REPO"  "qwen-tokenizer-12hz-Q8_0.gguf"
dl "$AEC_REPO"  "localvqe-v1.3-F32.gguf"
