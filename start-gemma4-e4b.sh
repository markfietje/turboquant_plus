#!/bin/bash
#
# Gemma 4 E4B Uncensored + TurboQuant -- M1 16 GB
#
# Model:  HauhauCS/Gemma-4-E4B-Uncensored-Aggressive (IQ4_XS, 4.7 GB)
#         Uncensored via aggressive abliteration
#         4.25-bit importance-matrix quant -- fast decode, strong quality
#
# Architecture: 42 layers, 8 Q heads / 2 KV heads (4:1 GQA), head_dim=512
#               24 full-KV layers + 18 SWA layers (window=512, shared KV)
#               Max context: 131,072 tokens
#
# KV Cache: Asymmetric q8_0-K + turbo4-V (default, best quality)
#   - q8_0 K: guards against GQA amplification of compression error
#   - turbo4 V: 3.8x compression, near-zero PPL impact ("V compression is free")
#   - V-rotation: -43% PPL improvement on Gemma4 Q8 models
#   - Full 131K context uses only ~4.6 GB KV -- fits easily in 16 GB
#
# Memory Budget (M1 16 GB, 131K ctx, q8_0-K + turbo4-V):
#   Model weights (IQ4_XS):   4.7 GB
#   KV cache (131K ctx):      ~4.6 GB  (q8_0-K + turbo4-V, full + SWA layers)
#   Process overhead:          ~0.5 GB
#   macOS:                     ~1.2 GB
#   -- -- -- -- -- -- -- -- -- -- -- --
#   Total:                    ~11.0 GB  -- comfortable on 16 GB M1 Pro
#
# MTP:  Not available -- no E4B GGUF model has MTP layers.
#
# -- Model Options -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# --opus    Use Opus 4.7 fine-tuned uncensored (mradermacher i1-Q6_K, 5.8 GB)
#          Fine-tuned on Claude Opus 4.7 data, then uncensored + imatrix.
#          Must be downloaded first.
#          curl -L -C - -o ~/LocalAI/models/gemma-4-E4B-it-Uncensored-MAX-opus-4.7.i1-Q6_K.gguf \
#            'https://huggingface.co/mradermacher/gemma-4-E4B-it-Uncensored-MAX-opus-4.7-i1-GGUF/resolve/main/gemma-4-E4B-it-Uncensored-MAX-opus-4.7.i1-Q6_K.gguf'
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#
# KV Modes:
#   --quality (default)  q8_0-K + turbo4-V  -- best quality, full 131K context
#   --compact            turbo4-K + turbo4-V -- 3.8x both, saves ~1.4 GB KV
#   --fast               turbo3-K + turbo3-V -- TurboFlash decode kernel
#
# Usage:
#   ./start-gemma4-e4b.sh                          # 131K ctx, best quality
#   ./start-gemma4-e4b.sh --compact                # 131K ctx, compact KV
#   ./start-gemma4-e4b.sh --fast                   # 131K ctx, turbo3 decode
#   ./start-gemma4-e4b.sh -c 32768                 # 32K ctx (faster loading)
#   ./start-gemma4-e4b.sh --opus                   # Opus 4.7 model
#   ./start-gemma4-e4b.sh --opus --compact         # Opus + compact KV
#   ./start-gemma4-e4b.sh --mtp                    # Official + MTP draft (speculative)
#   ./start-gemma4-e4b.sh -p 9090                  # Custom port
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -- Models -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

MODEL_IQ4="/Users/mark/LocalAI/models/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf"
MODEL_OPUS="/Users/mark/LocalAI/models/gemma-4-E4B-it-Uncensored-MAX-opus-4.7.i1-Q6_K.gguf"
MODEL_OFFICIAL="/Users/mark/LocalAI/models/gemma-4-E4B-it-IQ4_XS.gguf"
MODEL_MTP="/Users/mark/LocalAI/models/mtp-gemma-4-E4B-it-Q8_0.gguf"

# -- Binary & Paths -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

BINARY="/Users/mark/LocalAI/turboquant_plus/build/bin/llama-server"
LOG_PATH="/tmp/llama-server-gemma4-e4b.log"
PORT=8080

# -- Model architecture constants (for KV estimation) -- -- -- -- -- -- -- -- -- --

N_LAYERS=42
N_KV_HEADS=2
HEAD_DIM=512
FULL_KV_LAYERS=24   # 42 - 18 shared
SWA_LAYERS=18
SWA_WINDOW=512

# -- KV Cache Profiles -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

QUALITY_K="q8_0"
QUALITY_V="turbo4"

COMPACT_K="turbo4"
COMPACT_V="turbo4"

FAST_K="turbo3"
FAST_V="turbo3"

# -- Sampling (Gemma 4 recommended) -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

TEMP=0.6
TOP_P=0.95
TOP_K=64
REPEAT_PENALTY=1.1

THREADS=8
BATCH_SIZE=512
UBATCH=256
N_PARALLEL=1

# -- Defaults -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

KV_MODE="quality"
USE_OPUS=false
USE_OFFICIAL=false
USE_MTP=false
CUSTOM_CTX=""

# -- Colors -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

# -- macOS-compatible number formatting -- -- -- -- -- -- -- -- -- -- -- -- -- --

comma_fmt() {
    local s=$1 r=
    while [ ${#s} -gt 3 ]; do r=",${s: -3}$r"; s="${s:0:${#s}-3}"; done
    printf '%s' "$s$r"
}

# -- Parse Arguments -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quality)
            KV_MODE="quality"
            shift
            ;;
        --compact)
            KV_MODE="compact"
            shift
            ;;
        --fast)
            KV_MODE="fast"
            shift
            ;;
        --opus)
            USE_OPUS=true
            shift
            ;;
        --official)
            USE_OFFICIAL=true
            shift
            ;;
        --mtp)
            USE_MTP=true
            shift
            ;;
        -c|--context)
            CUSTOM_CTX="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            head -60 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quality|--compact|--fast] [--opus] [-c CTX] [-p PORT]"
            exit 1
            ;;
    esac
done

# -- Apply KV Profile -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

if [ "$KV_MODE" = "quality" ]; then
    CACHE_TYPE_K="$QUALITY_K"
    CACHE_TYPE_V="$QUALITY_V"
    KV_LABEL="quality: q8_0-K + turbo4-V (best intelligence)"
elif [ "$KV_MODE" = "compact" ]; then
    CACHE_TYPE_K="$COMPACT_K"
    CACHE_TYPE_V="$COMPACT_V"
    KV_LABEL="compact: turbo4-K + turbo4-V (3.8x both)"
elif [ "$KV_MODE" = "fast" ]; then
    CACHE_TYPE_K="$FAST_K"
    CACHE_TYPE_V="$FAST_V"
    KV_LABEL="fast: turbo3-K + turbo3-V (TurboFlash decode)"
fi

# -- Select Model -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

MTP_DRAFT_ARGS=()
if [ "$USE_MTP" = true ]; then
    # MTP head was trained on the official model, auto-select it
    MODEL_PATH="$MODEL_OFFICIAL"
    MODEL_LABEL="Official + MTP draft (IQ4_XS + Q8_0 MTP)"
    MODEL_SIZE_LABEL="4.4 + 0.1 GB"
    MODEL_GB=4.5

    if [ ! -f "$MODEL_MTP" ]; then
        echo -e "${RED}X MTP head not found:${RESET} $MODEL_MTP"
        echo "  Download with:"
        echo "  curl -L -o '$MODEL_MTP' \\/"
        echo "    'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/MTP/gemma-4-E4B-it-Q8_0-MTP.gguf'"
        exit 1
    fi
    MTP_DRAFT_ARGS=(-md "$MODEL_MTP" --spec-type draft-mtp)
elif [ "$USE_OFFICIAL" = true ]; then
    MODEL_PATH="$MODEL_OFFICIAL"
    MODEL_LABEL="Unsloth Official gemma-4-E4B-it (IQ4_XS)"
    MODEL_SIZE_LABEL="4.4 GB"
    MODEL_GB=4.4
elif [ "$USE_OPUS" = true ]; then
    MODEL_PATH="$MODEL_OPUS"
    MODEL_LABEL="Opus 4.7 fine-tuned uncensored (i1-Q6_K)"
    MODEL_SIZE_LABEL="5.8 GB"
    MODEL_GB=5.8
else
    MODEL_PATH="$MODEL_IQ4"
    MODEL_LABEL="HauhauCS Uncensored Aggressive (IQ4_XS)"
    MODEL_SIZE_LABEL="4.7 GB"
    MODEL_GB=4.7
fi

# Default context: full 131K -- the model supports it and it fits easily
CONTEXT="${CUSTOM_CTX:-131072}"

# -- Estimate KV Memory -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# Bytes per element for each cache type (vs f16=2.0)
bpe_of() {
    case "$1" in
        f16)    printf '2.0' ;;
        q8_0)   printf '1.0' ;;
        q4_0)   printf '0.5' ;;
        turbo4) printf '0.5263' ;;   # 2.0 / 3.8
        turbo3) printf '0.4348' ;;   # 2.0 / 4.6
        turbo2) printf '0.3125' ;;   # 2.0 / 6.4
        *)      printf '2.0' ;;
    esac
}

K_BPE=$(bpe_of "$CACHE_TYPE_K")
V_BPE=$(bpe_of "$CACHE_TYPE_V")

# Full-KV layers: store all ctx tokens
# SWA layers: store at most SWA_WINDOW tokens (shared)
ELEM_SIZE=$(awk "BEGIN {printf \"%.2f\", $K_BPE + $V_BPE}")
KV_FULL=$(awk "BEGIN {printf \"%.2f\", $FULL_KV_LAYERS * $N_KV_HEADS * $HEAD_DIM * $ELEM_SIZE * $CONTEXT / 1073741824}")
KV_SW=$(awk "BEGIN {printf \"%.2f\", $SWA_LAYERS * $N_KV_HEADS * $HEAD_DIM * $ELEM_SIZE * $SWA_WINDOW / 1073741824}")
KV_GB=$(awk "BEGIN {printf \"%.2f\", $KV_FULL + $KV_SW}")

TOTAL_GB=$(awk "BEGIN {printf \"%.1f\", $MODEL_GB + $KV_GB + 0.5 + 1.2}")
NEEDED_MB=$(awk "BEGIN {printf \"%d\", ($TOTAL_GB + 1.0) * 1024}")

# -- Preflight -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

echo ""
echo -e "${CYAN}${BOLD}  +================================================================+${RESET}"
echo -e "${CYAN}${BOLD}  |  Gemma 4 E4B Uncensored + TurboQuant                          |${RESET}"
echo -e "${CYAN}${BOLD}  |  M1 16 GB * Metal * $(comma_fmt "$CONTEXT") ctx                        |${RESET}"
echo -e "${CYAN}${BOLD}  +================================================================+${RESET}"
echo ""

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}X Model not found:${RESET} $MODEL_PATH"
    echo ""
    if [ "$USE_OPUS" = true ]; then
        echo "  Download with:"
        echo "  curl -L -C - -o '$MODEL_OPUS' \\"
        echo "    'https://huggingface.co/mradermacher/gemma-4-E4B-it-Uncensored-MAX-opus-4.7-i1-GGUF/resolve/main/gemma-4-E4B-it-Uncensored-MAX-opus-4.7.i1-Q6_K.gguf'"
    fi
    exit 1
fi
echo -e "${GREEN}*${RESET} Model:  ${MODEL_SIZE_LABEL}  ${MODEL_LABEL}"

if [ ! -f "$BINARY" ]; then
    echo -e "${RED}X Binary not found:${RESET} $BINARY"
    echo "  Build with: cd ~/LocalAI/turboquant_plus && cmake -B build && cmake --build build"
    exit 1
fi
echo -e "${GREEN}*${RESET} Binary: $BINARY"

# -- VRAM Unlock -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

WIRED=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "0")
if [ "$WIRED" -lt "$NEEDED_MB" ]; then
    echo -e "${YELLOW}!${RESET} Unlocking VRAM to ${NEEDED_MB} MB (requires sudo)..."
    sudo sysctl "iogpu.wired_limit_mb=$NEEDED_MB" > /dev/null 2>&1
    WIRED=$NEEDED_MB
fi
echo -e "${GREEN}*${RESET} VRAM:   iogpu.wired_limit_mb=${WIRED} (~${TOTAL_GB} GB budget)"

# -- Kill Existing -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

EXISTING=$(lsof -ti:"$PORT" 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo -e "${YELLOW}>${RESET} Stopping existing server on :$PORT"
    kill "$EXISTING" 2>/dev/null || true
    sleep 1
fi

# -- Clear Log -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

: > "$LOG_PATH"

# -- Launch -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

echo ""
echo -e "  ${BOLD}--- Config --- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --${RESET}"
echo -e "  ${BOLD}Model:${RESET}      ${MODEL_LABEL}"
echo -e "  ${BOLD}Context:${RESET}    $(comma_fmt "$CONTEXT") tokens"
echo -e "  ${BOLD}KV:${RESET}        ${KV_LABEL}"
echo -e "  ${BOLD}KV memory:${RESET} ~${KV_GB} GB (${KV_FULL} GB full + ${KV_SW} GB SWA)"
echo -e "  ${BOLD}Sampling:${RESET}  temp=${TEMP} top_p=${TOP_P} top_k=${TOP_K}"
echo -e "  ${BOLD}Reasoning:${RESET} auto (unrestricted budget, format=auto)"
if [ "$USE_MTP" = true ]; then
    echo -e "  ${BOLD}MTP:${RESET}       draft-mtp, n_max=3 (speculative decoding)"
fi
echo -e "  ${BOLD}Port:${RESET}      ${PORT}"
echo ""
echo -e "${DIM}  Loading model...${RESET}"

nohup "$BINARY" \
    -m "$MODEL_PATH" \
    "${MTP_DRAFT_ARGS[@]}" \
    -c "$CONTEXT" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --flash-attn on \
    --jinja \
    -ngl 99 \
    -t "$THREADS" \
    -tb "$BATCH_SIZE" \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --repeat-penalty "$REPEAT_PENALTY" \
    -ub "$UBATCH" \
    -np "$N_PARALLEL" \
    --cache-reuse 512 \
    --port "$PORT" \
    --host 0.0.0.0 \
    --reasoning on \
    --reasoning-budget -1 \
    --reasoning-format auto \
    >> "$LOG_PATH" 2>&1 &

PID=$!
echo -e "${GREEN}*${RESET} Started (PID $PID)"

# -- Wait for Ready -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

echo -n "  Waiting"
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo ""
        echo -e "${GREEN}${BOLD}  * Server ready${RESET}"
        echo ""
        echo -e "  ${BOLD}Chat UI:${RESET}     http://127.0.0.1:${PORT}"
        echo -e "  ${BOLD}OpenAI API:${RESET}  http://127.0.0.1:${PORT}/v1"
        echo -e "  ${BOLD}Health:${RESET}      http://127.0.0.1:${PORT}/health"
        echo ""
        echo -e "  ${BOLD}Stop:${RESET}        kill $PID"
        echo -e "  ${BOLD}Log:${RESET}         tail -f $LOG_PATH"
        echo ""
        echo -e "${DIM}  --- Other Modes --- -- -- -- -- -- -- -- -- -- -- -- -- -- --${RESET}"
        echo -e "${DIM}  $0 --compact            Compact KV (turbo4-K+V, saves ~1.4 GB)${RESET}"
        echo -e "${DIM}  $0 --fast               Fast decode (turbo3-K+V, TurboFlash)${RESET}"
        echo -e "${DIM}  $0 --opus               Opus 4.7 fine-tuned model${RESET}"
        echo -e "${DIM}  $0 --mtp                Official + MTP draft (speculative)${RESET}"
        echo -e "${DIM}  $0 -c 32768             Shorter context (faster loading)${RESET}"
        echo -e "${DIM}  $0 -p 9090              Custom port${RESET}"
        echo ""
        exit 0
    fi

    if ! kill -0 "$PID" 2>/dev/null; then
        echo ""
        echo ""
        echo -e "${RED}X Server crashed during startup.${RESET}"
        echo ""
        echo -e "  Last log entries:"
        tail -20 "$LOG_PATH" | sed 's/^/  /'
        echo ""
        echo -e "  Common fixes:"
        echo -e "    * Out of memory  -> ${BOLD}$0 --compact${RESET} (turbo4-KV)"
        echo -e "    * Out of memory  -> ${BOLD}$0 -c 32768${RESET} (shorter context)"
        echo -e "    * VRAM locked    -> ${BOLD}sudo sysctl iogpu.wired_limit_mb=15360${RESET}"
        echo -e "    * Bad GGUF       -> ${BOLD}ls -lh $MODEL_PATH${RESET}"
        echo ""
        exit 1
    fi

    sleep 2
    echo -n "."
done

echo ""
echo ""
echo -e "${YELLOW}! Not ready after 3 minutes.${RESET}"
echo ""
echo "  Log:   tail -f $LOG_PATH"
echo "  Check: curl http://127.0.0.1:${PORT}/health"
echo "  Kill:  kill $PID"
echo ""
exit 1
