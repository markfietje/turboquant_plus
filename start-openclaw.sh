#!/bin/bash
#
# TurboQuant+ Server for OpenClaw
#
# Best local inference: MTP speculative decoding + TurboQuant KV cache
# Optimized for Apple Silicon M1 16 GB
#
# Architecture:
#   Gemma 4 E4B (4.7 GB IQ4_XS) + MTP draft head (94 MB Q8_0)
#   42 layers, 8 Q / 2 KV heads (4:1 GQA), head_dim=512
#   24 full-KV + 18 SWA (window=512, shared KV)
#
# Performance stack:
#   - MTP speculative: 3-token lookahead, verified against target
#   - TurboQuant KV: 3.8x compression, <0.1 PPL impact
#   - Flash Attention: fused Q*K^T softmax
#   - Metal GPU: full offload (99 layers)
#   - Cache reuse: 512-token prompt prefix sharing
#   - Reasoning: auto-detect, unrestricted budget
#
# KV Modes:
#   --quality (default)  q8_0-K + turbo4-V  best quality, 131K ctx
#   --compact            turbo4-K + turbo4-V  3.8x both, saves ~1.4 GB
#   --fast               turbo3-K + turbo3-V  TurboFlash decode kernel
#   --mtp-off            disable MTP speculative decoding
#
# Usage:
#   ./start-openclaw.sh                    # 131K ctx, MTP, turbo4-V
#   ./start-openclaw.sh --compact          # compact KV
#   ./start-openclaw.sh --mtp-off          # no speculative decoding
#   ./start-openclaw.sh -c 32768           # shorter context
#   ./start-openclaw.sh --reasoning-off    # disable thinking/reasoning
#

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────

MODEL_IQ4="/Users/mark/LocalAI/models/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf"
MODEL_OPUS="/Users/mark/LocalAI/models/gemma-4-E4B-it-Uncensored-MAX-opus-4.7.i1-Q6_K.gguf"
MODEL_MTP="/Users/mark/LocalAI/models/mtp-gemma-4-E4B-it-Q8_0.gguf"
BINARY="/Users/mark/LocalAI/turboquant_plus/build/bin/llama-server"
LOG_PATH="/tmp/llama-server-openclaw.log"
PORT=8080

# ── Defaults ─────────────────────────────────────────────────────────────────

KV_MODE="quality"
USE_OPUS=false
USE_MTP=true
USE_REASONING=true
CUSTOM_CTX=""
N_PARALLEL=1

# ── Parse Arguments ──────────────────────────────────────────────────────────

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
        --mtp-off)
            USE_MTP=false
            shift
            ;;
        --reasoning-off)
            USE_REASONING=false
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
        -n|--parallel)
            N_PARALLEL="$2"
            shift 2
            ;;
        -h|--help)
            head -40 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quality|--compact|--fast] [--mtp-off] [--opus] [-c CTX] [-p PORT]"
            exit 1
            ;;
    esac
done

# ── Apply KV Profile ─────────────────────────────────────────────────────────

if [ "$KV_MODE" = "quality" ]; then
    CACHE_TYPE_K="q8_0"
    CACHE_TYPE_V="turbo4"
    KV_LABEL="q8_0-K + turbo4-V"
elif [ "$KV_MODE" = "compact" ]; then
    CACHE_TYPE_K="turbo4"
    CACHE_TYPE_V="turbo4"
    KV_LABEL="turbo4-K + turbo4-V"
elif [ "$KV_MODE" = "fast" ]; then
    CACHE_TYPE_K="turbo3"
    CACHE_TYPE_V="turbo3"
    KV_LABEL="turbo3-K + turbo3-V"
fi

# ── Select Model ─────────────────────────────────────────────────────────────

MTP_ARGS=()
if [ "$USE_MTP" = true ]; then
    if [ ! -f "$MODEL_MTP" ]; then
        echo "MTP head not found: $MODEL_MTP"
        echo "Downloading..."
        curl -L -o "$MODEL_MTP" \
            'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/MTP/gemma-4-E4B-it-Q8_0-MTP.gguf'
    fi
    MTP_ARGS=(-md "$MODEL_MTP" --spec-type draft-mtp --spec-draft-n-max 3)
fi

if [ "$USE_OPUS" = true ]; then
    MODEL_PATH="$MODEL_OPUS"
    MODEL_LABEL="Opus 4.7 Uncensored (i1-Q6_K)"
    MODEL_GB=5.8
else
    MODEL_PATH="$MODEL_IQ4"
    MODEL_LABEL="HauhauCS Uncensored Aggressive (IQ4_XS)"
    MODEL_GB=4.7
fi

CONTEXT="${CUSTOM_CTX:-131072}"

# ── Sampling (Gemma 4 recommended) ───────────────────────────────────────────

TEMP=0.6
TOP_P=0.95
TOP_K=64
REPEAT_PENALTY=1.1
THREADS=8
BATCH_SIZE=512
UBATCH=256

# ── Reasoning ────────────────────────────────────────────────────────────────

REASONING_ARGS=()
if [ "$USE_REASONING" = true ]; then
    REASONING_ARGS=(--reasoning on --reasoning-budget -1 --reasoning-format auto)
fi

# ── Colors ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

comma_fmt() {
    local s=$1 r=
    while [ ${#s} -gt 3 ]; do r=",${s: -3}$r"; s="${s:0:${#s}-3}"; done
    printf '%s' "$s$r"
}

# ── Preflight ────────────────────────────────────────────────────────────────

echo ""
echo -e "${CYAN}${BOLD}  +============================================================+${RESET}"
echo -e "${CYAN}${BOLD}  |  TurboQuant+ Server for OpenClaw                           |${RESET}"
echo -e "${CYAN}${BOLD}  |  Metal + MTP + Flash Attn + $(comma_fmt "$CONTEXT") ctx                 |${RESET}"
echo -e "${CYAN}${BOLD}  +============================================================+${RESET}"
echo ""

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}X Model not found:${RESET} $MODEL_PATH"
    exit 1
fi
echo -e "${GREEN}*${RESET} Model:   $(du -sh "$MODEL_PATH" | cut -f1) $MODEL_LABEL"

if [ ! -f "$BINARY" ]; then
    echo -e "${RED}X Binary not found:${RESET} $BINARY"
    exit 1
fi
echo -e "${GREEN}*${RESET} Binary:  $BINARY"

if [ "$USE_MTP" = true ]; then
    echo -e "${GREEN}*${RESET} MTP:     draft-mtp, n_max=3 ($(du -sh "$MODEL_MTP" | cut -f1))"
fi

# ── VRAM ─────────────────────────────────────────────────────────────────────

WIRED=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "0")
NEEDED=14336
if [ "$WIRED" -lt "$NEEDED" ]; then
    echo -e "${YELLOW}!${RESET} Unlocking VRAM to ${NEEDED} MB (requires sudo)..."
    sudo sysctl "iogpu.wired_limit_mb=$NEEDED" > /dev/null 2>&1
fi
WIRED=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "?")
echo -e "${GREEN}*${RESET} VRAM:    iogpu.wired_limit_mb=${WIRED}"

# ── Kill Existing ────────────────────────────────────────────────────────────

EXISTING=$(lsof -ti:"$PORT" 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo -e "${YELLOW}>${RESET} Stopping existing server on :$PORT"
    kill "$EXISTING" 2>/dev/null || true
    sleep 1
fi

# ── Launch ───────────────────────────────────────────────────────────────────

: > "$LOG_PATH"

echo ""
echo -e "  ${BOLD}KV:${RESET}        ${KV_LABEL}"
echo -e "  ${BOLD}Context:${RESET}    $(comma_fmt "$CONTEXT") tokens"
echo -e "  ${BOLD}Parallel:${RESET}   ${N_PARALLEL}"
echo -e "  ${BOLD}Sampling:${RESET}  temp=${TEMP} top_p=${TOP_P} top_k=${TOP_K}"
echo -e "  ${BOLD}Reasoning:${RESET} ${USE_REASONING}"
echo ""
echo -e "${DIM}  Loading...${RESET}"

nohup "$BINARY" \
    -m "$MODEL_PATH" \
    "${MTP_ARGS[@]}" \
    -c "$CONTEXT" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --flash-attn on \
    --jinja \
    -ngl 99 \
    -t "$THREADS" \
    -tb "$BATCH_SIZE" \
    -ub "$UBATCH" \
    -np "$N_PARALLEL" \
    --cache-reuse 512 \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --repeat-penalty "$REPEAT_PENALTY" \
    "${REASONING_ARGS[@]}" \
    --port "$PORT" \
    --host 127.0.0.1 \
    >> "$LOG_PATH" 2>&1 &

PID=$!
echo -e "${GREEN}*${RESET} Started (PID $PID)"

# ── Wait for Ready ───────────────────────────────────────────────────────────

echo -n "  Waiting"
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo ""
        echo -e "${GREEN}${BOLD}  * Server ready${RESET}"
        echo ""
        echo -e "  ${BOLD}Chat UI:${RESET}     http://127.0.0.1:${PORT}"
        echo -e "  ${BOLD}OpenAI API:${RESET}  http://127.0.0.1:${PORT}/v1"
        echo -e "  ${BOLD}OpenClaw:${RESET}    configure local-gemma4 provider -> http://127.0.0.1:${PORT}/v1"
        echo ""
        echo -e "  ${BOLD}Stop:${RESET}  kill $PID"
        echo -e "  ${BOLD}Log:${RESET}   tail -f $LOG_PATH"
        echo ""
        exit 0
    fi

    if ! kill -0 "$PID" 2>/dev/null; then
        echo ""
        echo ""
        echo -e "${RED}X Server crashed.${RESET}"
        echo ""
        tail -20 "$LOG_PATH" | sed 's/^/  /'
        echo ""
        echo -e "  Try: $0 --compact  or  $0 -c 32768"
        echo ""
        exit 1
    fi

    sleep 2
    echo -n "."
done

echo ""
echo ""
echo -e "${YELLOW}! Not ready after 3 minutes.${RESET}"
echo "  Log: tail -f $LOG_PATH"
echo "  Kill: kill $PID"
echo ""
exit 1
