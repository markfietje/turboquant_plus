#!/bin/bash
#
# Gemma 4 — Dual Model Launcher for M1 16 GB
#
# Usage:
#   ./start-gemma4-26b.sh              # 26B uncensored, 3072 ctx (deep reasoning)
#   ./start-gemma4-26b.sh --e4b        # E4B uncensored, 131072 ctx (long context)
#
# Model options:
#   --26b     26B IQ3_S uncensored — more intelligence, shorter context
#   --e4b     E4B IQ4_XS uncensored — less intelligence, huge context
#
# Context override:
#   -c 2048   Set custom context length
#
# Memory requirements (M1 16 GB, wired_limit_mb=14336):
#
#   26B at 3072 ctx:  ~13.3 GB GPU (fits with 1 GB headroom)
#   26B at 4096 ctx:  crashes (compute buffer exceeds free GPU memory)
#   E4B at 131072:    ~6.5 GB GPU (fits easily)
#
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────────

MODEL_26B="/Users/mark/LocalAI/models/gemma-4-26B-uncensored-IQ3_S.gguf"
MODEL_E4B="/Users/mark/LocalAI/models/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf"
BINARY="/Users/mark/LocalAI/turboquant_plus/build/bin/llama-server"
LOG_PATH="/tmp/llama-server.log"
PORT=8080

# ── Defaults ────────────────────────────────────────────────────────────────────

MODE="26b"
CUSTOM_CTX=""

# ── Parse Arguments ────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --26b)
            MODE="26b"
            shift
            ;;
        --e4b)
            MODE="e4b"
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
            head -35 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--26b|--e4b] [-c CTX] [-p PORT]"
            exit 1
            ;;
    esac
done

# ── Set Model-Specific Config ──────────────────────────────────────────────────

if [ "$MODE" = "e4b" ]; then
    MODEL_PATH="$MODEL_E4B"
    CONTEXT="${CUSTOM_CTX:-131072}"
    CACHE_TYPE_K="q8_0"
    CACHE_TYPE_V="turbo2"
    NGL="99"
    LABEL="E4B Uncensored · $(printf '%,d' "$CONTEXT") ctx"
    DESC="E4B IQ4_XS at 131K context — long context agentic work"
else
    MODEL_PATH="$MODEL_26B"
    CONTEXT="${CUSTOM_CTX:-3072}"
    CACHE_TYPE_K="turbo3"
    CACHE_TYPE_V="turbo3"
    NGL="auto"
    LABEL="26B Uncensored · $(printf '%,d' "$CONTEXT") ctx"
    DESC="26B IQ3_S at 3K context — deep reasoning, tool use"

    # Safety check for 26B — context > 3072 will crash on M1 16 GB
    if [ "$CONTEXT" -gt 3072 ] && [ "$CUSTOM_CTX" != "" ]; then
        echo ""
        echo "  WARNING: context $CONTEXT will likely crash on M1 16 GB."
        echo "  The 26B model's Metal compute buffer exceeds available GPU"
        echo "  memory above ~3072 context. Max tested working: 3072."
        echo "  For longer context, use: $0 --e4b"
        echo ""
        echo "  Press Ctrl+C to cancel, or Enter to try anyway..."
        read -r
    fi
fi

TEMP=0.7
TOP_P=0.95
TOP_K=64
REPEAT_PENALTY=1.1
THREADS=4
BATCH_SIZE=32

# ── Colors ──────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Preflight ──────────────────────────────────────────────────────────────────

echo ""
echo -e "${CYAN}${BOLD}  Gemma 4 + TurboQuant${RESET}"
echo -e "${DIM}  M1 16 GB · Metal · $LABEL${RESET}"
echo ""

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}✗ Model not found:${RESET} $MODEL_PATH"
    exit 1
fi
echo -e "${GREEN}✓${RESET} Model: $(du -sh "$MODEL_PATH" | cut -f1) $(basename "$MODEL_PATH")"

if [ ! -f "$BINARY" ]; then
    echo -e "${RED}✗ Binary not found:${RESET} $BINARY"
    echo "  Build: cd ~/LocalAI/turboquant_plus && cmake -B build -G Ninja"
    echo "         -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON"
    echo "         -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release"
    echo "         cmake --build build -j \$(sysctl -n hw.logicalcpu)"
    exit 1
fi
echo -e "${GREEN}✓${RESET} Binary: $BINARY"

# ── VRAM Unlock ────────────────────────────────────────────────────────────────

WIRED=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "0")
NEEDED=14336
if [ "$WIRED" -lt "$NEEDED" ]; then
    echo -e "${YELLOW}⚠${RESET} Unlocking VRAM to ${NEEDED} MB (requires sudo)..."
    sudo sysctl "iogpu.wired_limit_mb=$NEEDED" > /dev/null 2>&1
    echo -e "${GREEN}✓${RESET} iogpu.wired_limit_mb=$NEEDED"
else
    echo -e "${GREEN}✓${RESET} VRAM: iogpu.wired_limit_mb=$WIRED"
fi

# ── Kill Existing ─────────────────────────────────────────────────────────────

EXISTING=$(lsof -ti:"$PORT" 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo -e "${YELLOW}→${RESET} Stopping existing server on :$PORT"
    kill "$EXISTING" 2>/dev/null || true
    sleep 1
fi

# ── Clear Log ─────────────────────────────────────────────────────────────────

: > "$LOG_PATH"

# ── Launch ────────────────────────────────────────────────────────────────────

echo ""
echo -e "  ${BOLD}Mode:${RESET}        $DESC"
echo -e "  ${BOLD}Context:${RESET}     $(printf '%,d' "$CONTEXT") tokens"
echo -e "  ${BOLD}KV Cache:${RESET}   ${CACHE_TYPE_K}-K + ${CACHE_TYPE_V}-V"
echo -e "  ${BOLD}GPU layers:${RESET} $NGL"
echo -e "  ${BOLD}Sampling:${RESET}   temp=${TEMP} top_p=${TOP_P} top_k=${TOP_K}"
echo ""
echo -e "${DIM}  Loading model...${RESET}"

nohup "$BINARY" \
    -m "$MODEL_PATH" \
    -c "$CONTEXT" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --flash-attn on \
    --jinja \
    -ngl "$NGL" \
    -t "$THREADS" \
    -tb "$BATCH_SIZE" \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --repeat-penalty "$REPEAT_PENALTY" \
    --port "$PORT" \
    --host 127.0.0.1 \
    >> "$LOG_PATH" 2>&1 &

PID=$!
echo -e "${GREEN}✓${RESET} Started (PID $PID)"

# ── Wait for Ready ────────────────────────────────────────────────────────────

echo -n "  Waiting"
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo ""
        echo -e "${GREEN}${BOLD}  ✓ Server ready${RESET}"
        echo ""
        echo -e "  ${BOLD}Chat UI:${RESET}     http://127.0.0.1:${PORT}"
        echo -e "  ${BOLD}OpenAI API:${RESET}  http://127.0.0.1:${PORT}/v1"
        echo -e "  ${BOLD}Health:${RESET}      http://127.0.0.1:${PORT}/health"
        echo -e "  ${BOLD}Stop:${RESET}        kill $PID"
        echo -e "  ${BOLD}Log:${RESET}         tail -f $LOG_PATH"
        echo ""
        echo -e "${DIM}  Other modes:"
        echo -e "    $0 --26b          26B uncensored, 3072 ctx (deep reasoning)"
        echo -e "    $0 --e4b          E4B uncensored, 131K ctx (long agentic work)"
        echo -e "    $0 --e4b -c 8192  E4B at custom context${RESET}"
        echo ""
        exit 0
    fi

    if ! kill -0 "$PID" 2>/dev/null; then
        echo ""
        echo ""
        echo -e "${RED}✗ Server crashed.${RESET}"
        echo ""
        tail -15 "$LOG_PATH"
        echo ""
        echo -e "${DIM}  Try: $0 --e4b  (uses less GPU memory)${RESET}"
        echo ""
        exit 1
    fi

    sleep 2
    echo -n "."
done

echo ""
echo ""
echo -e "${YELLOW}⚠ Not ready after 3 minutes.${RESET}"
echo "  Log:   tail -f $LOG_PATH"
echo "  Check: curl http://127.0.0.1:${PORT}/health"
echo "  Kill:  kill $PID"
echo ""
exit 1