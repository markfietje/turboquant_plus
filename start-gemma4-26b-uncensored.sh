#!/bin/bash
#
# Optimal Gemma 4 26B Uncensored + TurboQuant launch for M1 16 GB
#
# Model:  mradermacher/gemma-4-26B-A4B-it-uncensored-i1-GGUF (IQ3_S, 12.2 GB)
#          Biprojected + EGA abliterated — 0.7% refusal rate, KL divergence 0.09
# Method:  IQ importance-matrix quantization — significantly better than standard Q3 at same size
#
# KV Cache Strategy (asymmetric — based on turboquant research):
#   K: q8_0  — uncompressed keys, this is where ALL quality degradation comes from
#   V: turbo4 — 3.8× compression, "V compression is free" (near-zero PPL impact)
#
# This asymmetric config rescues quality for lower-bit weight quantizations (IQ3_S).
# If you need longer context at the cost of some quality, swap to symmetric turbo4.
#
# Memory Budget (M1 16 GB):
#   Model weights:     12.2 GB
#   KV cache (8K ctx): ~0.9 GB  (q8_0-K + turbo4-V)
#   Process overhead:   ~0.4 GB
#   macOS:             ~1.5 GB
#   ───────────────────────────
#   Total:             ~15.0 GB  ✓ fits with ~1 GB headroom
#
# Alternative KV configs (uncomment one section below):
#   SYMMETRIC_TURBO4  — turbo4-K + turbo4-V: 3.8× both, +0.23% PPL, more context headroom
#   SYMMETRIC_TURBO3  — turbo3-K + turbo3-V: 4.6× both, +1.06% PPL, TurboFlash decode kernel
#   SYMMETRIC_TURBO2  — turbo2-K + turbo2-V: 6.4× both, +6.48% PPL, max context length
#

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH="/Users/mark/LocalAI/models/gemma-4-26B-uncensored-IQ3_S.gguf"
BINARY_PATH="/Users/mark/LocalAI/turboquant_plus/build/bin/llama-server"
LOG_PATH="/tmp/llama-server-gemma4-26b.log"
PORT=8080

# ── Preflight ──────────────────────────────────────────────────────────────────
if [ ! -f "$MODEL_PATH" ]; then
    echo "✗ Model not found: $MODEL_PATH"
    echo ""
    echo "  Download with:"
    echo "  curl -L -o '$MODEL_PATH' \\"
    echo "    'https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-uncensored-i1-GGUF/resolve/main/gemma-4-26B-A4B-it-uncensored.i1-IQ3_S.gguf'"
    exit 1
fi

if [ ! -f "$BINARY_PATH" ]; then
    echo "✗ Binary not found: $BINARY_PATH"
    echo "  Build with: cd ~/LocalAI/turboquant_plus && cmake -B build -G Ninja -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j \$(sysctl -n hw.logicalcpu)"
    exit 1
fi

# ── Unlock VRAM ────────────────────────────────────────────────────────────────
# macOS reserves ~75% of unified memory for GPU by default.
# Override to give the model + KV cache more room.
WIRED_LIMIT=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "0")
if [ "$WIRED_LIMIT" -lt 13312 ]; then
    echo "⚠ Unlocking VRAM (requires sudo)..."
    sudo sysctl iogpu.wired_limit_mb=13312
    echo "  Set iogpu.wired_limit_mb=13312 (13 GB for GPU + model + KV)"
else
    echo "✓ VRAM already unlocked (iogpu.wired_limit_mb=$WIRED_LIMIT)"
fi

# ── Kill existing server ───────────────────────────────────────────────────────
EXISTING_PID=$(lsof -ti:"$PORT" 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "→ Stopping existing server on port $PORT (PID $EXISTING_PID)..."
    kill "$EXISTING_PID" 2>/dev/null || true
    sleep 1
fi

# ── KV Cache Configuration ────────────────────────────────────────────────────
# DEFAULT: Asymmetric — best intelligence for IQ3_S weights
CACHE_TYPE_K="q8_0"
CACHE_TYPE_V="turbo4"
CONTEXT_LEN=8192
KV_LABEL="asymmetric q8_0-K + turbo4-V (best quality)"

# ── Alternative: Symmetric turbo4 (more context, near-lossless) ───────────────
# Uncomment the next 4 lines to use symmetric turbo4 instead of the default:
# CACHE_TYPE_K="turbo4"
# CACHE_TYPE_V="turbo4"
# CONTEXT_LEN=16384
# KV_LABEL="symmetric turbo4 (3.8×, +0.23% PPL, more context)"

# ── Alternative: Symmetric turbo3 (TurboFlash decode, good balance) ────────────
# Uncomment the next 4 lines to use symmetric turbo3 instead of the default:
# CACHE_TYPE_K="turbo3"
# CACHE_TYPE_V="turbo3"
# CONTEXT_LEN=16384
# KV_LABEL="symmetric turbo3 (4.6×, TurboFlash decode kernel)"

# ── Alternative: Symmetric turbo2 (maximum context, lowest quality) ────────────
# Uncomment the next 4 lines to use symmetric turbo2 instead of the default:
# CACHE_TYPE_K="turbo2"
# CACHE_TYPE_V="turbo2"
# CONTEXT_LEN=32768
# KV_LABEL="symmetric turbo2 (6.4×, +6.48% PPL, max context)"

# ── Sampling Parameters (Gemma 4 recommended + tuning) ────────────────────────
TEMP=0.7
TOP_P=0.95
TOP_K=64
REPEAT_PENALTY=1.1

# ── Launch ─────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Gemma 4 26B Uncensored + TurboQuant                       ║"
echo "║  M1 16 GB · Metal · IQ3_S (12.2 GB)                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  KV Cache:  $KV_LABEL"
echo "║  Context:   ${CONTEXT_LEN} tokens"
echo "║  Temp:      ${TEMP}  top_p: ${TOP_P}  top_k: ${TOP_K}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

nohup "$BINARY_PATH" \
    -m "$MODEL_PATH" \
    -c "$CONTEXT_LEN" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --flash-attn on \
    --jinja \
    -t 4 \
    -tb 32 \
    -ngl 99 \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --repeat-penalty "$REPEAT_PENALTY" \
    --port "$PORT" \
    --host 127.0.0.1 \
    > "$LOG_PATH" 2>&1 &

PID=$!
echo "PID: $PID"
echo "Log: $LOG_PATH"
echo ""
echo "Waiting for server..."

for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo "✓ Server ready at http://127.0.0.1:${PORT}"
        echo ""
        echo "  Chat UI:     http://127.0.0.1:${PORT}"
        echo "  OpenAI API:  http://127.0.0.1:${PORT}/v1"
        echo "  Log:         tail -f $LOG_PATH"
        echo ""
        exit 0
    fi
    sleep 1
done

echo ""
echo "✗ Server failed to start within 60s"
echo ""
echo "  Check log:"
echo "    tail -50 $LOG_PATH"
echo ""
echo "  Common issues:"
echo "    - Out of memory → reduce -c or use symmetric turbo4/turbo3"
echo "    - Model load error → verify GGUF integrity (file size should be ~12.2 GB)"
echo "    - VRAM lock failed → re-run: sudo sysctl iogpu.wired_limit_mb=13312"
exit 1