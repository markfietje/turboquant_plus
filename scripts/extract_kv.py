#!/usr/bin/env python3
"""Extract GGUF metadata and compute KV cache size estimates.

Usage:
    python3 extract_kv.py path/to/model.gguf [--ctx 8192 32768 131072] [--ram 16]
"""

import argparse
import struct
import sys
from pathlib import Path

# GGUF value type IDs
UINT32 = 4
INT32 = 6
FLOAT32 = 7
STRING = 8
ARRAY = 9
UINT64 = 11
INT64 = 12
FLOAT64 = 13

ELEM_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, UINT32: 4, INT32: 4, FLOAT32: 4, UINT64: 8, INT64: 8, FLOAT64: 8}

# Keys we care about for KV cache calculation
WANTED_KEYS = [
    "general.architecture",
    "gemma4.block_count",
    "gemma4.context_length",
    "gemma4.embedding_length",
    "gemma4.attention.head_count",
    "gemma4.attention.head_count_kv",
    "gemma4.attention.key_length",
    "gemma4.attention.value_length",
    "gemma4.attention.sliding_window",
    "gemma4.attention.shared_kv_layers",
    "gemma4.expert_count",
    "gemma4.expert_used_count",
    "gemma4.expert_feed_forward_length",
    "gemma4.attention.key_length_swa",
    "gemma4.attention.value_length_swa",
    "gemma4.attention.sliding_window_pattern",
    "gemma4.global_head_dim",
    "gemma4.num_global_key_value_heads",
    "gemma4.attention.head_count_kv_arr",
    # Generic fallbacks (non-gemma4)
    "llama.block_count",
    "llama.context_length",
    "llama.embedding_length",
    "llama.attention.head_count",
    "llama.attention.head_count_kv",
    "llama.attention.key_length",
    "llama.attention.value_length",
    "llama.attention.sliding_window",
    "llama.expert_count",
    "llama.expert_used_count",
    "llama.feed_forward_length",
]


def skip_value(f, vt):
    """Skip over a GGUF value we don't care about."""
    if vt in (UINT32, INT32, FLOAT32):
        f.read(4)
    elif vt in (UINT64, INT64, FLOAT64):
        f.read(8)
    elif vt == STRING:
        sl = struct.unpack("<Q", f.read(8))[0]
        f.seek(sl, 1)
    elif vt == ARRAY:
        at = struct.unpack("<I", f.read(4))[0]
        al = struct.unpack("<Q", f.read(8))[0]
        esz = ELEM_SIZES.get(at, 4)
        f.seek(al * esz, 1)
    else:
        # Unknown type — best effort skip
        pass


def read_value(f, vt):
    """Read a GGUF value, returning a Python object."""
    if vt in (UINT32,):
        return struct.unpack("<I", f.read(4))[0]
    if vt in (INT32,):
        return struct.unpack("<i", f.read(4))[0]
    if vt in (FLOAT32,):
        return struct.unpack("<f", f.read(4))[0]
    if vt in (UINT64,):
        return struct.unpack("<Q", f.read(8))[0]
    if vt in (INT64,):
        return struct.unpack("<q", f.read(8))[0]
    if vt in (FLOAT64,):
        return struct.unpack("<d", f.read(8))[0]
    if vt == STRING:
        sl = struct.unpack("<Q", f.read(8))[0]
        return f.read(sl).decode("utf-8", errors="replace")
    if vt == ARRAY:
        at = struct.unpack("<I", f.read(4))[0]
        al = struct.unpack("<Q", f.read(8))[0]
        esz = ELEM_SIZES.get(at, 4)
        raw = f.read(al * esz)
        if at == UINT32:
            return list(struct.unpack(f"<{al}I", raw))
        if at == INT32:
            return list(struct.unpack(f"<{al}i", raw))
        if at == FLOAT32:
            return list(struct.unpack(f"<{al}f", raw))
        if at == UINT8:
            return list(raw)
        return f"[array t={at} l={al}]"
    return None


def extract_meta(path):
    """Parse GGUF metadata, returning dict of key->value for wanted keys only."""
    meta = {}
    wanted = set(WANTED_KEYS)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"Error: not a GGUF file (magic={magic!r})", file=sys.stderr)
            sys.exit(1)
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        for _ in range(n_kv):
            raw = f.read(8)
            if len(raw) < 8:
                break
            kl = struct.unpack("<Q", raw)[0]
            key = f.read(kl).decode("utf-8", errors="replace")
            vt = struct.unpack("<I", f.read(4))[0]
            # Align to 8-byte boundary
            while f.tell() % 8:
                f.read(1)

            if key in wanted:
                val = read_value(f, vt)
                meta[key] = val
            else:
                skip_value(f, vt)
    return meta


def get(meta, key, default=0):
    """Get a value from meta dict, trying both gemma4 and llama prefixes."""
    if key in meta:
        return meta[key]
    # Try with swapped prefix
    alt = key.replace("gemma4.", "llama.").replace("llama.", "gemma4.")
    if alt in meta:
        return meta[alt]
    return default


def fmt(n):
    """Format bytes as human-readable."""
    if abs(n) < 1024:
        return f"{n} B"
    if abs(n) < 1024**2:
        return f"{n/1024:.1f} KB"
    if abs(n) < 1024**3:
        return f"{n/1024**2:.1f} MB"
    return f"{n/1024**3:.2f} GB"


def compute_kv_bytes(meta, ctx, bpe=2):
    """Compute raw KV cache size in bytes for a given context length.

    Handles Gemma 4 ISWA (interleaved sliding window attention):
    - Sliding window layers only store `sliding_window` tokens of KV
    - Global attention layers store full `ctx` tokens of KV
    - Shared KV layers reduce the number of unique KV buffers
    """
    n_layers = get(meta, "gemma4.block_count")
    if not n_layers:
        return 0, "no block_count"

    swa_pattern = get(meta, "gemma4.attention.sliding_window_pattern")
    sliding_window = get(meta, "gemma4.attention.sliding_window")
    shared_kv = get(meta, "gemma4.attention.shared_kv_layers")
    key_len_swa = get(meta, "gemma4.attention.key_length_swa")
    val_len_swa = get(meta, "gemma4.attention.value_length_swa")
    global_head_dim = get(meta, "gemma4.global_head_dim")
    n_global_kv = get(meta, "gemma4.num_global_key_value_heads")
    key_len = get(meta, "gemma4.attention.key_length")
    val_len = get(meta, "gemma4.attention.value_length")
    kv_heads_arr = get(meta, "gemma4.attention.head_count_kv_arr")

    # --- Gemma 4 ISWA path ---
    if swa_pattern and len(swa_pattern) == n_layers:
        # Count unique KV cache slots accounting for shared layers
        if shared_kv and shared_kv > 0:
            unique_slots = n_layers - shared_kv + 1
        else:
            unique_slots = n_layers

        n_swa = sum(1 for s in swa_pattern if s)
        n_global = unique_slots - n_swa
        sw = sliding_window if sliding_window and sliding_window > 0 else ctx

        if key_len_swa and val_len_swa:
            # Different head dims for sliding vs global
            sw_kv_per_tok = key_len_swa + val_len_swa  # K + V elements
            if global_head_dim and n_global_kv:
                gl_kv_per_tok = global_head_dim * 2 * n_global_kv  # K_eq_V: shared projection
            else:
                gl_kv_per_tok = key_len + val_len

            sw_bytes = n_swa * sw * sw_kv_per_tok * bpe
            gl_bytes = n_global * ctx * gl_kv_per_tok * bpe
            total = sw_bytes + gl_bytes
            desc = (
                f"ISWA: {n_swa}swa×{sw} + {n_global}global×{ctx} "
                f"(shared={shared_kv}, slots={unique_slots})"
            )
            return total, desc

        # Uniform key_length but different attention patterns
        kv_per_tok = key_len + val_len
        sw_bytes = n_swa * sw * kv_per_tok * bpe
        gl_bytes = n_global * ctx * kv_per_tok * bpe
        total = sw_bytes + gl_bytes
        desc = f"ISWA: {n_swa}swa×{sw} + {n_global}global×{ctx} (shared={shared_kv})"
        return total, desc

    # --- Per-layer KV heads path ---
    if kv_heads_arr and len(kv_heads_arr) == n_layers:
        total = 0
        for n_kv_h in kv_heads_arr:
            layer_kv = n_kv_h * key_len * 2  # K + V
            total += ctx * layer_kv * bpe
        return total, f"per-layer kv_heads ({set(kv_heads_arr)})"

    # --- Standard path ---
    n_kv_heads = get(meta, "gemma4.attention.head_count_kv", 0) or get(meta, "llama.attention.head_count_kv", 0)
    if n_kv_heads and key_len:
        kv_per_layer = n_kv_heads * key_len * 2  # K + V
        total = n_layers * ctx * kv_per_layer * bpe
        return total, f"standard: {n_layers}L × {n_kv_heads}kv × {key_len}dim"

    return 0, "unknown architecture"


def main():
    parser = argparse.ArgumentParser(description="GGUF KV cache size estimator")
    parser.add_argument("path", type=Path, help="Path to GGUF file")
    parser.add_argument("--ctx", nargs="+", type=int,
                        default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
                        help="Context lengths to estimate")
    parser.add_argument("--ram", type=float, default=16.0, help="Total RAM in GB (default: 16)")
    parser.add_argument("--model-gb", type=float, default=0, help="Model file size in GB")
    args = parser.parse_args()

    path = args.path
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    meta = extract_meta(path)
    model_gb = args.model_gb if args.model_gb else path.stat().st_size / (1024**3)
    model_bytes = model_gb * (1024**3)
    ram_bytes = args.ram * (1024**3)
    overhead_gb = 1.5
    overhead_bytes = overhead_gb * (1024**3)
    max_ctx = get(meta, "gemma4.context_length") or get(meta, "llama.context_length", 0)

    # Print metadata
    arch = get(meta, "general.architecture", "?")
    print("=" * 72)
    print(f"  {path.name}  ({model_gb:.1f} GB, arch={arch})")
    print("=" * 72)

    def show(key, label):
        v = get(meta, key)
        if v is None or v == 0:
            return
        if isinstance(v, list):
            if len(v) <= 35:
                print(f"  {label:<30} {v}")
            else:
                swa = sum(1 for x in v if x) if all(isinstance(x, (int, bool)) for x in v) else "?"
                print(f"  {label:<30} [{len(v)} items, swa={swa}]")
        else:
            print(f"  {label:<30} {v}")

    show("general.architecture", "Architecture")
    show("gemma4.context_length", "Max context")
    show("llama.context_length", "Max context (llama)")
    show("gemma4.block_count", "Layers")
    show("llama.block_count", "Layers (llama)")
    show("gemma4.embedding_length", "Embedding dim")
    show("gemma4.attention.head_count", "Attention heads")
    show("gemma4.attention.head_count_kv", "KV heads")
    show("gemma4.attention.key_length", "Key length")
    show("gemma4.attention.value_length", "Value length")
    show("gemma4.attention.sliding_window", "Sliding window")
    show("llama.attention.sliding_window", "Sliding window (llama)")
    show("gemma4.attention.shared_kv_layers", "Shared KV layers")
    show("gemma4.expert_count", "Expert count")
    show("gemma4.expert_used_count", "Experts/token")
    show("gemma4.attention.key_length_swa", "Key length (SWA)")
    show("gemma4.attention.value_length_swa", "Value length (SWA)")
    show("gemma4.global_head_dim", "Global head dim")
    show("gemma4.num_global_key_value_heads", "Global KV heads")
    show("gemma4.attention.sliding_window_pattern", "SWA pattern")
    show("gemma4.attention.head_count_kv_arr", "KV heads per layer")

    # KV cache estimates
    print()
    print("=" * 72)
    print("  KV Cache Size Estimates (fp16 baseline)")
    print("=" * 72)
    hdr = f"  {'Context':>10}  {'Raw fp16':>10}  {'turbo4':>10}  {'turbo3':>10}  {'turbo2':>10}  Method"
    div = f"  {'':->10}  {'':->10}  {'3.8x':>10}  {'4.6x':>10}  {'6.4x':>10}  {'':->24}"
    print(hdr)
    print(div)

    for ctx in args.ctx:
        if max_ctx and ctx > max_ctx:
            print(f"  {ctx:>10,}  (exceeds max {max_ctx:,})")
            continue
        raw, method = compute_kv_bytes(meta, ctx, bpe=2)
        t4 = raw / 3.8
        t3 = raw / 4.6
        t2 = raw / 6.4
        print(f"  {ctx:>10,}  {fmt(raw):>10}  {fmt(t4):>10}  {fmt(t3):>10}  {fmt(t2):>10}  {method[:24]}")

    # Memory budget
    print()
    print("=" * 72)
    print(f"  Memory Budget: {args.ram:.0f} GB RAM, model {model_gb:.1f} GB, ~{overhead_gb} GB overhead")
    print("=" * 72)
    print(f"  {'Context':>10}  {'KV Type':>8}  {'KV Size':>8}  {'Total':>8}  {'Fit?':>5}  {'Margin':>12}")
    print(f"  {'':->10}  {'':->8}  {'':->8}  {'':->8}  {'':->5}  {'':->12}")

    for ctx in args.ctx:
        if max_ctx and ctx > max_ctx:
            continue
        raw, _ = compute_kv_bytes(meta, ctx, bpe=2)
        for label, div in [("turbo4", 3.8), ("turbo3", 4.6), ("turbo2", 6.4)]:
            kv = raw / div
            total = model_bytes + kv + overhead_bytes
            fits = total <= ram_bytes
            margin = (ram_bytes - total) / (1024**3)
            margin_str = f"+{margin:.1f} GB" if margin >= 0 else f"{margin:.1f} GB"
            mark = "YES" if fits else "NO"
            print(f"  {ctx:>10,}  {label:>8}  {fmt(kv):>8}  {fmt(total):>8}  {mark:>5}  {margin_str:>12}")

    # Recommendation
    print()
    print("=" * 72)
    print("  Recommended KV Cache Configurations")
    print("=" * 72)
    print()
    print("  Max intelligence (short ctx):")
    print("    --cache-type-k q8_0 --cache-type-v turbo4")
    print("    Uncompressed K preserves quality; V compression is 'free'")
    print()
    print("  Balanced (medium ctx, TurboFlash):")
    print("    --cache-type-k turbo3 --cache-type-v turbo3")
    print("    New TurboFlash Metal kernel makes turbo3 fast on Apple Silicon")
    print()
    print("  Maximum context (long ctx):")
    print("    --cache-type-k turbo2 --cache-type-v turbo2")
    print("    6.4x compression, +6.48% PPL — use when context > quality")
    print()
    print("  Note: Model weights are mmap'd. Active memory for MoE is only")
    print("  the ~4B active params per token (~2-3 GB resident), so the")
    print("  model file size doesn't fully consume RAM. The above budget")
    print("  is conservative (assumes full model resident).")
    print()
    print("  For MoE models, actual headroom is ~{:.1f} GB more than shown.".format(
        model_gb - 3.0  # inactive expert pages can be paged out
    ))


if __name__ == "__main__":
    main()