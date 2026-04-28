#!/usr/bin/env python3
"""Extract GGUF model metadata for KV cache size calculation.

Reads a GGUF file and prints the architecture parameters needed to
estimate KV cache memory usage, then computes per-context-length
estimates for each TurboQuant level.

Usage:
    python3 extract_gguf_meta.py path/to/model.gguf [--ctx 8192 32768 131072]
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path


# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 6
GGUF_TYPE_FLOAT32 = 7
GGUF_TYPE_BOOL = 8
GGUF_TYPE_STRING = 9
GGUF_TYPE_ARRAY = 10
GGUF_TYPE_UINT64 = 11
GGUF_TYPE_INT64 = 12
GGUF_TYPE_FLOAT64 = 13

TYPE_SIZES = {
    GGUF_TYPE_UINT8: 1,
    GGUF_TYPE_INT8: 1,
    GGUF_TYPE_UINT16: 2,
    GGUF_TYPE_INT16: 2,
    GGUF_TYPE_UINT32: 4,
    GGUF_TYPE_INT32: 4,
    GGUF_TYPE_FLOAT32: 4,
    GGUF_TYPE_UINT64: 8,
    GGUF_TYPE_INT64: 8,
    GGUF_TYPE_FLOAT64: 8,
}


@dataclass
class GGUFMeta:
    """Extracted GGUF metadata relevant to KV cache sizing."""

    architecture: str = ""
    context_length: int = 0
    block_count: int = 0
    embedding_length: int = 0
    head_count: int = 0
    head_count_kv: int = 0
    key_length: int = 0
    value_length: int = 0
    sliding_window: int = 0
    shared_kv_layers: int = 0
    expert_count: int = 0
    expert_used_count: int = 0
    expert_feed_forward_length: int = 0
    key_length_swa: int = 0
    value_length_swa: int = 0
    sliding_window_pattern: list[bool] = field(default_factory=list)
    head_count_kv_per_layer: list[int] = field(default_factory=list)
    # Global attention (Gemma 4 specific)
    global_head_dim: int = 0
    num_global_key_value_heads: int = 0
    # Raw KV for anything else we find
    extra: dict[str, object] = field(default_factory=dict)


def _read_value(f, val_type: int) -> object:
    """Read a single GGUF value from the file handle."""
    if val_type == GGUF_TYPE_STRING:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8", errors="replace")
    if val_type == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        elem_size = TYPE_SIZES.get(elem_type, 4)
        raw = f.read(arr_len * elem_size)
        if elem_type in (GGUF_TYPE_UINT32,):
            fmt = f"<{arr_len}I"
            return list(struct.unpack(fmt, raw))
        if elem_type in (GGUF_TYPE_INT32,):
            fmt = f"<{arr_len}i"
            return list(struct.unpack(fmt, raw))
        if elem_type in (GGUF_TYPE_UINT8,):
            return list(raw)
        if elem_type in (GGUF_TYPE_FLOAT32,):
            fmt = f"<{arr_len}f"
            return list(struct.unpack(fmt, raw))
        if elem_type in (GGUF_TYPE_INT16,):
            fmt = f"<{arr_len}h"
            return list(struct.unpack(fmt, raw))
        if elem_type in (GGUF_TYPE_UINT16,):
            fmt = f"<{arr_len}H"
            return list(struct.unpack(fmt, raw))
        if elem_type in (GGUF_TYPE_INT64,):
            fmt = f"<{arr_len}q"
            return list(struct.unpack(fmt, raw))
        if elem_type in (GGUF_TYPE_UINT64,):
            fmt = f"<{arr_len}Q"
            return list(struct.unpack(fmt, raw))
        if elem_type in (GGUF_TYPE_FLOAT64,):
            fmt = f"<{arr_len}d"
            return list(struct.unpack(fmt, raw))
        if elem_type == GGUF_TYPE_STRING:
            result = []
            for _ in range(arr_len):
                sl = struct.unpack("<Q", f.read(8))[0]
                result.append(f.read(sl).decode("utf-8", errors="replace"))
            return result
        if elem_type == GGUF_TYPE_BOOL:
            return [bool(struct.unpack("<?", f.read(1))[0]) for _ in range(arr_len)]
        return f"[array type={elem_type} len={arr_len}]"
    if val_type == GGUF_TYPE_BOOL:
        return bool(struct.unpack("<?", f.read(1))[0])
    size = TYPE_SIZES.get(val_type, 4)
    raw = f.read(size)
    if val_type == GGUF_TYPE_UINT32:
        return struct.unpack("<I", raw)[0]
    if val_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", raw)[0]
    if val_type == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", raw)[0]
    if val_type == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", raw)[0]
    if val_type == GGUF_TYPE_INT64:
        return struct.unpack("<q", raw)[0]
    if val_type == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", raw)[0]
    if val_type == GGUF_TYPE_UINT16:
        return struct.unpack("<H", raw)[0]
    if val_type == GGUF_TYPE_INT16:
        return struct.unpack("<h", raw)[0]
    if val_type == GGUF_TYPE_UINT8:
        return raw[0]
    if val_type == GGUF_TYPE_INT8:
        return struct.unpack("<b", raw)[0]
    return raw


def extract_gguf_meta(path: Path) -> GGUFMeta:
    """Parse GGUF metadata from a model file."""
    meta = GGUFMeta()

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path} (magic={magic!r})")

        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        # Field mapping: GGUF key -> GGUFMeta attribute
        field_map = {
            "general.architecture": "architecture",
            f"{None}.context_length": "context_length",
            f"{None}.block_count": "block_count",
            f"{None}.embedding_length": "embedding_length",
            f"{None}.attention.head_count": "head_count",
            f"{None}.attention.head_count_kv": "head_count_kv",
            f"{None}.attention.key_length": "key_length",
            f"{None}.attention.value_length": "value_length",
            f"{None}.attention.sliding_window": "sliding_window",
            f"{None}.attention.shared_kv_layers": "shared_kv_layers",
            f"{None}.expert_count": "expert_count",
            f"{None}.expert_used_count": "expert_used_count",
            f"{None}.expert_feed_forward_length": "expert_feed_forward_length",
            f"{None}.attention.key_length_swa": "key_length_swa",
            f"{None}.attention.value_length_swa": "value_length_swa",
            f"{None}.attention.sliding_window_pattern": "sliding_window_pattern",
            f"{None}.attention.head_count_kv": "head_count_kv_per_layer",
            f"{None}.global_head_dim": "global_head_dim",
            f"{None}.num_global_key_value_heads": "num_global_key_value_heads",
        }

        arch_prefix = ""

        for _ in range(n_kv):
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8")
            val_type = struct.unpack("<I", f.read(4))[0]
            # Align to 8-byte boundary
            while f.tell() % 8 != 0:
                f.read(1)

            value = _read_value(f, val_type)

            # Capture architecture to prefix later keys
            if key == "general.architecture" and isinstance(value, str):
                arch_prefix = value
                meta.architecture = value
                continue

            # Try to map to a field
            mapped_attr = None
            for pattern, attr in field_map.items():
                prefix = arch_prefix if pattern and pattern.startswith(f"{None}.") else ""
                resolved = pattern.replace(f"{None}.", f"{prefix}.") if pattern else key
                if resolved == key:
                    mapped_attr = attr
                    break

            if mapped_attr and hasattr(meta, mapped_attr):
                current = getattr(meta, mapped_attr)
                # Don't overwrite arrays with scalars
                if isinstance(current, list) and not isinstance(value, list):
                    continue
                setattr(meta, mapped_attr, value)
            else:
                meta.extra[key] = value

    return meta


def compute_kv_cache_bytes(
    meta: GGUFMeta,
    context_length: int,
    bytes_per_element: int = 2,  # fp16
) -> tuple[int, str]:
    """Compute raw KV cache size in bytes.

    Returns (bytes, description).
    Handles ISWA (interleaved sliding window attention) for Gemma 4.
    """
    n_layers = meta.block_count
    n_ctx = context_length

    if not n_layers or not meta.key_length:
        # Fallback: standard calculation
        kv_per_layer = meta.head_count_kv * meta.key_length * 2  # K + V
        total = n_layers * n_ctx * kv_per_layer * bytes_per_element
        return total, "standard (all layers, uniform KV)"

    # Gemma 4 ISWA path
    has_swa_pattern = len(meta.sliding_window_pattern) == n_layers
    has_global_kv = meta.global_head_dim > 0 and meta.num_global_key_value_heads > 0
    has_swa_dims = meta.key_length_swa > 0 and meta.value_length_swa > 0
    has_per_layer_kv = len(meta.head_count_kv_per_layer) == n_layers

    if has_swa_pattern and (has_global_kv or has_swa_dims):
        # Count unique KV cache slots
        # ISWA: some layers share KV caches (shared_kv_layers at the tail)
        n_shared = meta.shared_kv_layers
        if n_shared > 0:
            unique_kv_slots = n_layers - n_shared + 1
        else:
            unique_kv_slots = n_layers

        n_swa = sum(1 for s in meta.sliding_window_pattern if s)
        n_global = unique_kv_slots - n_swa

        sw = meta.sliding_window if meta.sliding_window > 0 else n_ctx

        if has_swa_dims:
            # Different head dims for sliding vs global
            sw_kv_per_token = meta.key_length_swa * 2  # K + V (head_dim already includes n_heads implicitly in the key_length)
            gl_kv_per_token = meta.global_head_dim * 2 * meta.num_global_key_value_heads if has_global_kv else meta.key_length * 2

            sw_bytes = n_swa * sw * sw_kv_per_token * bytes_per_element
            gl_bytes = n_global * n_ctx * gl_kv_per_token * bytes_per_element
            total = sw_bytes + gl_bytes
            desc = (
                f"ISWA: {n_swa} swa layers x {sw} ctx x {sw_kv_per_token} elems + "
                f"{n_global} global layers x {n_ctx} ctx x {gl_kv_per_token} elems "
                f"(shared_kv={n_shared}, unique_slots={unique_kv_slots})"
            )
        else:
            # Uniform key_length but different attention patterns
            kv_per_token = meta.key_length * 2  # K + V
            sw_bytes = n_swa * sw * kv_per_token * bytes_per_element
            gl_bytes = n_global * n_ctx * kv_per_token * bytes_per_element
            total = sw_bytes + gl_bytes
            desc = (
                f"ISWA: {n_swa} swa layers x {sw} ctx + "
                f"{n_global} global layers x {n_ctx} ctx "
                f"(kv_per_token={kv_per_token}, shared_kv={n_shared})"
            )
        return total, desc

    if has_per_layer_kv:
        # Per-layer KV head count (some models vary heads per layer)
        total = 0
        parts = []
        for i, n_kv_h in enumerate(meta.head_count_kv_per_layer):
            layer_kv = n_kv_h * meta.key_length * 2
            layer_bytes = n_ctx * layer_kv * bytes_per_element
            total += layer_bytes
            if i < 5 or i >= n_layers - 2 or n_kv_h != meta.head_count_kv_per_layer[0]:
                parts.append(f"L{i}:{n_kv_h}h")
        if len(parts) < n_layers:
            parts.append("...")
        desc = f"per-layer KV heads: {', '.join(parts)}"
        return total, desc

    # Standard path
    kv_per_layer = meta.head_count_kv * meta.key_length * 2  # K + V
    total = n_layers * n_ctx * kv_per_layer * bytes_per_element
    return total, f"standard: {n_layers} layers x {n_ctx} ctx x {kv_per_layer} elems"


def fmt_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def fmt_bpwt(n: int, total_params: int) -> str:
    """Format bits per weight."""
    if total_params == 0:
        return "N/A"
    return f"{n * 8 / total_params:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract GGUF metadata for KV cache calculation"
    )
    parser.add_argument("gguf_path", type=Path, help="Path to GGUF model file")
    parser.add_argument(
        "--ctx",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384, 32768, 65536, 131072],
        help="Context lengths to calculate (default: 4K..131K)",
    )
    parser.add_argument(
        "--model-size-gb",
        type=float,
        default=0,
        help="Model file size in GB (for total memory budget)",
    )
    args = parser.parse_args()

    path = args.gguf_path
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    meta = extract_gguf_meta(path)

    # ── Print metadata ──────────────────────────────────────────────
    print("=" * 64)
    print(f"  GGUF Metadata: {path.name}")
    print("=" * 64)
    print(f"  Architecture:        {meta.architecture}")
    print(f"  Context length:      {meta.context_length:,}")
    print(f"  Block count:         {meta.block_count}")
    print(f"  Embedding length:    {meta.embedding_length:,}")
    print(f"  Attention heads:     {meta.head_count}")
    print(f"  KV heads:            {meta.head_count_kv}")
    print(f"  Key length:          {meta.key_length}")
    print(f"  Value length:        {meta.value_length}")
    print(f"  Sliding window:      {meta.sliding_window:,}" if meta.sliding_window else "  Sliding window:      N/A")
    print(f"  Shared KV layers:    {meta.shared_kv_layers}" if meta.shared_kv_layers else "  Shared KV layers:    N/A")
    if meta.expert_count:
        print(f"  Expert count:        {meta.expert_count}")
        print(f"  Experts used/tok:    {meta.expert_used_count}")
        print(f"  Expert FFN length:   {meta.expert_feed_forward_length:,}" if meta.expert_feed_forward_length else "")
    if meta.global_head_dim:
        print(f"  Global head dim:     {meta.global_head_dim}")
        print(f"  Global KV heads:     {meta.num_global_key_value_heads}")
    if meta.key_length_swa:
        print(f"  Key length (SWA):    {meta.key_length_swa}")
        print(f"  Value length (SWA):  {meta.value_length_swa}")
    if meta.sliding_window_pattern:
        swa_count = sum(1 for s in meta.sliding_window_pattern if s)
        print(f"  SWA pattern:         {swa_count} sliding / {len(meta.sliding_window_pattern) - swa_count} global")
    if meta.head_count_kv_per_layer:
        unique = set(meta.head_count_kv_per_layer)
        if len(unique) > 1:
            print(f"  KV heads per layer: {unique}")

    # ── KV cache estimates ─────────────────────────────────────────
    model_size_bytes = int(args.model_size_gb * 1024 ** 3) if args.model_size_gb else 0

    print("\n" + "=" * 64)
    print("  KV Cache Size Estimates (fp16 baseline)")
    print("=" * 64)
    print(f"  {'Context':>10}  {'Raw KV':>10}  {'turbo4':>10}  {'turbo3':>10}  {'turbo2':>10}  Method")
    print(f"  {'':->10}  {'':->10}  {'3.8x':>10}  {'4.6x':>10}  {'6.4x':>10}  {'':->30}")

    for ctx in args.ctx:
        if ctx > meta.context_length and meta.context_length > 0:
            print(f"  {ctx:>10,}  (exceeds model max {meta.context_length:,})")
            continue

        raw_bytes, method = compute_kv_cache_bytes(meta, ctx, bytes_per_element=2)
        t4 = raw_bytes / 3.8
        t3 = raw_bytes / 4.6
        t2 = raw_bytes / 6.4

        total_t4 = t4 + model_size_bytes if model_size_bytes else 0
        total_t3 = t3 + model_size_bytes if model_size_bytes else 0
        total_t2 = t2 + model_size_bytes if model_size_bytes else 0

        line = f"  {ctx:>10,}  {fmt_bytes(raw_bytes):>10}  {fmt_bytes(t4):>10}  {fmt_bytes(t3):>10}  {fmt_bytes(t2):>10}  {method[:30]}"
        print(line)

    # ── Memory budget (if model size provided) ─────────────────────
    if model_size_bytes and meta.block_count:
        print("\n" + "=" * 64)
        print(f"  Memory Budget (model={args.model_size_gb:.1f} GB + 1.5 GB overhead)")
        print("=" * 64)
        overhead = 1.5 * 1024 ** 3
        total_budget = 16 * 1024 ** 3  # 16 GB target
        for ctx in args.ctx:
            if ctx > meta.context_length and meta.context_length > 0:
                continue
            raw_bytes, _ = compute_kv_cache_bytes(meta, ctx, bytes_per_element=2)
            for label, divisor in [("turbo4", 3.8), ("turbo3", 4.6), ("turbo2", 6.4)]:
                kv = raw_bytes / divisor
                total = model_size_bytes + kv + overhead
                fits = "✓" if total <= total_budget else "✗"
                margin = (total_budget - total) / (1024 ** 3)
                if margin < 0:
                    margin_str = f"over by {-margin:.1f} GB"
                else:
                    margin_str = f"{margin:.1f} GB free"
                print(f"  {ctx:>10,} {label:>8}: {fmt_bytes(total):>8} total  {fits}  {margin_str}")

    # ── Asymmetric K/V note ────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Asymmetric K/V Recommendation")
    print("=" * 64)
    print("  For maximum intelligence: --cache-type-k q8_0 --cache-type-v turbo4")
    print("  For long context:         --cache-type-k turbo3 --cache-type-v turbo3  (TurboFlash)")
    print("  For max context:          --cache-type-k turbo2 --cache-type-v turbo2")
    print()
    print("  Research finding: 'V compression is free' — all quality degradation")
    print("  comes from K compression. Use q8_0-K + turbo4-V to rescue quality")
    print("  when running lower-bit weight quantizations (IQ3_S, Q3_K_M).")


if __name__ == "__main__":
    main()