#!/usr/bin/env python3
"""Dump all GGUF metadata keys and compute KV cache size estimates.

Usage:
    python3 dump_gguf.py path/to/model.gguf
    python3 dump_gguf.py path/to/model.gguf --ctx 4096 131072 --ram 16
"""

import argparse
import struct
import sys
from pathlib import Path

# GGUF value type IDs — from gguf/constants.py in this repo
UINT8   = 0
INT8    = 1
UINT16  = 2
INT16   = 3
UINT32  = 4
INT32   = 5
FLOAT32 = 6
BOOL    = 7
STRING  = 8
ARRAY   = 9
UINT64  = 10
INT64   = 11
FLOAT64 = 12

ELEM_SIZES = {
    UINT8: 1, INT8: 1,
    UINT16: 2, INT16: 2,
    UINT32: 4, INT32: 4, FLOAT32: 4,
    UINT64: 8, INT64: 8, FLOAT64: 8,
    BOOL: 1,
}

VT_NAMES = {
    UINT8: "u8", INT8: "i8",
    UINT16: "u16", INT16: "i16",
    UINT32: "u32", INT32: "i32", FLOAT32: "f32",
    UINT64: "u64", INT64: "i64", FLOAT64: "f64",
    BOOL: "bool", STRING: "str", ARRAY: "arr",
}

MAX_STR = 200_000  # safety cap for string reads


def align(f):
    pos = f.tell()
    pad = (8 - pos % 8) % 8
    if pad:
        f.seek(pad, 1)


def read_scalar(f, vt):
    if vt == UINT32:
        return struct.unpack("<I", f.read(4))[0]
    if vt == INT32:
        return struct.unpack("<i", f.read(4))[0]
    if vt == FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    if vt == UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    if vt == INT64:
        return struct.unpack("<q", f.read(8))[0]
    if vt == FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    if vt == UINT16:
        return struct.unpack("<H", f.read(2))[0]
    if vt == INT16:
        return struct.unpack("<h", f.read(2))[0]
    if vt == UINT8:
        return f.read(1)[0]
    if vt == INT8:
        return struct.unpack("<b", f.read(1))[0]
    if vt == BOOL:
        return bool(f.read(1)[0])
    return None


def read_string(f):
    sl = struct.unpack("<Q", f.read(8))[0]
    if sl > MAX_STR:
        f.seek(sl, 1)
        return f"<string {sl} bytes>"
    return f.read(sl).decode("utf-8", errors="replace")


def read_value(f, vt):
    if vt == STRING:
        return read_string(f)
    if vt == ARRAY:
        at = struct.unpack("<I", f.read(4))[0]
        al = struct.unpack("<Q", f.read(8))[0]
        if at == STRING:
            result = []
            for _ in range(al):
                result.append(read_string(f))
            return result
        if at == BOOL:
            return [bool(b) for b in f.read(al)]
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
        if at == INT8:
            return list(struct.unpack(f"<{al}b", raw))
        if at == UINT16:
            return list(struct.unpack(f"<{al}H", raw))
        if at == INT16:
            return list(struct.unpack(f"<{al}h", raw))
        if at == UINT64:
            return list(struct.unpack(f"<{al}Q", raw))
        if at == INT64:
            return list(struct.unpack(f"<{al}q", raw))
        if at == FLOAT64:
            return list(struct.unpack(f"<{al}d", raw))
        return f"[array elem_type={at} len={al}]"
    return read_scalar(f, vt)


def skip_value(f, vt):
    if vt == STRING:
        sl = struct.unpack("<Q", f.read(8))[0]
        f.seek(sl, 1)
    elif vt == ARRAY:
        at = struct.unpack("<I", f.read(4))[0]
        al = struct.unpack("<Q", f.read(8))[0]
        if at == STRING:
            for _ in range(al):
                sl = struct.unpack("<Q", f.read(8))[0]
                f.seek(sl, 1)
        elif at == BOOL:
            f.seek(al, 1)
        else:
            f.seek(al * ELEM_SIZES.get(at, 4), 1)
    elif vt in ELEM_SIZES:
        f.read(ELEM_SIZES[vt])


def dump_all_metadata(path):
    meta = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"Error: not GGUF (magic={magic!r})", file=sys.stderr)
            sys.exit(1)
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        file_gb = path.stat().st_size / (1024**3)

        print(f"GGUF v{version} | {n_tensors} tensors | {n_kv} metadata | {file_gb:.2f} GB")
        print("=" * 80)

        for i in range(n_kv):
            raw = f.read(8)
            if len(raw) < 8:
                print(f"[EOF at entry {i}]", file=sys.stderr)
                break
            kl = struct.unpack("<Q", raw)[0]
            if kl > 1000:
                print(f"[Entry {i}: invalid key_len={kl}, aborting]", file=sys.stderr)
                break
            key = f.read(kl).decode("utf-8", errors="replace")
            vt = struct.unpack("<I", f.read(4))[0]
            val = read_value(f, vt)
            meta[key] = val
    return meta


def fmt_bytes(n):
    if abs(n) < 1024:
        return f"{n} B"
    if abs(n) < 1024**2:
        return f"{n/1024:.1f} KB"
    if abs(n) < 1024**3:
        return f"{n/1024**2:.1f} MB"
    return f"{n/1024**3:.2f} GB"


def fmt_val(v):
    if isinstance(v, str):
        return v[:120] + "..." if len(v) > 120 else v
    if isinstance(v, list):
        n = len(v)
        if n == 0:
            return "[]"
        if n <= 8:
            return "[" + ", ".join(fmt_item(x) for x in v) + "]"
        head = ", ".join(fmt_item(x) for x in v[:3])
        tail = ", ".join(fmt_item(x) for x in v[-2:])
        return f"[{head}, ...({n - 5} more)..., {tail}]"
    if isinstance(v, float):
        return f"{v:.4f}" if v != int(v) else str(int(v))
    return str(v)


def fmt_item(v):
    if isinstance(v, str):
        return f'"{v[:40]}"' if len(v) > 40 else f'"{v}"'
    if isinstance(v, float):
        return f"{v:.2f}" if v != int(v) else str(int(v))
    return str(v)


def g(meta, key, default=0):
    return meta.get(key, default)


def compute_kv(meta, ctx, bpe=2):
    n_layers = g(meta, "gemma4.block_count") or g(meta, "llama.block_count", 0)
    if not n_layers:
        return 0, "no block_count"

    swa_pat = g(meta, "gemma4.attention.sliding_window_pattern")
    sw = g(meta, "gemma4.attention.sliding_window") or g(meta, "llama.attention.sliding_window", 0)
    shared = g(meta, "gemma4.attention.shared_kv_layers", 0)
    kl_swa = g(meta, "gemma4.attention.key_length_swa", 0)
    vl_swa = g(meta, "gemma4.attention.value_length_swa", 0)
    ghd = g(meta, "gemma4.global_head_dim", 0)
    ngkv = g(meta, "gemma4.num_global_key_value_heads", 0)
    kl = g(meta, "gemma4.attention.key_length") or g(meta, "llama.attention.key_length", 0)
    vl = g(meta, "gemma4.attention.value_length") or g(meta, "llama.attention.value_length", 0)
    nkv = g(meta, "gemma4.attention.head_count_kv") or g(meta, "llama.attention.head_count_kv", 0)
    kv_arr = g(meta, "gemma4.attention.head_count_kv_arr", [])

    # Gemma 4 ISWA path
    if swa_pat and len(swa_pat) == n_layers and (kl_swa or kl):
        unique = n_layers - shared + 1 if shared else n_layers
        n_swa = sum(1 for s in swa_pat if s)
        n_global = unique - n_swa
        window = sw if sw else ctx

        if kl_swa and ghd and ngkv:
            sw_per_tok = kl_swa + vl_swa
            gl_per_tok = ghd * 2 * ngkv
            total = n_swa * window * sw_per_tok * bpe + n_global * ctx * gl_per_tok * bpe
            desc = f"ISWA: {n_swa}swa×{window}tok×{sw_per_tok}B + {n_global}gl×{ctx}tok×{gl_per_tok}B (shared={shared})"
            return total, desc

        per_tok = kl + vl
        total = n_swa * window * per_tok * bpe + n_global * ctx * per_tok * bpe
        desc = f"ISWA: {n_swa}swa×{window} + {n_global}gl×{ctx} (per_tok={per_tok})"
        return total, desc

    # Standard path
    if nkv and kl:
        per_layer = nkv * kl * 2
        total = n_layers * ctx * per_layer * bpe
        return total, f"standard: {n_layers}L×{nkv}kv×{kl}dim"

    return 0, "unknown"


def main():
    p = argparse.ArgumentParser(description="GGUF metadata dumper + KV cache estimator")
    p.add_argument("path", type=Path)
    p.add_argument("--ctx", nargs="+", type=int,
                   default=[1024, 4096, 8192, 16384, 32768, 65536, 131072])
    p.add_argument("--ram", type=float, default=16)
    p.add_argument("--filter", "-f", type=str, default=None)
    args = p.parse_args()

    meta = dump_all_metadata(args.path)
    model_gb = args.path.stat().st_size / (1024**3)
    ram_gb = args.ram
    overhead_gb = 1.5

    # Print all metadata
    filt = args.filter.lower() if args.filter else None
    for key, val in meta.items():
        if filt and filt not in key.lower():
            continue
        print(f"  {key:<55} {fmt_val(val)}")

    # KV estimates
    max_ctx = g(meta, "gemma4.context_length") or g(meta, "llama.context_length", 0)
    arch = g(meta, "general.architecture", "?")

    print()
    print("=" * 80)
    print(f"  KV Cache Estimates — {arch} ({model_gb:.1f} GB file)")
    print("=" * 80)
    print(f"  {'Context':>10}  {'Raw fp16':>10}  {'turbo4':>10}  {'turbo3':>10}  {'turbo2':>10}  Method")
    print(f"  {'':->10}  {'':->10}  {'3.8x':>10}  {'4.6x':>10}  {'6.4x':>10}  {'':->30}")

    for ctx in args.ctx:
        if max_ctx and ctx > max_ctx:
            print(f"  {ctx:>10,}  (exceeds model max {max_ctx:,})")
            continue
        raw, method = compute_kv(meta, ctx)
        print(f"  {ctx:>10,}  {fmt_bytes(raw):>10}  {fmt_bytes(raw/3.8):>10}  "
              f"{fmt_bytes(raw/4.6):>10}  {fmt_bytes(raw/6.4):>10}  {method[:30]}")

    # Memory budget
    print()
    print("=" * 80)
    print(f"  Memory Budget — {ram_gb:.0f} GB RAM, {model_gb:.1f} GB model, ~{overhead_gb:.1f} GB overhead")
    print("=" * 80)
    print(f"  {'Ctx':>10}  {'KV':>8}  {'KV Size':>8}  {'Total':>8}  {'Fit':>4}  {'Margin':>10}")
    print(f"  {'':->10}  {'':->8}  {'':->8}  {'':->8}  {'':->4}  {'':->10}")

    model_b = model_gb * 1024**3
    ram_b = ram_gb * 1024**3
    oh_b = overhead_gb * 1024**3

    for ctx in args.ctx:
        if max_ctx and ctx > max_ctx:
            continue
        raw, _ = compute_kv(meta, ctx)
        for label, div in [("turbo4", 3.8), ("turbo3", 4.6), ("turbo2", 6.4)]:
            kv = raw / div
            total = model_b + kv + oh_b
            fits = total <= ram_b
            margin = (ram_b - total) / 1024**3
            mark = "YES" if fits else "NO"
            ms = f"+{margin:.1f}GB" if margin >= 0 else f"{margin:.1f}GB"
            print(f"  {ctx:>10,}  {label:>8}  {fmt_bytes(kv):>8}  {fmt_bytes(total):>8}  {mark:>4}  {ms:>10}")

    # MoE note
    experts = g(meta, "gemma4.expert_count") or g(meta, "llama.expert_count", 0)
    if experts:
        print()
        print(f"  MoE note: {experts} total experts, "
              f"{g(meta, 'gemma4.expert_used_count', '?')} active/token.")
        print(f"  Model weights are mmap'd. Only ~3-4 GB resident (active experts).")
        print(f"  Actual headroom is ~{model_gb - 3.5:.1f} GB MORE than shown above.")


if __name__ == "__main__":
    main()