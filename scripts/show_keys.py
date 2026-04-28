#!/usr/bin/env python3
"""Dump GGUF metadata keys from a model file.

Usage:
    python3 show_keys.py path/to/model.gguf [--filter block_count]
"""

import argparse
import struct
import sys
from pathlib import Path

# GGUF value type IDs
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

# Maximum string size to read into memory (avoid MemoryError on chat templates)
MAX_STRING_LEN = 1_000_000  # 1 MB
# Maximum key length to read into memory
MAX_KEY_LEN = 1_000  # 1 KB — keys should never be this long

ELEM_SIZES = {
    UINT8: 1, INT8: 1,
    UINT16: 2, INT16: 2,
    UINT32: 4, INT32: 4, FLOAT32: 4,
    UINT64: 8, INT64: 8, FLOAT64: 8,
    BOOL: 1,
}


def align8(f):
    """Advance file position to next 8-byte boundary."""
    pos = f.tell()
    pad = (8 - (pos % 8)) % 8
    if pad:
        f.seek(pad, 1)


def skip_value(f, vt):
    """Skip over a GGUF value we don't care about."""
    if vt in (UINT32, INT32, FLOAT32):
        f.read(4)
    elif vt in (UINT64, INT64, FLOAT64):
        f.read(8)
    elif vt in (UINT8, INT8):
        f.read(1)
    elif vt in (UINT16, INT16):
        f.read(2)
    elif vt == BOOL:
        f.read(1)
    elif vt == STRING:
        sl = struct.unpack("<Q", f.read(8))[0]
        f.seek(sl, 1)
    elif vt == ARRAY:
        at = struct.unpack("<I", f.read(4))[0]
        al = struct.unpack("<Q", f.read(8))[0]
        esz = ELEM_SIZES.get(at, 4)
        if at == STRING:
            for _ in range(al):
                sl = struct.unpack("<Q", f.read(8))[0]
                f.seek(sl, 1)
        elif at == BOOL:
            f.seek(al, 1)
        else:
            f.seek(al * esz, 1)
    # else: unknown type, can't skip safely


def read_value(f, vt):
    """Read a GGUF value, returning a Python object."""
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
    if vt == UINT8:
        return f.read(1)[0]
    if vt == INT8:
        return struct.unpack("<b", f.read(1))[0]
    if vt == UINT16:
        return struct.unpack("<H", f.read(2))[0]
    if vt == INT16:
        return struct.unpack("<h", f.read(2))[0]
    if vt == BOOL:
        return bool(f.read(1)[0])
    if vt == STRING:
        sl = struct.unpack("<Q", f.read(8))[0]
        if sl > MAX_STRING_LEN:
            f.seek(sl, 1)
            return f"<string {sl} bytes, skipped>"
        return f.read(sl).decode("utf-8", errors="replace")
    if vt == ARRAY:
        at = struct.unpack("<I", f.read(4))[0]
        al = struct.unpack("<Q", f.read(8))[0]
        esz = ELEM_SIZES.get(at, 4)
        if at == STRING:
            result = []
            for _ in range(al):
                sl = struct.unpack("<Q", f.read(8))[0]
                if sl > MAX_STRING_LEN:
                    f.seek(sl, 1)
                    result.append(f"<string {sl} bytes, skipped>")
                else:
                    result.append(f.read(sl).decode("utf-8", errors="replace"))
            return result
        if at == BOOL:
            return [bool(b) for b in f.read(al)]
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
        return f"[array type={at} len={al}]"
    return f"[unknown type={vt}]"


def vt_name(vt):
    """Human-readable value type name."""
    names = {
        UINT8: "u8", INT8: "i8",
        UINT16: "u16", INT16: "i16",
        UINT32: "u32", INT32: "i32", FLOAT32: "f32",
        UINT64: "u64", INT64: "i64", FLOAT64: "f64",
        STRING: "str", ARRAY: "arr", BOOL: "bool",
    }
    return names.get(vt, f"?{vt}")


def fmt_val(val, max_len=80):
    """Format a value for display."""
    if isinstance(val, str):
        if len(val) > max_len:
            return val[:max_len - 3] + "..."
        return val
    if isinstance(val, list):
        n = len(val)
        if n == 0:
            return "[]"
        if n <= 6:
            inner = ", ".join(fmt_item(v) for v in val)
            return f"[{inner}]"
        head = ", ".join(fmt_item(v) for v in val[:3])
        tail = ", ".join(fmt_item(v) for v in val[-2:])
        return f"[{head}, ...({n-5} more)..., {tail}]"
    if isinstance(val, float):
        if val == int(val):
            return str(int(val))
        return f"{val:.4f}"
    return str(val)


def fmt_item(v):
    """Format a single item in a list."""
    if isinstance(v, str):
        if len(v) > 30:
            return f'"{v[:27]}..."'
        return f'"{v}"'
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.2f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(description="Dump GGUF metadata keys")
    parser.add_argument("path", type=Path, help="Path to GGUF file")
    parser.add_argument("--filter", "-f", type=str, default=None,
                        help="Only show keys containing this substring")
    parser.add_argument("--values", "-v", action="store_true",
                        help="Show values (not just keys)")
    parser.add_argument("--types", "-t", action="store_true",
                        help="Show value types")
    args = parser.parse_args()

    path = args.path
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    filt = args.filter.lower() if args.filter else None

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"Error: not a GGUF file (magic={magic!r})", file=sys.stderr)
            sys.exit(1)
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        file_size = path.stat().st_size
        print(f"GGUF v{version}  |  {n_tensors} tensors  |  {n_kv} metadata entries  |  {file_size / (1024**3):.2f} GB")
        print("-" * 80)

        for i in range(n_kv):
            raw_kl = f.read(8)
            if len(raw_kl) < 8:
                print(f"\n[EOF at metadata entry {i} — file may be truncated]", file=sys.stderr)
                break
            kl = struct.unpack("<Q", raw_kl)[0]
            if kl > MAX_KEY_LEN:
                f.seek(kl, 1)
                vt = struct.unpack("<I", f.read(4))[0]
                align8(f)
                skip_value(f, vt)
                if not filt:
                    print(f"  <key {kl} bytes, skipped>")
                continue
            key = f.read(kl).decode("utf-8", errors="replace")
            vt = struct.unpack("<I", f.read(4))[0]

            align8(f)

            if filt and filt not in key.lower():
                skip_value(f, vt)
                continue

            val = read_value(f, vt)

            if args.values:
                type_str = f"({vt_name(vt)})" if args.types else ""
                val_str = fmt_val(val)
                print(f"  {key:<55} {type_str:<8} {val_str}")
            else:
                if args.types:
                    print(f"  {key:<55} ({vt_name(vt)})")
                else:
                    print(f"  {key}")


if __name__ == "__main__":
    main()