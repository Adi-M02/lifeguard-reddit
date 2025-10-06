#!/usr/bin/env python3
"""
merge_folder_to_parquet.py — merge ALL Parquet files under one folder into ONE Parquet.

- Recursively finds *.parquet under a directory.
- Builds a union-of-columns schema with safe promotions.
- Streams per-file, per-row-group (no cross-file dataset casts).
- Nested values (struct/list/map) → JSON strings if the chosen output type is string.
- Dictionary-encoded columns are decoded before casting.
- Columns that are entirely null in a shard are preserved as typed nulls (no string→null casts).

Usage
  python merge_folder_to_parquet.py -d /path/to/folder -o merged.parquet --compression snappy
"""
import argparse
import glob
import json
import os
from collections import OrderedDict, defaultdict
from typing import Dict, List, Set

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.types as pat
import pyarrow.compute as pc


# ---------- file discovery ----------
def find_parquet_files(root_dir: str, pattern: str) -> List[str]:
    root_dir = os.path.abspath(root_dir)
    paths = glob.glob(os.path.join(root_dir, pattern), recursive=True)
    files = sorted({p for p in paths if p.lower().endswith(".parquet")})
    if not files:
        raise SystemExit(f"No parquet files found under {root_dir} with pattern '{pattern}'.")
    return files


# ---------- schema unification helpers ----------
def _wider_int_type(types: Set[pa.DataType]) -> pa.DataType:
    max_bits = 0
    unsigned = None
    for t in types:
        is_unsigned = pat.is_unsigned_integer(t)
        bits = t.bit_width
        if unsigned is None:
            unsigned = is_unsigned
        max_bits = max(max_bits, bits)
    return {True: {8: pa.uint8(), 16: pa.uint16(), 32: pa.uint32(), 64: pa.uint64()},
            False:{8: pa.int8(), 16: pa.int16(), 32: pa.int32(), 64: pa.int64()}}[bool(unsigned)][max_bits]


def _choose_common_type(field_name: str, types: Set[pa.DataType], conflict_fallback: str) -> pa.DataType:
    # Drop null-only types from consideration; if everything is null, use string
    types = {t for t in types if not pat.is_null(t)}
    if not types:
        return pa.large_string()

    if len(types) == 1:
        return next(iter(types))

    # Numeric promotions
    if all(pat.is_integer(t) for t in types):
        has_signed = any(pat.is_signed_integer(t) for t in types)
        has_unsigned = any(pat.is_unsigned_integer(t) for t in types)
        return pa.float64() if (has_signed and has_unsigned) else _wider_int_type(types)

    if any(pat.is_floating(t) for t in types) or any(pat.is_integer(t) for t in types):
        return pa.float64()

    # Timestamps: if mixed timezones/units → fallback (avoid tricky tz drops)
    if all(pat.is_timestamp(t) for t in types):
        tzs = {t.tz for t in types}
        if len(tzs) == 1:
            tz = next(iter(tzs))
            return pa.timestamp("ms", tz=tz)
        # mixed tz → fallback
        return pa.large_string() if conflict_fallback == "string" else pa.float64()

    # If any side is string, choose string
    if any(pat.is_string(t) or pat.is_large_string(t) for t in types):
        return pa.large_string()

    # All binary → large_binary
    if all(pat.is_binary(t) or pat.is_large_binary(t) for t in types):
        return pa.large_binary()

    # For nested/struct/list/map disagreements → fallback
    return pa.large_string() if conflict_fallback == "string" else pa.float64()


def build_union_schema(files: List[str], order: str, conflict_fallback: str) -> pa.Schema:
    first_seen: "OrderedDict[str, pa.DataType]" = OrderedDict()
    by_name_types: Dict[str, Set[pa.DataType]] = defaultdict(set)

    for f in files:
        sch = pq.read_schema(f)
        for fld in sch:
            by_name_types[fld.name].add(fld.type)
            if fld.name not in first_seen:
                first_seen[fld.name] = fld.type

    names = list(first_seen.keys()) if order == "first_seen" else sorted(by_name_types.keys())
    fields = [pa.field(n, _choose_common_type(n, by_name_types[n], conflict_fallback)) for n in names]
    return pa.schema(fields)


# ---------- casting helpers ----------
def _json_default(o):
    if isinstance(o, (bytes, bytearray, memoryview)):
        try:
            return bytes(o).decode("utf-8", "replace")
        except Exception:
            return str(o)
    return str(o)


def _nested_to_json_strings(arr: pa.Array) -> pa.Array:
    pylist = arr.to_pylist()
    out = []
    for x in pylist:
        if x is None:
            out.append(None)
        else:
            try:
                out.append(json.dumps(x, ensure_ascii=False, default=_json_default))
            except TypeError:
                out.append(str(x))
    return pa.array(out, type=pa.large_string())


def _safe_cast_array(arr: pa.Array, target_type: pa.DataType) -> pa.Array:
    # Normalize dictionary-encoded columns first
    if pat.is_dictionary(arr.type):
        try:
            arr = arr.dictionary_decode()
        except Exception:
            # last resort: cast via values
            try:
                arr = pc.cast(arr, pa.large_string()).dictionary_decode()
            except Exception:
                pass

    # If the source is null-typed, just return typed nulls (avoid string→null casts anywhere)
    if pat.is_null(arr.type):
        return pa.nulls(len(arr), type=target_type)

    if arr.type == target_type:
        return arr

    # Nested → string via JSON
    if (pat.is_struct(arr.type) or pat.is_list(arr.type) or pat.is_large_list(arr.type) or pat.is_map(arr.type)) and (
        pat.is_string(target_type) or pat.is_large_string(target_type)
    ):
        return _nested_to_json_strings(arr)

    # Straightforward cast attempt
    try:
        return pc.cast(arr, target_type)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        # If target is string, stringify as a fallback
        if pat.is_string(target_type) or pat.is_large_string(target_type):
            try:
                return pc.utf8_replace_codepoints(pc.cast(arr, pa.large_string()), [], [])
            except Exception:
                return _nested_to_json_strings(arr)
        # Otherwise, give up: return typed nulls (length preserved)
        return pa.nulls(len(arr), type=target_type)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="Directory with Parquet files (searched recursively).")
    ap.add_argument("-o", "--out", required=True, help="Output Parquet file.")
    ap.add_argument("--pattern", default="**/*.parquet", help="Glob under the directory. Default: **/*.parquet")
    ap.add_argument("--compression", default="snappy", help="snappy|zstd|gzip|none (default: snappy)")
    ap.add_argument("--batch-rows", type=int, default=1_000_000, help="Approx rows per written row group.")
    ap.add_argument("--order", choices=["first_seen", "name"], default="first_seen", help="Output column order.")
    ap.add_argument("--coerce-conflicts-to", choices=["string", "float64"], default="string",
                    help="Fallback type when files disagree (default: string).")
    args = ap.parse_args()

    files = find_parquet_files(args.dir, args.pattern)
    print(f"Found {len(files)} parquet files. Building unified schema...")
    target_schema = build_union_schema(files, args.order, args.coerce_conflicts_to)

    writer = pq.ParquetWriter(
        where=args.out,
        schema=target_schema,
        compression=None if args.compression.lower() == "none" else args.compression,
        use_dictionary=True,
        version="2.6",
    )

    total_rows = 0
    try:
        for path in files:
            pf = pq.ParquetFile(path)
            # Intersect columns to read from this file (others will be filled with nulls)
            file_cols = [n for n in target_schema.names if n in pf.schema_arrow.names]
            # If odd shard has zero overlapping cols (rare), skip — nothing to contribute
            if not file_cols:
                continue

            for batch in pf.iter_batches(columns=file_cols, batch_size=args.batch_rows, use_threads=True):
                arrays = []
                for fld in target_schema:
                    idx = batch.schema.get_field_index(fld.name)
                    if idx == -1:
                        arrays.append(pa.nulls(batch.num_rows, type=fld.type))
                    else:
                        col = batch.column(idx)  # already a single array
                        arrays.append(_safe_cast_array(col, fld.type))

                out_batch = pa.RecordBatch.from_arrays(arrays, schema=target_schema)
                writer.write_batch(out_batch)
                total_rows += out_batch.num_rows
    finally:
        writer.close()

    print(f"✅ Wrote {args.out} with {total_rows} rows and {len(target_schema.names)} columns from {len(files)} files.")


if __name__ == "__main__":
    main()
