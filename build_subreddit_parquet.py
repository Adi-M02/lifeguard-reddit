# build_subreddit_parquet.py
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import threading
import time
import traceback
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.types as patypes  # <-- needed for schema guard

import parse_zsts as pz  # read_lines_zst, process_json_object, infer_kind_and_month_from_name, list_zst_files

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it=None, **kwargs):
        return it if it is not None else range(0)


# --------------------- record iteration ---------------------

def _infer_meta(path: Path) -> Tuple[Optional[str], Optional[str]]:
    kind, month = pz.infer_kind_and_month_from_name(path)
    return kind, month  # kind in {"comments","submissions"} or None; month "YYYY-MM" or None


def iter_matching_records_for_file(
    path: Path,
    subreddit_lower: str,
    stats: Dict[str, int],
    progress_q=None,
    heartbeat_lines: int = 200_000,
) -> Iterator[Dict[str, Any]]:
    """
    Yield cleaned records for a single file and update stats.
    If progress_q is provided, emit periodic heartbeats with lines/matched counts.
    """
    kind, month = _infer_meta(path)
    last_lines = 0

    # tell main we're starting this file
    if progress_q is not None:
        progress_q.put({
            "event": "start",
            "file": path.name,
            "file_path": str(path),
            "pid": os.getpid(),
            "lines": 0,
            "matched": 0,
        })

    for line, _ in pz.read_lines_zst(path):
        if not line.strip():
            continue
        stats["lines_total"] += 1

        # heartbeat by lines scanned (even if no matches yet)
        if progress_q is not None and stats["lines_total"] - last_lines >= heartbeat_lines:
            progress_q.put({
                "event": "progress",
                "file": path.name,
                "file_path": str(path),
                "pid": os.getpid(),
                "lines": stats["lines_total"],
                "matched": stats["matched"],
            })
            last_lines = stats["lines_total"]

        try:
            obj = json.loads(line)
        except Exception:
            stats["bad_json"] += 1
            continue

        sr = obj.get("subreddit")
        if isinstance(sr, str) and sr.lower() == subreddit_lower:
            rec = pz.process_json_object(obj)  # drops 'edited', lower subreddit, cast times, html-unescape
            rec["_source_path"] = str(path)
            if kind:
                rec["_kind"] = kind
            if month:
                rec["_month"] = month
            stats["matched"] += 1
            yield rec

    # final heartbeat at EOF for this file's scan (counts only)
    if progress_q is not None:
        progress_q.put({
            "event": "progress",
            "file": path.name,
            "file_path": str(path),
            "pid": os.getpid(),
            "lines": stats["lines_total"],
            "matched": stats["matched"],
        })


def _iter_batches_for_file(
    path: Path,
    subreddit_lower: str,
    batch_size: int,
    stats: Dict[str, int],
    progress_q=None,
    heartbeat_lines: int = 200_000,
) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for rec in iter_matching_records_for_file(
        path, subreddit_lower, stats, progress_q=progress_q, heartbeat_lines=heartbeat_lines
    ):
        batch.append(rec)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# --------------------- empty-struct sanitization ---------------------

def _drop_empty_struct_columns_from_batch(
    batch: List[Dict[str, Any]],
    already_dropped: set[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Remove top-level keys that are {} (empty dict) across *all* records in the batch,
    plus any keys we've already decided to drop. Returns (new_batch, newly_dropped_keys).
    """
    if not batch:
        return batch, []

    # Prune previously dropped keys
    pruned: List[Dict[str, Any]] = []
    for rec in batch:
        if already_dropped:
            rec = {k: v for k, v in rec.items() if k not in already_dropped}
        pruned.append(rec)

    # Detect keys that are {} in every record (or missing), and never non-empty dict
    counts: Dict[str, int] = {}
    disqualify: set[str] = set()
    total = len(pruned)
    for rec in pruned:
        for k, v in rec.items():
            if isinstance(v, dict):
                if len(v) == 0:
                    counts[k] = counts.get(k, 0) + 1
                else:
                    disqualify.add(k)

    to_drop = [k for k, c in counts.items() if c == total and k not in disqualify and k not in already_dropped]
    if to_drop:
        pruned2: List[Dict[str, Any]] = []
        drop_set = set(to_drop).union(already_dropped)
        for rec in pruned:
            pruned2.append({k: v for k, v in rec.items() if k not in drop_set})
        return pruned2, to_drop

    return pruned, []


# --------------------- worker: write one shard per file ---------------------

def _write_shard_for_file(args: Tuple[str, str, int, str, str, int, Any]) -> Tuple[Optional[str], Dict[str, int]]:
    """
    Worker run in a separate process.
    Args: (path_str, subreddit_lower, batch_size, compression, tmp_dir, heartbeat_lines, progress_q)
    Returns: (shard_path or None, stats)
    """
    path_str, subreddit_lower, batch_size, compression, tmp_dir, heartbeat_lines, progress_q = args
    path = Path(path_str)
    stats = {"lines_total": 0, "bad_json": 0, "matched": 0}

    shard_path = Path(tmp_dir) / f"shard_{path.stem}.parquet"
    writer: Optional[pq.ParquetWriter] = None
    written_any = False

    # Track fields that caused empty-struct issues so we drop them consistently for this file
    dropped_empty_struct_fields: set[str] = set()

    try:
        for raw_batch in _iter_batches_for_file(
            path, subreddit_lower, batch_size, stats, progress_q=progress_q, heartbeat_lines=heartbeat_lines
        ):
            # 1) Remove any top-level {} columns from this batch (+ previously dropped)
            batch_list, newly_dropped = _drop_empty_struct_columns_from_batch(
                raw_batch, dropped_empty_struct_fields
            )
            if newly_dropped and progress_q is not None:
                progress_q.put({
                    "event": "drop_field_empty_struct",
                    "file": path.name,
                    "file_path": str(path),
                    "pid": os.getpid(),
                    "fields": newly_dropped,
                })
                dropped_empty_struct_fields.update(newly_dropped)

            # 2) Build table
            tbl = pa.Table.from_pylist(batch_list)

            # 3) Second guard: if Arrow still produced any struct<> (zero children), drop those columns
            empty_struct_cols: List[str] = []
            for field in tbl.schema:
                if patypes.is_struct(field.type) and len(field.type) == 0:
                    empty_struct_cols.append(field.name)
            if empty_struct_cols:
                # Drop from the table…
                tbl = tbl.drop(empty_struct_cols)
                # …and remember to strip them from future batches
                for col in empty_struct_cols:
                    if col not in dropped_empty_struct_fields:
                        dropped_empty_struct_fields.add(col)
                if progress_q is not None:
                    progress_q.put({
                        "event": "drop_field_empty_struct_schema_guard",
                        "file": path.name,
                        "file_path": str(path),
                        "pid": os.getpid(),
                        "fields": empty_struct_cols,
                    })

            # 4) Skip if the table ended up empty (no rows or no columns)
            if tbl.num_rows == 0 or tbl.num_columns == 0:
                continue

            # 5) Open writer lazily after sanitization so schema is Parquet-safe
            if writer is None:
                writer = pq.ParquetWriter(
                    where=str(shard_path),
                    schema=tbl.schema,
                    compression=compression,
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_table(tbl)
            written_any = True

        # tell main this file is done (normal completion)
        if progress_q is not None:
            progress_q.put({
                "event": "done",
                "file": path.name,
                "file_path": str(path),
                "pid": os.getpid(),
                "lines": stats["lines_total"],
                "matched": stats["matched"],
                "shard": str(shard_path) if written_any else None,
                "dropped_empty_struct_fields": sorted(dropped_empty_struct_fields),
            })

    except Exception as e:
        # Emit an explicit error event to the logger thread
        if progress_q is not None:
            progress_q.put({
                "event": "error",
                "file": path.name,
                "file_path": str(path),
                "pid": os.getpid(),
                "lines": stats.get("lines_total", 0),
                "matched": stats.get("matched", 0),
                "error": f"{e.__class__.__name__}: {e}",
                "traceback": traceback.format_exc(limit=20),
            })
        # best-effort cleanup of a partial shard
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        try:
            if shard_path.exists() and shard_path.stat().st_size == 0:
                shard_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None, stats

    finally:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            # closing errors are non-fatal; will be visible via 0-byte shard if any
            pass

    if not written_any:
        try:
            if shard_path.exists():
                shard_path.unlink()
        except Exception:
            pass
        return None, stats

    return str(shard_path), stats


# --------------------- coalesce shards → single parquet ---------------------

def _coalesce_shards_to_single(
    shards: List[str],
    out_parquet: Path,
    compression: str = "zstd",
    coalesce_batch_rows: int = 64_000,
) -> int:
    """
    Read all shard parquet files and write a single unified parquet.
    Uses the dataset API to unify schemas safely. Streams batches to avoid OOM.
    Returns total rows written.
    """
    dataset = ds.dataset(shards, format="parquet")
    unified_schema = dataset.schema

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0

    with pq.ParquetWriter(
        where=str(out_parquet),
        schema=unified_schema,
        compression=compression,
        use_dictionary=True,
        write_statistics=True,
    ) as writer:
        for batch in tqdm(
            dataset.to_batches(batch_size=coalesce_batch_rows),
            desc="Coalescing shards (batches)",
            unit="batch",
            dynamic_ncols=True,
        ):
            tbl = pa.Table.from_batches([batch], schema=unified_schema)
            total_rows += tbl.num_rows
            writer.write_table(tbl)

    return total_rows


# --------------------- logging helpers (main process only) ---------------------

def _iso_now() -> str:
    # local time with offset for easy reading
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())

def _log_line(log_path: Path, lock: threading.Lock, payload: Dict[str, Any]) -> None:
    payload = {"ts": _iso_now(), **payload}
    text = json.dumps(payload, ensure_ascii=False)
    with lock:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(text + "\n")

def _start_progress_monitor(expected_workers: int, log_path: Path, log_lock: threading.Lock):
    """
    Create a monitor with a thread that reads progress events,
    updates per-worker tqdm bars (one per file), and writes JSON lines to log.
    """
    mgr = Manager()
    q = mgr.Queue()

    # bars keyed by file name -> (tqdm, position)
    bars: Dict[str, Tuple[tqdm, int]] = {}
    free_positions = list(range(expected_workers))  # reuse bar slots

    stop_token = object()

    def run():
        _log_line(log_path, log_lock, {"event": "logger_start", "msg": f"monitor online; workers={expected_workers}"})
        while True:
            msg = q.get()
            if msg is stop_token:
                break
            if not isinstance(msg, dict):
                continue

            # Always log every event as JSON line
            _log_line(log_path, log_lock, msg)

            # Progress bars for non-error events
            file = msg.get("file", "unknown")
            event = msg.get("event", "progress")
            lines = int(msg.get("lines", 0))
            matched = int(msg.get("matched", 0))

            if event in {"start", "progress", "done"}:
                if file not in bars:
                    pos = free_positions.pop(0) if free_positions else 0
                    bar = tqdm(total=None, position=pos, desc=file, unit="lines", leave=False, dynamic_ncols=True)
                    bars[file] = (bar, pos)

                bar, pos = bars[file]
                bar.n = lines
                bar.set_postfix(matched=matched)
                bar.refresh()

                if event == "done":
                    bar.close()
                    free_positions.append(pos)
                    del bars[file]

        _log_line(log_path, log_lock, {"event": "logger_stop", "msg": "monitor offline"})

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return q, stop_token, t


# --------------------- orchestration ---------------------

def build_subreddit_parquet(
    input_folder: Path,
    out_parquet: Path,
    subreddit: str = "Lifeguards",
    recursive: bool = True,   # default recursive ON
    ignore: Optional[List[str]] = None,
    batch_size: int = 50_000,
    workers: int = 0,               # 0 => auto (one process per file up to CPU)
    compression: str = "zstd",
    tmp_dir: Optional[Path] = None,
    keep_shards: bool = False,
    progress_heartbeat_lines: int = 200_000,
    coalesce_batch_rows: int = 64_000,
    log_file: Optional[Path] = None,
) -> None:
    # Prepare log
    if log_file is None:
        log_file = out_parquet.parent / "build_subreddit_parquet.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # truncate on each run to keep logs clean; switch to append if you prefer history
    with open(log_file, "w", encoding="utf-8") as _:
        pass
    log_lock = threading.Lock()

    def LOG(payload: Dict[str, Any]) -> None:
        _log_line(log_file, log_lock, payload)

    ignore_set = set(ignore or [])
    files = [p for p in pz.list_zst_files(input_folder, recursive=recursive) if p.name not in ignore_set]
    if not files:
        LOG({"event": "config", "status": "no_input", "input_folder": str(input_folder), "recursive": recursive})
        raise SystemExit(f"No input files found in {input_folder} (recursive={recursive})")

    if workers <= 0:
        workers = min(len(files), os.cpu_count() or 1)

    if tmp_dir is None:
        tmp_dir = out_parquet.parent / f"{out_parquet.stem}_shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Run configuration log
    LOG({
        "event": "config",
        "status": "start",
        "input_folder": str(input_folder),
        "out_parquet": str(out_parquet),
        "subreddit": subreddit,
        "recursive": recursive,
        "ignore_count": len(ignore_set),
        "file_count": len(files),
        "workers": workers,
        "batch_size": batch_size,
        "compression": compression,
        "tmp_dir": str(tmp_dir),
        "keep_shards": keep_shards,
        "heartbeat_lines": progress_heartbeat_lines,
        "coalesce_batch_rows": coalesce_batch_rows,
        "log_file": str(log_file),
    })

    print(f"Discovered {len(files)} files; spawning {workers} workers; tmp_dir={tmp_dir}; recursive={recursive}")
    LOG({"event": "discover", "files": [str(p) for p in files[:50]], "note": "first_50_only"})

    # Start progress monitor (also writes to log)
    progress_q, stop_token, monitor_thread = _start_progress_monitor(
        expected_workers=workers, log_path=log_file, log_lock=log_lock
    )

    # 1) shard in parallel (one core per file)
    tasks = [
        (str(p), subreddit.lower(), batch_size, compression, str(tmp_dir), progress_heartbeat_lines, progress_q)
        for p in files
    ]
    shard_paths: List[str] = []
    agg = {"lines_total": 0, "bad_json": 0, "matched": 0}
    error_workers = 0

    try:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_write_shard_for_file, t) for t in tasks]
            fut_to_task = {f: t for f, t in zip(futs, tasks)}
            for fut in tqdm(cf.as_completed(futs), total=len(futs), desc="Sharding (files)", unit="file"):
                try:
                    shard, stats = fut.result()
                except Exception as e:
                    # Worker crashed in a way it couldn't report via queue
                    t = fut_to_task.get(fut)
                    LOG({
                        "event": "worker_crash",
                        "error": f"{e.__class__.__name__}: {e}",
                        "traceback": traceback.format_exc(limit=20),
                        "task": {
                            "path": t[0] if t else None,
                            "subreddit_lower": t[1] if t else None,
                            "batch_size": t[2] if t else None,
                            "compression": t[3] if t else None,
                            "tmp_dir": t[4] if t else None,
                        },
                    })
                    error_workers += 1
                    continue

                if shard:
                    shard_paths.append(shard)
                    LOG({"event": "shard_written", "shard": shard, "rows": stats.get("matched", 0)})
                else:
                    LOG({"event": "shard_empty_or_skipped", "task_path": fut_to_task[fut][0]})

                for k in agg:
                    agg[k] += stats.get(k, 0)

    finally:
        # stop the monitor thread
        progress_q.put(stop_token)
        monitor_thread.join(timeout=5)

    if not shard_paths:
        LOG({
            "event": "finish",
            "status": "no_matches",
            "matched": agg["matched"],
            "lines_total": agg["lines_total"],
            "bad_json": agg["bad_json"],
            "error_workers": error_workers,
        })
        raise SystemExit(f"No records found for subreddit='{subreddit}'. Nothing to write.")

    LOG({"event": "coalesce_start", "shard_count": len(shard_paths), "out": str(out_parquet)})
    rows_written = _coalesce_shards_to_single(
        shard_paths, out_parquet, compression=compression, coalesce_batch_rows=coalesce_batch_rows
    )
    LOG({"event": "coalesce_done", "rows_written": rows_written, "out_size_bytes": out_parquet.stat().st_size})

    # 3) optional cleanup
    if not keep_shards:
        removed = 0
        for sp in shard_paths:
            try:
                Path(sp).unlink(missing_ok=True)
                removed += 1
            except Exception as e:
                LOG({"event": "cleanup_warn", "shard": sp, "error": f"{e.__class__.__name__}: {e}"})
        try:
            tmp_dir.rmdir()
        except Exception:
            pass
        LOG({"event": "cleanup", "removed_shards": removed})

    # Summary
    def _fmt(n): return f"{n:,}"
    print("\nSummary")
    print("-------")
    print(f"Files scanned      : {_fmt(len(files))}")
    print(f"Lines read         : {_fmt(agg['lines_total'])}")
    print(f"Bad JSON lines     : {_fmt(agg['bad_json'])}")
    print(f"Matched rows       : {_fmt(agg['matched'])}")
    print(f"Rows written       : {_fmt(rows_written)}")
    print(f"Output             : {out_parquet}")
    print(f"Shards kept        : {keep_shards}")
    print(f"Shard count        : {len(shard_paths)}")

    LOG({
        "event": "finish",
        "status": "ok",
        "files_scanned": len(files),
        "lines_total": agg["lines_total"],
        "bad_json": agg["bad_json"],
        "matched": agg["matched"],
        "rows_written": rows_written,
        "output": str(out_parquet),
        "keep_shards": keep_shards,
        "shard_count": len(shard_paths),
        "error_workers": error_workers,
    })


# --------------------- CLI ---------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build ONE Parquet for a target subreddit from compressed Reddit JSONL via per-file shards."
    )
    p.add_argument(
        "--input_folder",
        type=Path,
        required=False,
        default=Path(r"C:\Users\amukundan\Documents\torrent_files"),
        help="Folder containing RC_*/RS_* files (.zst/.xz/.bz2/.gz/.json[l])",
    )
    p.add_argument("--out_parquet", type=Path, default=Path("lifeguards.parquet"), help="Output Parquet path")
    p.add_argument("--subreddit", default="Lifeguards", help="Subreddit name (case-insensitive)")
    # Default recursive True, but allow turning it off:
    try:
        # Python 3.9+: BooleanOptionalAction lets you pass --no-recursive
        from argparse import BooleanOptionalAction
        p.add_argument("--recursive", default=True, action=BooleanOptionalAction, help="Recurse into subdirectories")
    except Exception:
        p.add_argument("--recursive", action="store_true", default=True, help="Recurse into subdirectories")
    p.add_argument("--ignore", nargs="*", default=[], help="File names to ignore (exact match)")
    p.add_argument("--batch_size", type=int, default=50_000, help="Rows per row group / memory batch per shard")
    p.add_argument("--workers", type=int, default=0, help="Shard pass workers; 0=auto (one per file up to CPU)")
    p.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4"],
        help="Parquet compression codec for shards and final file",
    )
    p.add_argument("--tmp_dir", type=Path, default=None, help="Where to write shard files (defaults near output)")
    p.add_argument("--keep_shards", action="store_true", help="Keep shard Parquet files after coalescing")
    p.add_argument(
        "--progress_heartbeat_lines",
        type=int,
        default=200_000,
        help="Per-worker heartbeat frequency in scanned lines for progress bars",
    )
    p.add_argument(
        "--coalesce_batch_rows",
        type=int,
        default=64_000,
        help="Row batch size when reading shards during final coalesce",
    )
    p.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Path to JSON-lines log file (default: <out_parquet_dir>/build_subreddit_parquet.log)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_subreddit_parquet(
        input_folder=args.input_folder,
        out_parquet=args.out_parquet,
        subreddit=args.subreddit,
        recursive=args.recursive,
        ignore=args.ignore,
        batch_size=args.batch_size,
        workers=args.workers,
        compression=args.compression,
        tmp_dir=args.tmp_dir,
        keep_shards=args.keep_shards,
        progress_heartbeat_lines=args.progress_heartbeat_lines,
        coalesce_batch_rows=args.coalesce_batch_rows,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
