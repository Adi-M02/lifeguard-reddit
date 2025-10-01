#!/usr/bin/env python3
"""
parquet_shell.py — Interactive CLI to explore a Parquet dataset fully in memory.

New corpus-level metadata commands
- corpusmeta                High-level corpus stats (files, total size, mtime range, rows/cols, RAM used)
- files [n]                 Show top-n largest data files with sizes (default 10)
- extcounts                 Count files by extension (e.g., .parquet, .parq)
- rowgroups [n]             (Optional) Sum Parquet row-groups; list top-n files by row-groups (can be slow)

Other data exploration commands (unchanged / expanded)
- cols / dtypes / shape / info / count / head / tail
- minmax <col>              Min/Max for a column (respects current filter)
- value_counts <col> [n]    Top-n value counts
- samplecol <col> [n]       Random sample of values in a column
- samplecols [n]            Sample values from all columns
- samplerows [n]            Random sample of rows
- nulls [n]                 Columns with highest null % (default 20)
- setfilter/clearfilter     Pandas .query() filter for subsequent commands
- showfilter
- seed <int>                RNG seed for reproducible sampling
- exportcsv <path> [n]      Export the current (filtered) view to CSV

Start:
  python parquet_shell.py --path path/to/file_or_directory
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import sys
import textwrap
from collections import Counter
from datetime import datetime
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except Exception:
    ds = None
    pq = None


# ------------------------- IO helpers -------------------------

def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.2f} {u}"
        size /= 1024.0


def _list_parquet_files(path: str) -> List[str]:
    """
    Return a deterministic list of data files that make up the dataset.
    If `path` is a directory, prefer pyarrow.dataset to find files; fallback to os.walk.
    If `path` is a file, return [path].
    """
    if os.path.isdir(path):
        if ds is not None:
            try:
                dset = ds.dataset(path, format="parquet")
                files = sorted(dset.files)
                return files
            except Exception:
                pass
        # Fallback: walk for .parquet-like files
        files = []
        for root, _, fnames in os.walk(path):
            for fn in fnames:
                lf = fn.lower()
                if lf.endswith(".parquet") or lf.endswith(".parq"):
                    files.append(os.path.join(root, fn))
        files.sort()
        return files
    else:
        return [path]


def _file_stats(files: List[str]) -> Dict[str, object]:
    """
    Compute file-level aggregate stats: counts, sizes, modified time range, extension counts.
    """
    sizes = []
    mtimes = []
    ext_counter = Counter()
    for f in files:
        try:
            st = os.stat(f)
            sizes.append(st.st_size)
            mtimes.append(st.st_mtime)
        except FileNotFoundError:
            # Skip missing
            continue
        # extension
        _, ext = os.path.splitext(f)
        ext_counter[ext.lower()] += 1

    sizes_sorted = sorted(sizes)
    total = sum(sizes)
    file_count = len(sizes)
    size_min = sizes_sorted[0] if sizes_sorted else 0
    size_max = sizes_sorted[-1] if sizes_sorted else 0
    size_med = sizes_sorted[len(sizes_sorted)//2] if sizes_sorted else 0
    size_avg = (total / file_count) if file_count else 0
    mtime_min = min(mtimes) if mtimes else None
    mtime_max = max(mtimes) if mtimes else None

    return {
        "file_count": file_count,
        "total_bytes": total,
        "total_human": human_bytes(total),
        "size_min_bytes": size_min,
        "size_min_human": human_bytes(size_min) if file_count else "0 B",
        "size_median_bytes": size_med,
        "size_median_human": human_bytes(size_med) if file_count else "0 B",
        "size_avg_bytes": int(size_avg),
        "size_avg_human": human_bytes(int(size_avg)) if file_count else "0 B",
        "size_max_bytes": size_max,
        "size_max_human": human_bytes(size_max) if file_count else "0 B",
        "mtime_earliest": _fmt_ts(mtime_min) if mtime_min else None,
        "mtime_latest": _fmt_ts(mtime_max) if mtime_max else None,
        "ext_counts": dict(ext_counter),
    }


def _fmt_ts(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except Exception:
        return str(ts)


def load_parquet_in_memory(path: str) -> pd.DataFrame:
    """
    Load a Parquet file (or directory of Parquet files) into a pandas DataFrame.
    """
    if os.path.isdir(path):
        if ds is None:
            sys.stderr.write("[error] Reading a directory requires pyarrow.dataset.\n")
            sys.exit(1)
        try:
            dataset = ds.dataset(path, format="parquet")
            table = dataset.to_table()
            df = table.to_pandas()
        except Exception as e:
            sys.stderr.write(f"[error] Failed to read parquet directory: {e}\n")
            sys.exit(1)
    else:
        try:
            df = pd.read_parquet(path, engine="pyarrow")
        except Exception:
            # Fallback to fastparquet if available
            try:
                df = pd.read_parquet(path, engine="fastparquet")
            except Exception as e2:
                sys.stderr.write(f"[error] Failed to read parquet file: {e2}\n")
                sys.exit(1)

    return df


# ------------------------- REPL core -------------------------

HELP_TEXT = textwrap.dedent("""
Commands
  help                          Show this help
  cols                          List columns (name, dtype, non-null count)
  dtypes                        Show only dtypes
  shape                         Show (rows, cols)
  info                          Pandas-style info summary
  count                         Row count (respects current filter)
  head [n]                      Show first n rows (default 5)
  tail [n]                      Show last n rows (default 5)

  minmax <col>                  Min/Max for a column (respects filter)
  value_counts <col> [n]        Top n value counts (default 20)
  samplecol <col> [n]           Sample n non-null values from a column (default 5)
  samplecols [n]                For each column, sample up to n non-null values (default 1)
  samplerows [n]                Sample n rows (default 5)
  nulls [n]                     Columns with highest null %% (default 20)

  setfilter <expr>              Set a pandas .query() filter for subsequent commands
                                Example: setfilter score > 100 and subreddit == "noburp"
  clearfilter                   Clear the active filter
  showfilter                    Show current filter

  seed <int>                    Set RNG seed for reproducible sampling
  exportcsv <path> [n]          Export current (filtered) view to CSV; optional row limit n

  # Corpus-level
  corpusmeta                    Show corpus stats (files, total size, mtime range, rows/cols, memory)
  files [n]                     Show top-n largest files with sizes (default 10)
  extcounts                     File extension counts
  rowgroups [n]                 Sum Parquet row-groups; list top-n files by row-groups (may be slow)

  quit | exit                   Leave the shell
""")


class ParquetShell:
    def __init__(self, df: pd.DataFrame, path: str):
        self.df = df
        self.path = path
        self.filter_expr: Optional[str] = None
        self.rng = np.random.default_rng()

        # Corpus metadata
        self.files: List[str] = _list_parquet_files(path)
        self.file_stats: Dict[str, object] = _file_stats(self.files)
        self._rowgroup_cache: Optional[List[Tuple[str, int]]] = None  # (file, rowgroups)

    # ----------- view handling -----------
    def view(self) -> pd.DataFrame:
        if not self.filter_expr:
            return self.df
        try:
            return self.df.query(self.filter_expr, engine="python")
        except Exception as e:
            print(f"[error] filter failed: {e}")
            return self.df

    # ----------- commands -----------
    def cmd_help(self, *_: str) -> None:
        print(HELP_TEXT)

    def cmd_cols(self, *_: str) -> None:
        v = self.view()
        nn = v.notna().sum()
        width = max((len(c) for c in v.columns), default=0)
        print(f"{'column'.ljust(width)}  {'dtype':<12}  non_null  nulls")
        for c in v.columns:
            dtype = str(v[c].dtype)
            non_null = int(nn[c])
            nulls = int(len(v) - non_null)
            print(f"{c.ljust(width)}  {dtype:<12}  {non_null:>8}  {nulls}")

    def cmd_dtypes(self, *_: str) -> None:
        v = self.view()
        print(v.dtypes.to_string())

    def cmd_shape(self, *_: str) -> None:
        v = self.view()
        print(v.shape)

    def cmd_info(self, *_: str) -> None:
        v = self.view()
        buf = io.StringIO()
        v.info(buf=buf)
        print(buf.getvalue())

    def cmd_count(self, *_: str) -> None:
        v = self.view()
        print(len(v))

    def cmd_head(self, n: str = "5", *_: str) -> None:
        v = self.view()
        n_i = _int_or(n, 5)
        print(v.head(n_i).to_string())

    def cmd_tail(self, n: str = "5", *_: str) -> None:
        v = self.view()
        n_i = _int_or(n, 5)
        print(v.tail(n_i).to_string())

    def cmd_minmax(self, col: Optional[str] = None, *_: str) -> None:
        if not col:
            print("[error] usage: minmax <col>")
            return
        v = self.view()
        if col not in v.columns:
            print(f"[error] column not found: {col}")
            return
        s = v[col].dropna()
        if s.empty:
            print(f"[info] all values are null in column '{col}' (under current filter).")
            return
        print(json.dumps({"column": col, "min": _to_py(s.min()), "max": _to_py(s.max())}, ensure_ascii=False))

    def cmd_value_counts(self, col: Optional[str] = None, n: str = "20", *_: str) -> None:
        if not col:
            print("[error] usage: value_counts <col> [n]")
            return
        v = self.view()
        if col not in v.columns:
            print(f"[error] column not found: {col}")
            return
        k = _int_or(n, 20)
        counts = v[col].value_counts(dropna=False).head(k)
        for i, (idx, cnt) in enumerate(counts.items()):
            label = repr(idx) if pd.isna(idx) else str(idx)
            print(f"{i+1:>3}. {label} — {cnt}")

    def cmd_samplecol(self, col: Optional[str] = None, n: str = "5", *_: str) -> None:
        if not col:
            print("[error] usage: samplecol <col> [n]")
            return
        v = self.view()
        if col not in v.columns:
            print(f"[error] column not found: {col}")
            return
        k = _int_or(n, 5)
        s = v[col].dropna()
        if s.empty:
            print(f"[info] no non-null values in '{col}' under current filter.")
            return
        replace = k > len(s)
        idx = self.rng.choice(s.index, size=min(k, len(s)), replace=replace, shuffle=True)
        for i, val in enumerate(s.loc[idx].tolist(), start=1):
            print(f"{i:>3}. {_repr_val(val)}")

    def cmd_samplecols(self, n: str = "1", *_: str) -> None:
        k = _int_or(n, 1)
        v = self.view()
        width = max((len(c) for c in v.columns), default=0)
        for c in v.columns:
            s = v[c].dropna()
            if s.empty:
                print(f"{c.ljust(width)} : [no non-null values]")
                continue
            sample_n = min(k, len(s))
            idx = self.rng.choice(s.index, size=sample_n, replace=False, shuffle=True)
            vals = ", ".join(_repr_val(x) for x in s.loc[idx].tolist())
            print(f"{c.ljust(width)} : {vals}")

    def cmd_samplerows(self, n: str = "5", *_: str) -> None:
        v = self.view()
        if len(v) == 0:
            print("[info] no rows under current filter.")
            return
        k = max(1, min(_int_or(n, 5), len(v)))
        idx = self.rng.choice(v.index, size=k, replace=False, shuffle=True)
        print(v.loc[idx].to_string())

    def cmd_nulls(self, n: str = "20", *_: str) -> None:
        v = self.view()
        k = _int_or(n, 20)
        if v.shape[1] == 0:
            print("[info] no columns.")
            return
        frac = v.isna().mean().sort_values(ascending=False)
        for name, p in frac.head(k).items():
            print(f"{name}: {p:.2%}")

    def cmd_setfilter(self, *expr_parts: str) -> None:
        if not expr_parts:
            print('[error] usage: setfilter <pandas-query-expr>\nExample: setfilter score > 100 and subreddit == "noburp"')
            return
        expr = " ".join(expr_parts)
        try:
            _ = self.df.query(expr, engine="python")
        except Exception as e:
            print(f"[error] invalid filter: {e}")
            return
        self.filter_expr = expr
        print(f"[ok] filter set: {self.filter_expr}")

    def cmd_clearfilter(self, *_: str) -> None:
        self.filter_expr = None
        print("[ok] filter cleared")

    def cmd_showfilter(self, *_: str) -> None:
        print(self.filter_expr if self.filter_expr else "[none]")

    def cmd_seed(self, seed: Optional[str] = None, *_: str) -> None:
        if seed is None:
            print("[error] usage: seed <int>")
            return
        try:
            i = int(seed)
        except Exception:
            print("[error] seed must be an integer")
            return
        self.rng = np.random.default_rng(i)
        print(f"[ok] seed set to {i}")

    def cmd_exportcsv(self, path: Optional[str] = None, n: Optional[str] = None, *_: str) -> None:
        if not path:
            print("[error] usage: exportcsv <path> [n]")
            return
        v = self.view()
        if n is not None:
            try:
                k = int(n)
                v = v.head(k)
            except Exception:
                pass
        try:
            v.to_csv(path, index=False)
            print(f"[ok] wrote {len(v)} rows to {path}")
        except Exception as e:
            print(f"[error] failed to write: {e}")

    # ----- Corpus-level -----

    def cmd_corpusmeta(self, *_: str) -> None:
        mem = self.df.memory_usage(index=True, deep=True).sum()
        meta = {
            "path": os.path.abspath(self.path),
            "is_directory": os.path.isdir(self.path),
            "rows": int(len(self.df)),
            "cols": int(self.df.shape[1]),
            "in_memory_bytes": int(mem),
            "in_memory_human": human_bytes(int(mem)),
        }
        meta.update(self.file_stats)
        print(json.dumps(meta, indent=2))

    def cmd_files(self, n: str = "10", *_: str) -> None:
        k = _int_or(n, 10)
        # pick top-k largest
        sized = []
        for f in self.files:
            try:
                sz = os.path.getsize(f)
            except FileNotFoundError:
                continue
            sized.append((f, sz))
        sized.sort(key=lambda t: t[1], reverse=True)
        print(f"[files] showing top {min(k, len(sized))} of {len(sized)} (largest first)")
        base = os.path.abspath(self.path)
        basedir = base if os.path.isdir(base) else os.path.dirname(base)
        for i, (f, sz) in enumerate(sized[:k], start=1):
            rel = os.path.relpath(f, basedir)
            print(f"{i:>3}. {rel} — {human_bytes(sz)}")

    def cmd_extcounts(self, *_: str) -> None:
        for ext, cnt in sorted(self.file_stats.get("ext_counts", {}).items(), key=lambda x: (-x[1], x[0])):
            label = ext or "[no ext]"
            print(f"{label}: {cnt}")

    def cmd_rowgroups(self, n: str = "10", *_: str) -> None:
        if pq is None:
            print("[error] pyarrow.parquet not available.")
            return
        if self._rowgroup_cache is None:
            out = []
            for f in self.files:
                try:
                    pf = pq.ParquetFile(f)
                    out.append((f, pf.metadata.num_row_groups))
                except Exception:
                    # non-parquet or unreadable
                    continue
            self._rowgroup_cache = out
        k = _int_or(n, 10)
        total = sum(cnt for _, cnt in self._rowgroup_cache)
        print(json.dumps({"total_row_groups": int(total), "files_counted": len(self._rowgroup_cache)}, ensure_ascii=False))
        # show top-n files by rowgroups
        top = sorted(self._rowgroup_cache, key=lambda t: t[1], reverse=True)[:k]
        base = os.path.abspath(self.path)
        basedir = base if os.path.isdir(base) else os.path.dirname(base)
        for i, (f, rg) in enumerate(top, start=1):
            rel = os.path.relpath(f, basedir)
            print(f"{i:>3}. {rel} — {rg} row-groups")

    # ----------- dispatch -----------
    def dispatch(self, line: str) -> bool:
        """
        Return False to exit.
        """
        if not line.strip():
            return True
        try:
            parts = shlex.split(line)
        except ValueError as e:
            print(f"[error] {e}")
            return True
        cmd, *args = parts
        cmd = cmd.lower()

        if cmd in {"quit", "exit"}:
            return False

        # Data exploration
        if cmd == "help": self.cmd_help(*args); return True
        if cmd == "cols": self.cmd_cols(*args); return True
        if cmd == "dtypes": self.cmd_dtypes(*args); return True
        if cmd == "shape": self.cmd_shape(*args); return True
        if cmd == "info": self.cmd_info(*args); return True
        if cmd == "count": self.cmd_count(*args); return True
        if cmd == "head": self.cmd_head(*args); return True
        if cmd == "tail": self.cmd_tail(*args); return True

        if cmd == "minmax": self.cmd_minmax(*args); return True
        if cmd == "value_counts": self.cmd_value_counts(*args); return True
        if cmd == "samplecol": self.cmd_samplecol(*args); return True
        if cmd == "samplecols": self.cmd_samplecols(*args); return True
        if cmd == "samplerows": self.cmd_samplerows(*args); return True
        if cmd == "nulls": self.cmd_nulls(*args); return True

        if cmd == "setfilter": self.cmd_setfilter(*args); return True
        if cmd == "clearfilter": self.cmd_clearfilter(*args); return True
        if cmd == "showfilter": self.cmd_showfilter(*args); return True

        if cmd == "seed": self.cmd_seed(*args); return True
        if cmd == "exportcsv": self.cmd_exportcsv(*args); return True

        # Corpus-level
        if cmd == "corpusmeta": self.cmd_corpusmeta(*args); return True
        if cmd == "files": self.cmd_files(*args); return True
        if cmd == "extcounts": self.cmd_extcounts(*args); return True
        if cmd == "rowgroups": self.cmd_rowgroups(*args); return True

        print("[error] unknown command. Type 'help'.")
        return True


def _int_or(x: str, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _to_py(x):
    # Convert numpy/pandas scalar to plain Python for JSON printing.
    try:
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass
    return x


def _repr_val(v) -> str:
    if isinstance(v, float):
        if pd.isna(v):
            return "NaN"
        return f"{v:.6g}"
    if isinstance(v, (pd.Timestamp, pd.Timedelta)):
        return str(v)
    return str(v)


# ------------------------- main -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive Parquet shell (loads fully into memory).")
    ap.add_argument("--path", required=True, help="Path to a Parquet file or a directory of Parquet files.")
    args = ap.parse_args()

    df = load_parquet_in_memory(args.path)

    mem = df.memory_usage(index=True, deep=True).sum()
    files = _list_parquet_files(args.path)
    file_count = len(files)
    print(f"[ok] loaded into memory: rows={len(df):,}, cols={df.shape[1]:,}, approx_mem={human_bytes(mem)}")
    print(f"[ok] corpus detected at {os.path.abspath(args.path)} — files={file_count}")
    print("Type 'help' for commands. 'quit' to exit.")

    shell = ParquetShell(df, args.path)
    while True:
        try:
            line = input("pq> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        cont = shell.dispatch(line)
        if not cont:
            break


if __name__ == "__main__":
    main()
