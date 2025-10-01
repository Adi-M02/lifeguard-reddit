# parse_zsts.py
from __future__ import annotations

import bz2
import gzip
import html
import io
import json
import lzma
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Any

# Optional zstandard support
_ZSTD_AVAILABLE = True
try:
    import zstandard as zstd  # type: ignore
except Exception:
    _ZSTD_AVAILABLE = False


@dataclass(frozen=True)
class RedditPath:
    month_label: Optional[str]   # e.g., "2025-03" if parsed from filename, else None
    kind: Optional[str]          # "comments" | "submissions" | None
    path: Path                   # file path


# --------------------- Core readers ---------------------

def read_lines_zst(file_name: str | os.PathLike[str]) -> Iterator[Tuple[str, int]]:
    """
    Stream text lines from a .zst (or any file—we sniff compression).
    Yields (line_text, file_byte_offset_at_yield_time).
    """
    path = Path(file_name)
    with _open_text_stream(path) as fh:
        buffer = ""
        while True:
            chunk = fh.read(2**20)  # 1 MiB text chunks (post-decompression)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                # we can't easily get the exact byte offset after TextIO decoding;
                # return -1 to indicate "unknown" for generic readers
                yield line, -1
            buffer = lines[-1]


def process_json_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a Reddit record lightly:
      - drop 'edited'
      - cast 'created'/'created_utc' to int when possible
      - lowercase 'subreddit'
      - HTML-unescape 'title'/'selftext'/'body' if present
    """
    rec = dict(obj)
    rec.pop("edited", None)

    for k in ("created", "created_utc"):
        if k in rec:
            try:
                rec[k] = int(rec[k])
            except (ValueError, TypeError):
                pass

    if "subreddit" in rec and isinstance(rec["subreddit"], str):
        rec["subreddit"] = rec["subreddit"].lower()

    for fld in ("title", "selftext", "body"):
        v = rec.get(fld)
        if isinstance(v, str):
            rec[fld] = html.unescape(v)

    return rec


# --------------------- Helpers for file discovery/meta ---------------------

_FILENAME_RX = re.compile(r"^(RC|RS)_(\d{4})-(\d{2})", re.IGNORECASE)

def infer_kind_and_month_from_name(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Infer "comments"/"submissions" and YYYY-MM from file name like RC_2025-03.zst
    Returns (kind, month_label) or (None, None) if not matched.
    """
    m = _FILENAME_RX.match(path.name)
    if not m:
        return None, None
    kind = "comments" if m.group(1).upper() == "RC" else "submissions"
    month = f"{m.group(2)}-{m.group(3)}"  # YYYY-MM
    return kind, month


def list_zst_files(folder: str | os.PathLike[str], recursive: bool = False) -> List[Path]:
    """
    Return sorted list of candidate files (.zst, .xz, .bz2, .gz, or plain .json/.jsonl).
    """
    root = Path(folder)
    patterns = ["*.zst", "*.zstd", "*.xz", "*.bz2", "*.gz", "*.json", "*.jsonl"]
    files: List[Path] = []
    if recursive:
        for pat in patterns:
            files.extend(root.rglob(pat))
    else:
        for pat in patterns:
            files.extend(root.glob(pat))
    # unique + sorted by name for determinism
    return sorted({f.resolve() for f in files if f.is_file()}, key=lambda p: p.name)


# --------------------- Internal: compression & sniffing ---------------------

def _open_text_stream(path: Path) -> io.TextIOBase:
    ext = path.suffix.lower()
    if ext in {".zst", ".zstd"}:
        return _open_zstd_text(path)
    if ext == ".xz":
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    if ext == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if ext == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")

    kind = _sniff_magic(path)
    if kind == "zstd":
        return _open_zstd_text(path)
    if kind == "xz":
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    if kind == "bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if kind == "gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")

    return open(path, "rt", encoding="utf-8", errors="replace")


def _open_zstd_text(path: Path) -> io.TextIOBase:
    if not _ZSTD_AVAILABLE:
        raise RuntimeError(
            f"{path.name} is Zstandard compressed, but the 'zstandard' package "
            "is not installed. Try: pip install zstandard"
        )
    fh = open(path, "rb")
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    stream = dctx.stream_reader(fh)
    return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")


def _sniff_magic(path: Path) -> str:
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
    except Exception:
        return "plain"
    if head.startswith(b"\x28\xB5\x2F\xFD"):
        return "zstd"
    if head.startswith(b"\xFD7zXZ\x00"):
        return "xz"
    if head.startswith(b"BZh"):
        return "bz2"
    if head.startswith(b"\x1F\x8B"):
        return "gz"
    return "plain"
