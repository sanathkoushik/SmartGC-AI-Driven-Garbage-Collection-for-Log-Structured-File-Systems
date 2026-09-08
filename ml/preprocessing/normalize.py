"""Phase 2 - trace ingestion & normalization.

Parses either the project's own synthetic workload CSV *or* a real public
block-I/O trace and emits the normalized-trace contract
(``timestamp,lba,size,operation``; ``size`` in 4 KiB blocks) to
``data/processed/``.

Supported ``--source`` values
-----------------------------
``synthetic``
    The ``timestamp,lba,size,operation`` CSV produced by
    ``smartgc_sim --export-trace`` (``size`` may be given in bytes and is
    converted to blocks).

``msr``
    MSR Cambridge block-I/O traces (SNIA IOTTA repository).  Header-less CSV
    rows of ``Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime``
    where ``Timestamp`` is a Windows FILETIME (100 ns ticks since 1601),
    ``Offset``/``Size`` are bytes and ``Type`` is ``Read``/``Write``.
    See ``docs/architecture.md`` for the licence/attribution note.

``auto``
    Sniff the header and map columns by name (works for many FIU / blktrace
    exports that carry a header row).
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Optional

import numpy as np
import pandas as pd

from ml.common.config import load_config, get, repo_path
from ml.common.io_contracts import TRACE_COLUMNS

_OP_MAP = {
    "w": "W", "write": "W", "1": "W", "ws": "W",
    "r": "R", "read": "R", "0": "R", "rs": "R",
    "d": "D", "discard": "D", "trim": "D",
}


def _to_op(value) -> str:
    return _OP_MAP.get(str(value).strip().lower(), "W")


def _bytes_to_blocks(size_series: pd.Series, block_size: int) -> pd.Series:
    s = pd.to_numeric(size_series, errors="coerce").fillna(block_size)
    # Heuristic: values already <= 8 are treated as block counts; larger values
    # are byte counts and get rounded up to whole blocks.
    blocks = np.where(
        s <= 8,
        np.maximum(1, np.round(s)),
        np.maximum(1, np.ceil(s / float(block_size))),
    )
    return pd.Series(blocks.astype("int64"), index=size_series.index)


def _finalize(df: pd.DataFrame, dense_lba: bool, working_set: Optional[int],
              max_events: Optional[int]) -> pd.DataFrame:
    df = df[df["operation"].isin(["W", "R", "D"])].copy()
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    if working_set:
        # Fold the address space onto a fixed working set so the trace exercises
        # a comparable amount of over-provisioning to the synthetic workload.
        df["lba"] = df["lba"].astype("int64") % int(working_set)
    if dense_lba:
        codes, _ = pd.factorize(df["lba"], sort=True)
        df["lba"] = codes.astype("int64")

    if max_events:
        df = df.iloc[: int(max_events)].copy()

    # Re-base the timestamp to a monotonic non-decreasing integer sequence.
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    if ts.isna().any() or (ts.diff().dropna() < 0).any():
        df["timestamp"] = np.arange(len(df), dtype="int64")
    else:
        df["timestamp"] = (ts - ts.min()).round().astype("int64")

    df["size"] = df["size"].astype("int64").clip(lower=1)
    return df[TRACE_COLUMNS]


def normalize_synthetic(path: str, block_size: int, **kw) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    out = pd.DataFrame({
        "timestamp": pd.to_numeric(df.get("timestamp", pd.Series(range(len(df)))), errors="coerce"),
        "lba": pd.to_numeric(df["lba"], errors="coerce").astype("int64"),
        "size": _bytes_to_blocks(df.get("size", pd.Series([block_size] * len(df))), block_size),
        "operation": df.get("operation", pd.Series(["W"] * len(df))).map(_to_op),
    })
    return _finalize(out, **kw)


def normalize_msr(path: str, block_size: int, **kw) -> pd.DataFrame:
    cols = ["timestamp", "hostname", "disk", "type", "offset", "size", "response_time"]
    df = pd.read_csv(path, header=None, names=cols, engine="c", on_bad_lines="skip")
    out = pd.DataFrame({
        "timestamp": pd.to_numeric(df["timestamp"], errors="coerce"),
        "lba": (pd.to_numeric(df["offset"], errors="coerce").fillna(0) // block_size).astype("int64"),
        "size": _bytes_to_blocks(df["size"], block_size),
        "operation": df["type"].map(_to_op),
    })
    return _finalize(out, **kw)


# 512-byte-sector index space large enough to keep every ASU's addresses disjoint.
_SPC_ASU_STRIDE = 1 << 36


def normalize_spc(path: str, block_size: int, **kw) -> pd.DataFrame:
    """UMass / SPC storage traces (e.g. Financial1/Financial2 OLTP).

    Header-less CSV rows of ``ASU,LBA,Size,Opcode,Timestamp`` where ``LBA`` is a
    512-byte-sector index *within its ASU*, ``Size`` is a byte count, ``Opcode``
    is ``r``/``w`` and ``Timestamp`` is elapsed seconds (float).  Each ASU is
    given a disjoint region of the address space and the combined sector index is
    folded to the simulator's 4 KiB page granularity (8 sectors per page).
    """
    cols = ["asu", "lba", "size", "opcode", "timestamp"]
    df = pd.read_csv(path, header=None, names=cols, engine="c", on_bad_lines="skip")
    asu = pd.to_numeric(df["asu"], errors="coerce").fillna(0).astype("int64")
    sector = pd.to_numeric(df["lba"], errors="coerce").fillna(0).astype("int64")
    sectors_per_page = max(1, block_size // 512)
    out = pd.DataFrame({
        # scale seconds -> microseconds so sub-second event order survives rebasing
        "timestamp": pd.to_numeric(df["timestamp"], errors="coerce") * 1_000_000.0,
        "lba": ((asu * _SPC_ASU_STRIDE + sector) // sectors_per_page).astype("int64"),
        "size": _bytes_to_blocks(df["size"], block_size),
        "operation": df["opcode"].map(_to_op),
    })
    return _finalize(out, **kw)


def normalize_auto(path: str, block_size: int, **kw) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    def pick(*names):
        for n in names:
            if n in df.columns:
                return df[n]
        return None

    lba_col = pick("lba", "offset", "blockno", "block", "sector", "address")
    if lba_col is None:
        raise ValueError(f"auto: could not find an address column in {path} ({list(df.columns)})")
    off = pd.to_numeric(lba_col, errors="coerce").fillna(0)
    # If it looks like a byte offset, convert to a block index.
    lba = (off // block_size if off.max() > 1 << 24 else off).astype("int64")

    ts = pick("timestamp", "time", "ts")
    op = pick("operation", "type", "rw", "op")
    size = pick("size", "bytes", "nbytes", "length")

    out = pd.DataFrame({
        "timestamp": pd.to_numeric(ts, errors="coerce") if ts is not None else pd.Series(range(len(df))),
        "lba": lba,
        "size": _bytes_to_blocks(size if size is not None else pd.Series([block_size] * len(df)), block_size),
        "operation": (op.map(_to_op) if op is not None else pd.Series(["W"] * len(df))),
    })
    return _finalize(out, **kw)


_DISPATCH = {
    "synthetic": normalize_synthetic,
    "msr": normalize_msr,
    "spc": normalize_spc,
    "auto": normalize_auto,
}


def main() -> None:
    cfg = load_config()
    block_size = int(get(cfg, "simulator.block_size_bytes", 4096))
    default_ws = int(get(cfg, "synthetic_workload.lba_range_max", 500))

    ap = argparse.ArgumentParser(description="SmartGC Phase 2 trace normalizer")
    ap.add_argument("--source", choices=sorted(_DISPATCH), required=True)
    ap.add_argument("--input", required=True, help="raw trace file")
    ap.add_argument("--output", help="normalized CSV path (default: data/processed/<name>.csv)")
    ap.add_argument("--name", help="logical trace name (default: input stem)")
    ap.add_argument("--dense-lba", action="store_true",
                    help="remap addresses to a contiguous 0..N-1 space")
    ap.add_argument("--working-set", type=int, default=None,
                    help=f"fold addresses onto this many LBAs (0=off; synthetic default {default_ws})")
    ap.add_argument("--max-events", type=int, default=None,
                    help="truncate to the first N events after sorting")
    args = ap.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.output or repo_path("data", "processed", f"{name}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ws = args.working_set
    if ws is None and args.source != "synthetic":
        ws = 0  # keep real addresses unless the caller asks to fold them

    df = _DISPATCH[args.source](
        args.input, block_size,
        dense_lba=args.dense_lba,
        working_set=(ws or None),
        max_events=args.max_events,
    )
    df.to_csv(out_path, index=False)

    writes = int((df["operation"] == "W").sum())
    print(f"[normalize] source={args.source} name={name}")
    print(f"[normalize] {len(df):,} events ({writes:,} writes) -> {out_path}")
    print(f"[normalize] unique LBAs={df['lba'].nunique():,}  "
          f"lba range=[{df['lba'].min()}, {df['lba'].max()}]")


if __name__ == "__main__":
    main()
