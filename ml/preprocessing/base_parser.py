"""Shared machinery for the raw block-I/O trace parsers.

Every dataset gets its own reader (``msr_parser``, ``fiu_parser``,
``systor_parser``) that knows only how to turn its native format into a common
intermediate frame::

    raw_ts (microseconds) | offset_bytes | size_bytes | operation | device

This module then applies the rules that must be identical for every dataset:
validation, 4 KB block expansion, device-namespace safety, chronological
ordering, timestamp rebasing and the audit trail.

Device-namespace safety
-----------------------
Logical block addresses are only meaningful *within one volume*. Two different
disks both have a block 0, and pooling them would invent rewrites between
unrelated data and corrupt every downstream measurement. Each parser therefore
reports a device key per record, and a normalized trace always describes exactly
one device:

* ``device="auto"`` (default) - accept the file only if every record names the
  same device; otherwise raise, naming the devices found.
* ``device="dominant"`` - keep the busiest device and record the choice.
* ``device="<key>"`` - keep exactly that device.

The selected device is recorded in the trace statistics, so the provenance of a
normalized file is never ambiguous.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

# 100-nanosecond ticks per microsecond, for Windows FILETIME conversion.
FILETIME_TICKS_PER_MICROSECOND = 10
NANOSECONDS_PER_MICROSECOND = 1000
SECTOR_BYTES = 512

NORMALIZED_COLUMNS = ["timestamp", "lba", "size", "operation", "trace_id"]
VALID_OPERATIONS = ("W", "R", "D")


class MultipleDevicesError(ValueError):
    """Raised when one trace file mixes several independent block devices."""


@dataclass
class ParseStats:
    """Audit trail for one trace file. Every dropped record is accounted for."""

    trace_id: str = ""
    dataset: str = ""
    source_file: str = ""
    raw_records_read: int = 0
    records_accepted: int = 0
    blocks_emitted: int = 0
    dropped_malformed: int = 0
    dropped_bad_offset: int = 0
    dropped_bad_size: int = 0
    dropped_oversized_request: int = 0
    dropped_unknown_operation: int = 0
    dropped_other_device: int = 0
    unknown_operation_examples: list[str] = field(default_factory=list)
    out_of_order_records: int = 0
    duplicate_timestamps: int = 0
    unaligned_requests: int = 0
    multi_block_requests: int = 0
    write_records: int = 0
    read_records: int = 0
    discard_records: int = 0
    write_blocks: int = 0
    read_blocks: int = 0
    first_raw_timestamp: int = 0
    last_raw_timestamp: int = 0
    duration_microseconds: int = 0
    total_bytes: int = 0
    truncated_by_limit: bool = False
    device_selected: str = ""
    device_record_counts: dict[str, int] = field(default_factory=dict)
    source_files: list[str] = field(default_factory=list)

    @property
    def dropped_total(self) -> int:
        return (self.dropped_malformed + self.dropped_bad_offset + self.dropped_bad_size
                + self.dropped_oversized_request + self.dropped_unknown_operation
                + self.dropped_other_device)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dropped_total"] = self.dropped_total
        return data


def open_maybe_compressed(path: Path) -> io.TextIOBase:
    """Open a trace file, transparently handling .gz, .bz2 and .xz."""
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    if suffix == ".bz2":
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="utf-8", errors="replace")
    if suffix in (".xz", ".lzma"):
        return io.TextIOWrapper(lzma.open(path, "rb"), encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def map_operation(values: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Map a raw operation column to W/R/D, preserving unrecognised spellings.

    Anything the mapping misses keeps its original text (blank becomes "BLANK")
    so it is counted as an *unknown operation* and shows up in the audit trail,
    rather than being silently lumped in with malformed rows.
    """
    raw = values.astype("string").str.strip().str.upper().fillna("")
    mapped = raw.map(mapping)
    fallback = raw.replace("", "BLANK")
    return mapped.fillna(fallback)


def ragged_arange(counts: np.ndarray) -> np.ndarray:
    """Concatenate ``range(c)`` for each c in `counts`. Every count must be >= 1.

    Expands multi-block requests into consecutive block indices without a
    Python-level loop, which at full trace scale would dominate runtime.
    """
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    out = np.ones(total, dtype=np.int64)
    out[0] = 0
    if counts.size > 1:
        starts = np.cumsum(counts)[:-1]
        out[starts] = 1 - counts[:-1]
    return np.cumsum(out)


class BaseTraceParser:
    """Validation, block expansion and statistics shared by all trace formats."""

    dataset = "unknown"
    #: Human-readable description of what a device key means for this format.
    device_description = "device"

    def __init__(self,
                 block_size_bytes: int = 4096,
                 max_blocks_per_request: int = 2048,
                 max_offset_bytes: int = 1 << 50,
                 chunk_size: int = 500_000,
                 device: str = "auto") -> None:
        if block_size_bytes <= 0:
            raise ValueError("block_size_bytes must be > 0")
        self.block_size_bytes = block_size_bytes
        self.max_blocks_per_request = max_blocks_per_request
        self.max_offset_bytes = max_offset_bytes
        self.chunk_size = chunk_size
        self.device = device
        self.selected_device: str | None = None

    # -- provided by subclasses ---------------------------------------------
    def _read_chunks(self, path: Path) -> Iterator[pd.DataFrame]:
        """Yield frames with columns raw_ts, offset_bytes, size_bytes, operation, device."""
        raise NotImplementedError

    # -----------------------------------------------------------------------
    def parse(self,
              paths: Path | list[Path],
              trace_id: str,
              max_records: int | None = None) -> tuple[pd.DataFrame, ParseStats]:
        """Parse one or more raw files of the same workload into normalized records.

        Several files are accepted so that formats which shard a workload by
        time (SYSTOR stores one file per hour) can be reassembled into a single
        chronological trace. `max_records` caps the raw records consumed and
        always keeps a chronological prefix, never a sample.
        """
        if isinstance(paths, Path):
            paths = [paths]
        if not paths:
            raise ValueError("no input files given")

        stats = ParseStats(trace_id=trace_id, dataset=self.dataset,
                           source_file=str(paths[0]),
                           source_files=[str(p) for p in paths])

        if self.device == "dominant":
            self.selected_device = self._scan_dominant_device(paths, stats)
        elif self.device == "auto":
            self.selected_device = None          # verified while reading
        else:
            self.selected_device = self.device

        frames: list[pd.DataFrame] = []
        consumed = 0
        seen_devices: dict[str, int] = {}

        for path in paths:
            for chunk in self._read_chunks(path):
                if max_records is not None and consumed >= max_records:
                    stats.truncated_by_limit = True
                    break
                if max_records is not None and consumed + len(chunk) > max_records:
                    chunk = chunk.iloc[: max_records - consumed]
                    stats.truncated_by_limit = True
                consumed += len(chunk)
                stats.raw_records_read += len(chunk)

                chunk = self._apply_device_policy(chunk, stats, seen_devices)
                if chunk is None or chunk.empty:
                    continue
                expanded = self._expand_chunk(chunk, stats)
                if expanded is not None and len(expanded):
                    frames.append(expanded)
            if max_records is not None and consumed >= max_records:
                break

        stats.device_record_counts = dict(seen_devices)
        if self.selected_device:
            stats.device_selected = self.selected_device
        elif len(seen_devices) == 1:
            stats.device_selected = next(iter(seen_devices))

        if not frames:
            empty = pd.DataFrame({
                "timestamp": pd.Series(dtype="int64"),
                "lba": pd.Series(dtype="int64"),
                "size": pd.Series(dtype="int32"),
                "operation": pd.Series(dtype="object"),
                "trace_id": pd.Series(dtype="object"),
            })
            return empty, stats

        table = pd.concat(frames, ignore_index=True)

        # Chronological order is a hard requirement of the rewrite-interval
        # definition. Count anything out of order, then stable-sort so ties keep
        # their original file order.
        timestamps = table["timestamp"].to_numpy()
        if len(timestamps) > 1:
            diffs = np.diff(timestamps)
            stats.out_of_order_records = int((diffs < 0).sum())
            stats.duplicate_timestamps = int((diffs == 0).sum())
            if stats.out_of_order_records:
                table = table.sort_values("timestamp", kind="stable").reset_index(drop=True)

        first_ts = int(table["timestamp"].iloc[0])
        last_ts = int(table["timestamp"].iloc[-1])
        stats.first_raw_timestamp = first_ts
        stats.last_raw_timestamp = last_ts
        stats.duration_microseconds = last_ts - first_ts

        # Rebase so intervals are comparable across traces and datasets.
        table["timestamp"] = (table["timestamp"] - first_ts).astype("int64")
        table["trace_id"] = trace_id
        table = table[NORMALIZED_COLUMNS]

        operations = table["operation"].to_numpy()
        stats.write_blocks = int((operations == "W").sum())
        stats.read_blocks = int((operations == "R").sum())
        stats.blocks_emitted = len(table)
        return table, stats

    # -----------------------------------------------------------------------
    def _scan_dominant_device(self, paths: list[Path], stats: ParseStats) -> str | None:
        """First pass over the device column only, to find the busiest device."""
        counts: dict[str, int] = {}
        for path in paths:
            for chunk in self._read_chunks(path):
                if "device" not in chunk:
                    return None
                for key, count in chunk["device"].value_counts().items():
                    counts[str(key)] = counts.get(str(key), 0) + int(count)
        if not counts:
            return None
        return max(counts.items(), key=lambda item: item[1])[0]

    def _apply_device_policy(self, chunk: pd.DataFrame, stats: ParseStats,
                             seen: dict[str, int]) -> pd.DataFrame | None:
        if "device" not in chunk:
            return chunk
        devices = chunk["device"].astype("string")
        for key, count in devices.value_counts().items():
            seen[str(key)] = seen.get(str(key), 0) + int(count)

        if self.selected_device is None:
            # "auto": a single device is required; more than one is an error,
            # because silently merging them would fabricate rewrites.
            if len(seen) > 1:
                raise MultipleDevicesError(
                    f"{self.dataset} trace mixes {len(seen)} devices ({sorted(seen)[:8]}). "
                    f"A normalized trace must describe one {self.device_description}: pass "
                    "device='dominant' to keep the busiest, or an explicit device key."
                )
            return chunk

        keep = devices == self.selected_device
        dropped = int((~keep).sum())
        if dropped:
            stats.dropped_other_device += dropped
        return chunk[keep]

    def _expand_chunk(self, chunk: pd.DataFrame, stats: ParseStats) -> pd.DataFrame | None:
        """Validate one chunk and expand each request into per-block records."""
        n_in = len(chunk)
        if n_in == 0:
            return None

        raw_ts = pd.to_numeric(chunk["raw_ts"], errors="coerce")
        offset = pd.to_numeric(chunk["offset_bytes"], errors="coerce")
        size = pd.to_numeric(chunk["size_bytes"], errors="coerce")
        operation = chunk["operation"].astype("string")

        usable = raw_ts.notna() & offset.notna() & size.notna() & operation.notna()
        stats.dropped_malformed += int(n_in - usable.sum())
        raw_ts, offset, size, operation = (raw_ts[usable], offset[usable],
                                           size[usable], operation[usable])

        # Operations outside the contract are dropped, but their spellings are
        # recorded so an unexpected format is visible rather than quietly
        # halving the trace.
        known = operation.isin(VALID_OPERATIONS)
        n_unknown = int((~known).sum())
        if n_unknown:
            stats.dropped_unknown_operation += n_unknown
            for value in operation[~known].dropna().unique()[:5]:
                if (str(value) not in stats.unknown_operation_examples
                        and len(stats.unknown_operation_examples) < 5):
                    stats.unknown_operation_examples.append(str(value))
        raw_ts, offset, size, operation = (raw_ts[known], offset[known],
                                           size[known], operation[known])

        offset_ok = (offset >= 0) & (offset <= self.max_offset_bytes)
        stats.dropped_bad_offset += int((~offset_ok).sum())
        raw_ts, offset, size, operation = (raw_ts[offset_ok], offset[offset_ok],
                                           size[offset_ok], operation[offset_ok])

        size_ok = size > 0
        stats.dropped_bad_size += int((~size_ok).sum())
        raw_ts, offset, size, operation = (raw_ts[size_ok], offset[size_ok],
                                           size[size_ok], operation[size_ok])

        if len(raw_ts) == 0:
            return None

        ts_arr = raw_ts.to_numpy(dtype=np.int64)
        off_arr = offset.to_numpy(dtype=np.int64)
        size_arr = size.to_numpy(dtype=np.int64)
        op_arr = operation.to_numpy(dtype=object)

        stats.total_bytes += int(size_arr.sum())
        stats.write_records += int((op_arr == "W").sum())
        stats.read_records += int((op_arr == "R").sum())
        stats.discard_records += int((op_arr == "D").sum())
        stats.unaligned_requests += int((off_arr % self.block_size_bytes != 0).sum())

        # A request covers every block it touches, including partial first and
        # last blocks: an unaligned 8 KB request spans three 4 KB blocks. The
        # blocks are real accesses, so they are emitted rather than rounded away.
        first_block = off_arr // self.block_size_bytes
        last_block = (off_arr + size_arr - 1) // self.block_size_bytes
        counts = (last_block - first_block + 1).astype(np.int64)
        stats.multi_block_requests += int((counts > 1).sum())

        within_cap = counts <= self.max_blocks_per_request
        n_over = int((~within_cap).sum())
        if n_over:
            stats.dropped_oversized_request += n_over
            ts_arr, first_block, counts, op_arr = (ts_arr[within_cap], first_block[within_cap],
                                                   counts[within_cap], op_arr[within_cap])
        if counts.size == 0:
            return None

        stats.records_accepted += int(counts.size)

        lba = np.repeat(first_block, counts) + ragged_arange(counts)
        return pd.DataFrame({
            "timestamp": np.repeat(ts_arr, counts),
            "lba": lba,
            "size": np.ones(lba.size, dtype=np.int32),
            "operation": np.repeat(op_arr, counts),
        })
