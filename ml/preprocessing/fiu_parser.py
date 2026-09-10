"""Reader for the FIU block-I/O trace format (blkparse-derived).

Original data: Florida International University, School of Computing and
Information Sciences, 2008-2009. The IODedup set accompanies Koller and
Rangaswami, "I/O Deduplication: Utilizing Content Similarity to Improve I/O
Performance" (FAST '10); the SRCMap set accompanies Verma et al., "SRCMap:
Energy Proportional Storage Using Dynamic Consolidation" (FAST '10).

Record layout, quoted from the README shipped with SNIA IOTTA trace 391::

    [ts in ns] [pid] [process] [lba] [size in 512 Bytes blocks] [Write or Read]
    [major device number] [minor device number] [MD5 per 4096 Bytes]

Transformations applied here:

* ``ts`` is nanoseconds; divided by 1000 to give microseconds.
* ``lba`` counts 512-byte sectors, so the byte offset is ``lba * 512``.
* ``size`` counts 512-byte blocks, so the byte length is ``size * 512``.
  An 8-sector request is therefore 4096 bytes, i.e. exactly one 4 KB block.
* ``major:minor`` is the device key. Several FIU files interleave more than one
  block device, whose LBA spaces are independent, so the parser refuses to pool
  them under the default ``device="auto"`` policy. Use ``device="dominant"`` to
  keep the busiest device and record the choice.

Note on availability: FIU is not downloadable without submitting personal
details to the SNIA IOTTA download form, and the original FIU host is offline
(see docs/dataset_source_decision.md). This parser is verified against real FIU
sample records and is ready for files placed in ``data/raw/fiu/``, but SmartGC's
external-validation results use SYSTOR '17 instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from ml.preprocessing.base_parser import (
    NANOSECONDS_PER_MICROSECOND,
    SECTOR_BYTES,
    BaseTraceParser,
    map_operation,
    open_maybe_compressed,
)

_OP_MAP = {"W": "W", "R": "R", "WS": "W", "RS": "R", "WRITE": "W", "READ": "R", "D": "D"}


class FiuTraceParser(BaseTraceParser):
    """Parses whitespace-separated FIU blktrace records."""

    dataset = "fiu"
    device_description = "major:minor block device"
    RAW_COLUMNS = ["ts_ns", "pid", "process", "lba", "size_blocks", "op",
                   "major", "minor", "md5"]

    def _read_chunks(self, path: Path) -> Iterator[pd.DataFrame]:
        handle = open_maybe_compressed(path)
        try:
            reader = pd.read_csv(
                handle,
                header=None,
                names=self.RAW_COLUMNS,
                usecols=["ts_ns", "lba", "size_blocks", "op", "major", "minor"],
                sep=r"\s+",
                dtype="string",
                chunksize=self.chunk_size,
                on_bad_lines="skip",
                engine="c",
            )
            for chunk in reader:
                yield pd.DataFrame({
                    "raw_ts": pd.to_numeric(chunk["ts_ns"], errors="coerce")
                              // NANOSECONDS_PER_MICROSECOND,
                    "offset_bytes": pd.to_numeric(chunk["lba"], errors="coerce") * SECTOR_BYTES,
                    "size_bytes": pd.to_numeric(chunk["size_blocks"], errors="coerce") * SECTOR_BYTES,
                    "operation": map_operation(chunk["op"], _OP_MAP),
                    "device": (chunk["major"].astype("string").str.strip() + ":"
                               + chunk["minor"].astype("string").str.strip()),
                })
        finally:
            handle.close()
