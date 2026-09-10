"""Reader for the SYSTOR '17 enterprise VDI block-I/O trace format.

Original data: Fujitsu Laboratories Ltd., February-March 2016. Cite Lee,
Kumano, Matsuki, Endo, Fukumoto and Sugawara, "Understanding storage traffic
characteristics on enterprise virtual desktop infrastructure", SYSTOR '17.

Record layout, quoted from the README.txt shipped with the distribution::

    Timestamp,Response,IOType,LUN,Offset,Size

    - Timestamp is the time the I/O was issued.
      The timestamp is given as a Unix time (seconds since 1/1/1970) with a
      fractional part. Although the fractional part is nine digits, it is
      accurate only to the microsecond level; please ignore the nanosecond part.
    - Response is the time needed to complete the I/O.
    - IOType is "Read(R)", "Write(W)", or ""(blank).

Transformations applied here:

* ``Timestamp`` is a fractional Unix time in seconds; multiplied by 1e6 and
  truncated to microseconds, matching the stated microsecond accuracy.
* ``Offset`` and ``Size`` are bytes, as in MSR, so block expansion needs no unit
  conversion.
* ``IOType`` blank means the trace did not record the direction. Such records
  cannot be classified as reads or writes, so they are dropped and counted under
  ``dropped_unknown_operation`` rather than being guessed at.
* ``LUN`` is the device key. The distribution stores one file per hour *per*
  LUN, so a workload is assembled by passing several hourly files for the same
  LUN to ``parse()``; the default ``device="auto"`` policy then verifies that no
  two LUNs were mixed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from ml.preprocessing.base_parser import (
    BaseTraceParser,
    map_operation,
    open_maybe_compressed,
)

_TYPE_MAP = {"W": "W", "R": "R", "WRITE": "W", "READ": "R"}
MICROSECONDS_PER_SECOND = 1_000_000


class SystorTraceParser(BaseTraceParser):
    """Parses ``<YYYYMMDDHH>-LUN<n>.csv[.gz]`` files."""

    dataset = "systor"
    device_description = "LUN"
    RAW_COLUMNS = ["Timestamp", "Response", "IOType", "LUN", "Offset", "Size"]

    def _read_chunks(self, path: Path) -> Iterator[pd.DataFrame]:
        handle = open_maybe_compressed(path)
        try:
            reader = pd.read_csv(
                handle,
                header=0,
                names=self.RAW_COLUMNS,
                usecols=["Timestamp", "IOType", "LUN", "Offset", "Size"],
                dtype="string",
                chunksize=self.chunk_size,
                on_bad_lines="skip",
                engine="c",
            )
            for chunk in reader:
                seconds = pd.to_numeric(chunk["Timestamp"], errors="coerce")
                yield pd.DataFrame({
                    # Multiply before truncating so sub-second resolution
                    # survives; the source is only microsecond-accurate anyway.
                    "raw_ts": (seconds * MICROSECONDS_PER_SECOND).astype("Float64")
                              .round().astype("Int64"),
                    "offset_bytes": pd.to_numeric(chunk["Offset"], errors="coerce"),
                    "size_bytes": pd.to_numeric(chunk["Size"], errors="coerce"),
                    "operation": map_operation(chunk["IOType"], _TYPE_MAP),
                    "device": chunk["LUN"].astype("string").str.strip(),
                })
        finally:
            handle.close()
