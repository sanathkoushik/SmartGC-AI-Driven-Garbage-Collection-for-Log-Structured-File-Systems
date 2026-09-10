"""Reader for the MSR Cambridge block-I/O trace format.

Original data: Microsoft Research Cambridge, February 2007. Cite Narayanan,
Donnelly and Rowstron, "Write Off-Loading: Practical Power Management for
Enterprise Storage", USENIX FAST '08.

Record layout, quoted from the README.txt shipped inside the distribution::

    Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime

    Timestamp is the time the I/O was issued in "Windows filetime"
    Hostname is the hostname (should be the same as that in the trace file name)
    DiskNumber is the disknumber (should be the same as in the trace file name)
    Type is "Read" or "Write"
    Offset is the starting offset of the I/O in bytes from the start of the
    logical disk.
    Size is the transfer size of the I/O request in bytes.
    ResponseTime is the time taken by the I/O to complete, in Windows filetime
    units.

Transformations applied here:

* ``Timestamp`` is a Windows FILETIME, i.e. 100-nanosecond ticks since
  1601-01-01 UTC. It is divided by 10 to give microseconds; the base epoch is
  irrelevant because ``base_parser`` rebases every trace to its own first
  record.
* ``Type`` "Write"/"Read" becomes ``W``/``R``.
* ``Offset`` and ``Size`` are already bytes, so block expansion in
  ``base_parser`` needs no unit conversion.
* ``Hostname`` and ``DiskNumber`` form the device key ``<host>_<disk>``. The
  distribution ships one volume per file, so the default ``device="auto"``
  policy simply verifies that; a file mixing volumes is rejected rather than
  silently pooled, which would fabricate rewrites between independent disks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from ml.preprocessing.base_parser import (
    FILETIME_TICKS_PER_MICROSECOND,
    BaseTraceParser,
    map_operation,
    open_maybe_compressed,
)

_TYPE_MAP = {"WRITE": "W", "READ": "R", "W": "W", "R": "R"}


class MsrTraceParser(BaseTraceParser):
    """Parses ``<hostname>_<disknumber>.csv[.gz]`` files."""

    dataset = "msr"
    device_description = "host/disk volume"
    RAW_COLUMNS = ["Timestamp", "Hostname", "DiskNumber", "Type", "Offset", "Size", "ResponseTime"]

    def _read_chunks(self, path: Path) -> Iterator[pd.DataFrame]:
        handle = open_maybe_compressed(path)
        try:
            reader = pd.read_csv(
                handle,
                header=None,
                names=self.RAW_COLUMNS,
                usecols=["Timestamp", "Hostname", "DiskNumber", "Type", "Offset", "Size"],
                dtype="string",
                chunksize=self.chunk_size,
                on_bad_lines="skip",
                engine="c",
            )
            for chunk in reader:
                # The distribution files carry no header row, but tolerating a
                # stray one costs nothing and avoids a spurious malformed record.
                if len(chunk) and str(chunk["Timestamp"].iloc[0]).strip().lower() == "timestamp":
                    chunk = chunk.iloc[1:]
                    if chunk.empty:
                        continue
                yield pd.DataFrame({
                    "raw_ts": pd.to_numeric(chunk["Timestamp"], errors="coerce")
                              // FILETIME_TICKS_PER_MICROSECOND,
                    "offset_bytes": pd.to_numeric(chunk["Offset"], errors="coerce"),
                    "size_bytes": pd.to_numeric(chunk["Size"], errors="coerce"),
                    "operation": map_operation(chunk["Type"], _TYPE_MAP),
                    "device": (chunk["Hostname"].astype("string").str.strip() + "_"
                               + chunk["DiskNumber"].astype("string").str.strip()),
                })
        finally:
            handle.close()
