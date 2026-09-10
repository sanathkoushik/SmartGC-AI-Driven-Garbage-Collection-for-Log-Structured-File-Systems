"""Shared fixtures.

Trace *data* is never committed: the MSR distribution's DISCLAIMER.txt reserves
reproduction rights to Microsoft, so the repository ships no trace bytes. Unit
tests therefore run on small synthetic files written in the real formats, and the
tests that need genuine data are skipped unless it has been downloaded into
``data/raw/`` (see ``real_msr_trace``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def msr_csv(tmp_path: Path) -> Path:
    """A small file in the exact MSR Cambridge format.

    Timestamps are Windows FILETIME ticks (100 ns). The third record is a
    20 KB request, which must expand to five 4 KB blocks; the fifth is
    deliberately unaligned.
    """
    content = "\n".join([
        # ts(FILETIME),host,disk,type,offset_bytes,size_bytes,response
        "128166372010810581,mds,0,Write,2654453760,4096,54086",     # 1 block
        "128166372011594132,mds,0,Write,3154132992,4096,51785",     # 1 block
        "128166372011600556,mds,0,Write,3201662976,20480,45361",    # 5 blocks
        "128166372011606862,mds,0,Read,3154124800,4096,39055",      # 1 block, read
        "128166372017062367,mds,0,Write,57819138,8192,52301",       # unaligned -> 3 blocks
        "128166372018000000,mds,0,Write,2654453760,4096,1000",      # rewrite of block 1
    ]) + "\n"
    path = tmp_path / "mds_0.csv"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def fiu_txt(tmp_path: Path) -> Path:
    """A small file in the FIU blkparse-derived format (ts in ns, LBA in sectors)."""
    content = "\n".join([
        # ts_ns pid process lba(512B) size(512B blocks) op major minor md5
        "777599999908936 2552 kjournald 160806224 8 W 6 0 abc",     # 4096 B -> 1 block
        "777599999953380 2552 kjournald 73397240 8 W 6 0 def",
        "777600000000000 2552 kjournald 160806224 16 W 6 0 aaa",    # 8192 B -> 2 blocks
        "777600000500000 4253 nfsd 160806232 8 R 6 0 bbb",
        "777601000000000 2552 kjournald 160806224 8 W 6 0 ccc",     # rewrite
    ]) + "\n"
    path = tmp_path / "homes.txt"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def fiu_multidevice_txt(tmp_path: Path) -> Path:
    """FIU records from two different block devices in one file."""
    content = "\n".join([
        # LBAs are multiples of 8 sectors so each request is exactly one
        # aligned 4 KB block; this fixture is about device separation only.
        "1000000 1 p 96 8 W 6 0 a",
        "2000000 1 p 200 8 W 6 0 b",
        "3000000 1 p 96 8 W 8 1 c",     # different major:minor, same LBA
        "4000000 1 p 96 8 W 6 0 d",
    ]) + "\n"
    path = tmp_path / "mixed.txt"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def systor_csv(tmp_path: Path) -> Path:
    """A small file in the SYSTOR '17 format (fractional Unix seconds)."""
    content = "\n".join([
        "Timestamp,Response,IOType,LUN,Offset,Size",
        "1456135200.013118000,0.001,W,0,4096,4096",
        "1456135200.113118000,0.001,R,0,8192,4096",
        "1456135201.013118000,0.001,W,0,4096,8192",     # 2 blocks
        "1456135202.013118000,0.001,,0,4096,4096",      # blank IOType -> dropped
        "1456135203.013118000,0.001,W,0,4096,4096",     # rewrite
    ]) + "\n"
    path = tmp_path / "2016022207-LUN0.csv"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def real_msr_trace() -> Path:
    """A genuine MSR volume, if one has been downloaded. Skips otherwise."""
    candidates = sorted((REPO_ROOT / "data" / "raw" / "msr").glob("*.csv.gz"))
    if not candidates:
        pytest.skip("no real MSR trace in data/raw/msr (run scripts/download_datasets.py)")
    return candidates[0]
