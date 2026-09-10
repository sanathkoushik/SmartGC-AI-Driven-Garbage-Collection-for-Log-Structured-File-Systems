"""Trace parsing, unit conversion, block expansion and device-namespace safety."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from ml.preprocessing.base_parser import MultipleDevicesError, ragged_arange
from ml.preprocessing.parsers import get_parser


def test_ragged_arange_expands_each_count():
    assert ragged_arange(np.array([1, 3, 2])).tolist() == [0, 0, 1, 2, 0, 1]
    assert ragged_arange(np.array([1])).tolist() == [0]
    assert ragged_arange(np.array([], dtype=np.int64)).size == 0


def test_msr_parser_expands_and_converts(msr_csv: Path):
    parser = get_parser("msr")
    table, stats = parser.parse(msr_csv, trace_id="msr_mds_0")

    # 6 raw requests -> 1 + 1 + 5 + 1 + 3 + 1 = 12 block records.
    assert stats.raw_records_read == 6
    assert stats.records_accepted == 6
    assert len(table) == 12
    assert stats.dropped_total == 0
    assert stats.multi_block_requests == 2          # the 20 KB and the unaligned 8 KB
    assert stats.unaligned_requests == 1

    # FILETIME ticks -> microseconds, rebased so the first record is 0.
    assert table["timestamp"].iloc[0] == 0
    # 128166372011594132 - 128166372010810581 = 783551 ticks = 78355 us
    assert table["timestamp"].iloc[1] == 78355

    # Byte offsets -> 4 KB blocks: 2654453760 / 4096 = 648060.
    assert table["lba"].iloc[0] == 648060

    # The 20 KB request at 3201662976 covers blocks 781656..781660 inclusive.
    twenty_kb = table[table["timestamp"] == 78997]
    assert twenty_kb["lba"].tolist() == [781656, 781657, 781658, 781659, 781660]
    assert set(twenty_kb["size"]) == {1}

    # An unaligned 8192-byte request at 57819138 spans three blocks because it
    # starts part-way into one: 57819138 // 4096 = 14116, end 57827329 // 4096 = 14118.
    unaligned = table[table["lba"].between(14116, 14118)]
    assert sorted(unaligned["lba"].tolist()) == [14116, 14117, 14118]

    assert set(table["operation"]) == {"W", "R"}
    assert stats.write_records == 5 and stats.read_records == 1
    assert set(table["trace_id"]) == {"msr_mds_0"}
    assert stats.device_selected == "mds_0"


def test_msr_parser_reads_gzip(tmp_path: Path, msr_csv: Path):
    gz_path = tmp_path / "mds_0.csv.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write(msr_csv.read_text(encoding="utf-8"))

    plain, _ = get_parser("msr").parse(msr_csv, trace_id="t")
    compressed, _ = get_parser("msr").parse(gz_path, trace_id="t")
    assert plain.equals(compressed)


def test_msr_parser_rejects_mixed_volumes(tmp_path: Path):
    # Two disks in one file: block 100 of disk 0 and block 100 of disk 1 are
    # unrelated, so merging them would fabricate a rewrite.
    path = tmp_path / "mixed.csv"
    path.write_text(
        "128166372010810581,hm,0,Write,409600,4096,1\n"
        "128166372010810582,hm,1,Write,409600,4096,1\n", encoding="utf-8")

    with pytest.raises(MultipleDevicesError):
        get_parser("msr").parse(path, trace_id="mixed")

    # With an explicit policy the busiest device is kept and the rest counted.
    table, stats = get_parser("msr", device="hm_0").parse(path, trace_id="mixed")
    assert len(table) == 1
    assert stats.dropped_other_device == 1
    assert stats.device_selected == "hm_0"


def test_fiu_parser_units(fiu_txt: Path):
    table, stats = get_parser("fiu").parse(fiu_txt, trace_id="fiu_homes")

    assert stats.raw_records_read == 5
    # 8 sectors = 4096 B = 1 block; 16 sectors = 8192 B = 2 blocks.
    assert len(table) == 1 + 1 + 2 + 1 + 1
    # LBA 160806224 sectors * 512 B = 82332786688 B; / 4096 = 20100778.
    assert table["lba"].iloc[0] == 160806224 * 512 // 4096
    # Nanoseconds -> microseconds, rebased.
    assert table["timestamp"].iloc[0] == 0
    # Each record is converted to microseconds first, then rebased, so the
    # difference is taken between the truncated values.
    assert table["timestamp"].iloc[1] == 777599999953380 // 1000 - 777599999908936 // 1000
    assert stats.device_selected == "6:0"


def test_fiu_parser_refuses_to_pool_devices(fiu_multidevice_txt: Path):
    with pytest.raises(MultipleDevicesError):
        get_parser("fiu").parse(fiu_multidevice_txt, trace_id="mixed")

    table, stats = get_parser("fiu", device="dominant").parse(fiu_multidevice_txt,
                                                              trace_id="mixed")
    assert stats.device_selected == "6:0"
    assert len(table) == 3
    assert stats.dropped_other_device == 1


def test_systor_parser_units_and_blank_iotype(systor_csv: Path):
    table, stats = get_parser("systor").parse(systor_csv, trace_id="systor_lun0")

    assert stats.raw_records_read == 5
    # The blank IOType row cannot be classified and must be dropped, not guessed.
    assert stats.dropped_unknown_operation == 1
    assert len(table) == 1 + 1 + 2 + 1
    # Fractional Unix seconds -> microseconds, rebased.
    assert table["timestamp"].iloc[0] == 0
    assert table["timestamp"].iloc[1] == 100000        # 0.1 s
    assert table["lba"].iloc[0] == 1                   # offset 4096 / 4096
    assert stats.device_selected == "0"


def test_parser_counts_every_dropped_record(tmp_path: Path):
    path = tmp_path / "dirty.csv"
    path.write_text(
        "128166372010810581,hm,0,Write,4096,4096,1\n"
        "not_a_number,hm,0,Write,4096,4096,1\n"          # malformed timestamp
        "128166372010810583,hm,0,Flush,4096,4096,1\n"    # unknown operation
        "128166372010810584,hm,0,Write,-1,4096,1\n"      # negative offset
        "128166372010810585,hm,0,Write,4096,0,1\n"       # zero size
        "128166372010810586,hm,0,Write,4096,99999999,1\n",  # exceeds the block cap
        encoding="utf-8")

    table, stats = get_parser("msr", max_blocks_per_request=8).parse(path, trace_id="dirty")
    assert len(table) == 1
    assert stats.dropped_malformed == 1
    assert stats.dropped_unknown_operation == 1
    assert stats.dropped_bad_offset == 1
    assert stats.dropped_bad_size == 1
    assert stats.dropped_oversized_request == 1
    assert stats.dropped_total == 5
    assert "FLUSH" in stats.unknown_operation_examples


def test_parser_output_is_chronological(msr_csv: Path, tmp_path: Path):
    shuffled = tmp_path / "shuffled.csv"
    lines = msr_csv.read_text(encoding="utf-8").strip().splitlines()
    shuffled.write_text("\n".join([lines[3], lines[0], lines[5], lines[1], lines[2], lines[4]])
                        + "\n", encoding="utf-8")

    table, stats = get_parser("msr").parse(shuffled, trace_id="t")
    timestamps = table["timestamp"].to_numpy()
    assert (np.diff(timestamps) >= 0).all()
    assert stats.out_of_order_records > 0        # the disorder is reported, not hidden


def test_parser_handles_empty_file(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    table, stats = get_parser("msr").parse(path, trace_id="empty")
    assert table.empty
    assert stats.blocks_emitted == 0


def test_max_records_keeps_a_chronological_prefix(msr_csv: Path):
    full, _ = get_parser("msr").parse(msr_csv, trace_id="t")
    capped, stats = get_parser("msr").parse(msr_csv, trace_id="t", max_records=3)
    assert stats.truncated_by_limit
    assert stats.raw_records_read == 3
    # The prefix must be identical to the head of the full parse.
    assert capped["lba"].tolist() == full["lba"].tolist()[: len(capped)]


def test_real_msr_trace_parses(real_msr_trace: Path):
    """Sanity check against genuine data when it is available on this machine."""
    table, stats = get_parser("msr").parse(real_msr_trace, trace_id="real",
                                           max_records=50_000)
    assert stats.raw_records_read == 50_000
    assert stats.dropped_malformed == 0
    assert len(table) >= 50_000              # expansion never loses records
    assert (np.diff(table["timestamp"].to_numpy()) >= 0).all()
    assert set(table["operation"]) <= {"W", "R", "D"}
    assert (table["size"] == 1).all()
    # One volume per file is the distribution's own guarantee.
    assert stats.device_selected != ""
    assert len(stats.device_record_counts) == 1
