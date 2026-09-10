"""Prediction contract and end-to-end integration with the C++ simulator.

These tests build a tiny model on a synthetic normalized trace, export
predictions through the real contract, and replay the trace through the built
simulator binary under all three policies. Synthetic data is legitimate here:
the test is about the plumbing being correct, not about a research result.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ml.config import MlConfig
from ml.inference.predict import PREDICTION_COLUMNS, generate_predictions
from ml.models.lstm import LstmHyperparameters, build_model
from ml.preprocessing.sequences import IntervalScaler
from ml.training.common import save_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent
SIMULATOR_CANDIDATES = (
    REPO_ROOT / "simulator" / "build" / "smartgc_sim.exe",
    REPO_ROOT / "simulator" / "build" / "smartgc_sim",
    REPO_ROOT / "simulator" / "build" / "Release" / "smartgc_sim.exe",
)


def simulator_binary() -> Path:
    for candidate in SIMULATOR_CANDIDATES:
        if candidate.is_file():
            return candidate
    pytest.skip("simulator not built (python scripts/build_simulator.py)")


def write_synthetic_normalized_trace(path: Path, n_lbas: int = 60,
                                     n_events: int = 24_000,
                                     hot_fraction: float = 0.25,
                                     hot_traffic: float = 0.8) -> None:
    """A skewed workload in the normalized contract, with real rewrite structure.

    The clock advances one microsecond per event and the hot quarter of the LBA
    space receives 80% of the writes, so a hot LBA is revisited roughly every
    `n_hot / hot_traffic` events and a cold one roughly every
    `n_cold / (1 - hot_traffic)`. That produces genuinely different per-LBA
    rewrite intervals - which a naive "increment the clock by a different amount
    per LBA" generator does not, because the gap between two writes to one LBA
    is the sum of every increment in between and ends up the same for all LBAs.
    """
    rng = np.random.default_rng(3)
    n_hot = max(1, int(n_lbas * hot_fraction))
    rows = []
    for clock in range(1, n_events + 1):
        if rng.random() < hot_traffic:
            lba = int(rng.integers(0, n_hot))
        else:
            lba = int(rng.integers(n_hot, n_lbas))
        rows.append((clock, lba, 1, "W", "synthetic_test"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "lba", "size", "operation", "trace_id"])
        writer.writerows(rows)


@pytest.fixture()
def prepared_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A normalized trace plus a checkpoint, wired into temporary directories."""
    normalized_dir = tmp_path / "normalized"
    trace_id = "synthetic_test"
    trace_path = normalized_dir / f"{trace_id}.csv"
    write_synthetic_normalized_trace(trace_path)

    import ml.training.common as common
    import ml.inference.predict as predict_module
    monkeypatch.setattr(common, "NORMALIZED_DIR", normalized_dir)
    monkeypatch.setattr(common, "SEQUENCE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(predict_module, "normalized_trace_path",
                        lambda tid: normalized_dir / f"{tid}.csv")

    sequence_length = 4
    hyperparameters = LstmHyperparameters(sequence_length=sequence_length, hidden_dim=8,
                                          num_layers=1, dropout=0.0)
    model = build_model(hyperparameters, seed=0)
    scaler = IntervalScaler(mean=3.0, std=1.5, fitted=True)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(model, scaler,
                    {"stage": "unit-test", "hot_threshold_microseconds": 30.0}, checkpoint)
    return trace_id, trace_path, checkpoint, tmp_path


def test_prediction_csv_matches_the_contract(prepared_trace):
    trace_id, trace_path, checkpoint, tmp_path = prepared_trace
    output_dir = tmp_path / "predictions"

    summary = generate_predictions(trace_id, checkpoint, model_version="unit",
                                   output_dir=output_dir, verbose=False)
    predictions_path = output_dir / f"unit__{trace_id}.csv"
    assert predictions_path.is_file()

    with predictions_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)

    assert header == PREDICTION_COLUMNS
    assert len(rows) == summary["predicted_write_events"]
    for row in rows[:200]:
        assert len(row) == len(PREDICTION_COLUMNS)
        assert int(row[0]) >= 0                 # timestamp
        assert int(row[1]) >= 0                 # lba
        assert float(row[2]) >= 0               # predicted interval, never negative
        assert row[3] in ("HOT", "COLD")
        assert row[4] == "unit"
        assert row[5] == trace_id

    # Rows must be chronological so the simulator can merge-join in one pass.
    timestamps = [int(row[0]) for row in rows]
    assert timestamps == sorted(timestamps)

    # Coverage is measured, never assumed to be total: early writes of each LBA
    # have too little history to score.
    assert 0.0 < summary["prediction_coverage"] < 1.0
    assert summary["duplicate_timestamp_lba_rows"] == 0
    assert summary["hot_threshold_microseconds"] == 30.0

    sidecar = json.loads((output_dir / f"unit__{trace_id}.json").read_text(encoding="utf-8"))
    assert sidecar["trace_id"] == trace_id
    assert sidecar["model_stage"] == "unit-test"


def test_predictions_align_with_actual_write_events(prepared_trace):
    trace_id, trace_path, checkpoint, tmp_path = prepared_trace
    output_dir = tmp_path / "predictions"
    generate_predictions(trace_id, checkpoint, model_version="unit",
                         output_dir=output_dir, verbose=False)

    import pandas as pd
    trace = pd.read_csv(trace_path)
    predictions = pd.read_csv(output_dir / f"unit__{trace_id}.csv")

    write_keys = set(zip(trace.loc[trace["operation"] == "W", "timestamp"],
                         trace.loc[trace["operation"] == "W", "lba"]))
    prediction_keys = set(zip(predictions["timestamp"], predictions["lba"]))
    # Every prediction must correspond to a real write event in the trace.
    assert prediction_keys <= write_keys


def test_prediction_requires_a_training_derived_threshold(prepared_trace, tmp_path: Path):
    trace_id, _, _, _ = prepared_trace
    model = build_model(LstmHyperparameters(sequence_length=4, hidden_dim=8, num_layers=1))
    scaler = IntervalScaler(mean=0.0, std=1.0, fitted=True)
    checkpoint = tmp_path / "no_threshold.pt"
    save_checkpoint(model, scaler, {"stage": "unit-test"}, checkpoint)

    with pytest.raises(ValueError, match="threshold"):
        generate_predictions(trace_id, checkpoint, output_dir=tmp_path / "out", verbose=False)


def run_simulator(args: list[str]) -> dict:
    binary = simulator_binary()
    result = subprocess.run([str(binary), *args], capture_output=True, text=True,
                            cwd=str(REPO_ROOT))
    assert result.returncode == 0, f"simulator failed: {result.stderr}"
    return result


def test_end_to_end_all_three_policies(prepared_trace, tmp_path: Path):
    trace_id, trace_path, checkpoint, _ = prepared_trace
    output_dir = tmp_path / "predictions"
    generate_predictions(trace_id, checkpoint, model_version="unit",
                         output_dir=output_dir, verbose=False)
    predictions_path = output_dir / f"unit__{trace_id}.csv"

    metrics_csv = tmp_path / "metrics.csv"
    reports: dict[str, dict] = {}

    common = ["--trace", str(trace_path),
              "--total-segments", "24", "--blocks-per-segment", "8",
              "--gc-reserved", "2", "--dataset", "synthetic",
              "--trace-name", trace_id, "--quiet",
              "--export-metrics", str(metrics_csv)]

    for policy, extra in (
            ("MIXED", ["--model-type", "none"]),
            ("RULE_BASED", ["--model-type", "rule", "--rule-threshold-us", "30",
                            "--rule-min-history", "4"]),
            ("LSTM_SMARTGC", ["--model-type", "unit",
                              "--predictions", str(predictions_path)])):
        report = tmp_path / f"{policy}.json"
        run_simulator(common + ["--placement", policy, "--report-json", str(report)] + extra)
        reports[policy] = json.loads(report.read_text(encoding="utf-8"))

    # Fairness: every policy consumed exactly the same logical workload.
    write_counts = {p: r["write_requests"] for p, r in reports.items()}
    logical = {p: r["logical_bytes_written"] for p, r in reports.items()}
    assert len(set(write_counts.values())) == 1, write_counts
    assert len(set(logical.values())) == 1, logical

    for policy, report in reports.items():
        # Accounting identity holds for every policy.
        assert report["physical_bytes_written"] == (report["logical_bytes_written"]
                                                    + report["gc_bytes_copied"])
        assert report["waf"] >= 1.0
        assert report["policy"] == policy

    # MIXED never classifies; the other two do.
    assert reports["MIXED"]["cold_placements"] == 0
    assert reports["MIXED"]["unknown_placements"] == reports["MIXED"]["write_requests"]
    assert reports["RULE_BASED"]["rule_classified_hot"] > 0
    assert reports["LSTM_SMARTGC"]["prediction_rows_matched"] > 0
    assert reports["LSTM_SMARTGC"]["prediction_malformed_rows"] == 0
    assert reports["LSTM_SMARTGC"]["prediction_out_of_order_input"] is False

    # Every prediction row was either matched to a write or accounted as unmatched.
    smart = reports["LSTM_SMARTGC"]
    assert smart["prediction_rows_matched"] + smart["prediction_unmatched"] == \
        smart["prediction_rows_read"]

    # One metrics row per policy, all sharing the same header.
    with metrics_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {row["policy"] for row in rows} == {"MIXED", "RULE_BASED", "LSTM_SMARTGC"}


def test_simulator_refuses_smart_placement_without_predictions(prepared_trace, tmp_path: Path):
    trace_id, trace_path, _, _ = prepared_trace
    binary = simulator_binary()
    result = subprocess.run(
        [str(binary), "--trace", str(trace_path), "--placement", "LSTM_SMARTGC", "--quiet"],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    # Silently degrading to MIXED would invalidate the comparison, so this is an error.
    assert result.returncode == 2
    assert "requires --predictions" in result.stderr

    result = subprocess.run(
        [str(binary), "--trace", str(trace_path), "--placement", "RULE_BASED", "--quiet"],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 2
    assert "rule-threshold-us" in result.stderr


def test_simulator_reports_device_full_distinctly(tmp_path: Path):
    trace_path = tmp_path / "big.csv"
    write_synthetic_normalized_trace(trace_path, n_lbas=4000, n_events=12_000)
    binary = simulator_binary()
    result = subprocess.run(
        [str(binary), "--trace", str(trace_path), "--placement", "MIXED",
         "--total-segments", "6", "--blocks-per-segment", "4", "--quiet"],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 3
    assert "DEVICE FULL" in result.stderr
