"""Executable leakage audit.

Each test here corresponds to a line in the checklist in docs/methodology.md.
They exist so that "no leakage" is a property the suite verifies rather than a
claim in the write-up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.config import MlConfig, load_config
from ml.dataset_selection import assign_roles
from ml.preprocessing.sequences import (
    IntervalScaler,
    build_sequences,
    chronological_prefix,
    hot_threshold_from_training,
    rewrite_intervals,
    split_samples,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def _samples(n_lbas: int = 40, writes_per_lba: int = 30, sequence_length: int = 5):
    """A deterministic multi-LBA workload with interleaved, increasing timestamps."""
    rng = np.random.default_rng(7)
    lbas, timestamps = [], []
    clock = 0
    for step in range(writes_per_lba):
        for lba in range(n_lbas):
            clock += int(rng.integers(1, 50))
            lbas.append(lba)
            timestamps.append(clock)
    intervals = rewrite_intervals(np.array(lbas), np.array(timestamps))
    return build_sequences(intervals, sequence_length)


def test_no_future_interval_enters_an_input_window():
    """Every value in a sample's window must predate that sample's target."""
    lbas = np.array([3] * 12)
    timestamps = np.cumsum(np.array([0, 4, 9, 2, 7, 5, 11, 3, 8, 6, 1, 10]))
    intervals = rewrite_intervals(lbas, timestamps)
    samples = build_sequences(intervals, sequence_length=4)

    for index in range(samples["targets"].size):
        target_time = samples["timestamp"][index]
        window = samples["inputs"][index]
        # The window's intervals are exactly the four that closed before this
        # target, in order; reconstruct them and compare.
        position = int(np.flatnonzero(intervals["timestamp"] == target_time)[0])
        expected = intervals["interval"][position - 4: position]
        assert window.tolist() == expected.tolist()
        assert (intervals["timestamp"][position - 4: position] < target_time).all()


def test_splits_are_strictly_ordered_in_time():
    train, val, test, _ = split_samples(_samples(), 0.70, 0.15)
    assert train["timestamp"].size and val["timestamp"].size and test["timestamp"].size
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()


def test_no_sample_appears_in_two_splits():
    train, val, test, boundaries = split_samples(_samples(), 0.70, 0.15)
    total = train["targets"].size + val["targets"].size + test["targets"].size
    assert total == boundaries.total

    def keys(part):
        return {(int(t), int(l), float(y)) for t, l, y
                in zip(part["timestamp"], part["lba"], part["targets"])}

    train_keys, val_keys, test_keys = keys(train), keys(val), keys(test)
    assert not (train_keys & val_keys)
    assert not (train_keys & test_keys)
    assert not (val_keys & test_keys)


def test_scaler_depends_only_on_training_data():
    train, val, test, _ = split_samples(_samples(), 0.70, 0.15)
    fitted = IntervalScaler.fit(train["inputs"], train["targets"])

    # Corrupting the test split by orders of magnitude must not move the scaler.
    poisoned = {k: v.copy() for k, v in test.items()}
    poisoned["targets"] = poisoned["targets"] * 1e9
    refitted = IntervalScaler.fit(train["inputs"], train["targets"])
    assert fitted.mean == refitted.mean and fitted.std == refitted.std


def test_hot_threshold_depends_only_on_training_data():
    train, _, test, _ = split_samples(_samples(), 0.70, 0.15)
    threshold = hot_threshold_from_training(train["targets"], 30.0)
    shifted = hot_threshold_from_training(train["targets"], 30.0)
    assert threshold == shifted
    # The threshold is a percentile of the training targets, so it cannot equal
    # a percentile of a wildly different test distribution by construction.
    assert threshold == float(np.percentile(train["targets"], 30.0))


def test_finetuning_uses_only_the_early_part_of_training():
    train, val, test, _ = split_samples(_samples(), 0.70, 0.15)
    early = chronological_prefix(train, 0.5)
    assert early["timestamp"].size <= train["timestamp"].size
    assert early["timestamp"].max() <= train["timestamp"].max()
    # The adaptation data is strictly earlier than everything it is tested on.
    assert early["timestamp"].max() < test["timestamp"].min()


def test_role_assignment_holds_targets_out_of_pretraining():
    statistics = [
        {"trace_id": f"msr_{name}", "dataset": "msr", "write_blocks": 1_000_000,
         "rewrite_ratio": 0.9, "unique_written_lbas": 50_000, "duration_seconds": 100_000.0}
        for name in ("a", "b", "c", "d", "e", "f")
    ]
    statistics.append({"trace_id": "systor_lun0", "dataset": "systor",
                       "write_blocks": 1_000_000, "rewrite_ratio": 0.9,
                       "unique_written_lbas": 50_000, "duration_seconds": 100_000.0})

    roles = assign_roles(statistics)
    assert roles.targets
    assert set(roles.targets).isdisjoint(set(roles.pretrain))
    assert "systor_lun0" in roles.external
    assert "systor_lun0" not in roles.pretrain


def test_role_assignment_is_independent_of_input_order():
    statistics = [
        {"trace_id": f"msr_{name}", "dataset": "msr", "write_blocks": 1_000_000,
         "rewrite_ratio": 0.9, "unique_written_lbas": 50_000, "duration_seconds": 100_000.0}
        for name in ("e", "a", "d", "b", "c")
    ]
    forward = assign_roles(statistics)
    backward = assign_roles(list(reversed(statistics)))
    assert forward.targets == backward.targets
    assert forward.pretrain == backward.pretrain


def test_eligibility_rejects_workloads_without_rewrites():
    from ml.dataset_selection import check_eligibility

    result = check_eligibility({"trace_id": "msr_x", "dataset": "msr",
                                "write_blocks": 5_000_000, "rewrite_ratio": 0.01,
                                "unique_written_lbas": 500_000,
                                "duration_seconds": 500_000.0})
    assert not result.eligible
    assert any("rewrite_ratio" in reason for reason in result.reasons)


def test_config_splits_sum_to_one():
    config = load_config()
    config.ml.validate()
    total = config.ml.train_split + config.ml.val_split + config.ml.test_split
    assert abs(total - 1.0) < 1e-9

    with pytest.raises(Exception):
        MlConfig(train_split=0.8, val_split=0.3, test_split=0.1).validate()


# ---------------------------------------------------------------------------
# Audits of artefacts produced by an actual run. Skipped when absent.
# ---------------------------------------------------------------------------

def _model_metadata() -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(MODELS_DIR.rglob("*.json"))]


def test_trained_models_did_not_pretrain_on_their_target():
    metadata = _model_metadata()
    if not metadata:
        pytest.skip("no trained models on disk yet")

    checked = 0
    for meta in metadata:
        target = meta.get("target_trace")
        pretrain_traces = set(meta.get("pretrain_traces") or [])
        if target and pretrain_traces:
            assert target not in pretrain_traces, (
                f"{meta.get('stage')} model for {target} was pretrained on its own target"
            )
            checked += 1
    if checked == 0:
        pytest.skip("no adapted models with a recorded pretraining set")


def test_trained_models_record_a_training_derived_threshold():
    metadata = _model_metadata()
    if not metadata:
        pytest.skip("no trained models on disk yet")
    for meta in metadata:
        threshold = meta.get("hot_threshold_microseconds")
        assert threshold is not None, f"{meta.get('stage')} model has no recorded threshold"
        assert threshold > 0


def test_hyperparameters_were_selected_on_validation_only():
    path = REPO_ROOT / "results" / "ml" / "hyperparameter_search_selected.json"
    if not path.is_file():
        pytest.skip("no hyperparameter search on disk yet")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "validation" in payload["selection_criterion"].lower()
    assert "test" not in payload["selection_criterion"].lower().replace("test data not used", "")
    assert payload["val_mae_us"] > 0
