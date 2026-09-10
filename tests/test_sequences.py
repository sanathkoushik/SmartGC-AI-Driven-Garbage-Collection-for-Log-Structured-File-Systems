"""Rewrite-interval extraction, sequence windows and chronological splitting."""

from __future__ import annotations

import numpy as np
import pytest

from ml.preprocessing.sequences import (
    IntervalScaler,
    build_sequences,
    chronological_prefix,
    chronological_split,
    classify,
    hot_threshold_from_training,
    order_chronologically,
    rewrite_intervals,
    split_samples,
)


def test_rewrite_intervals_match_the_worked_example():
    # LBA 100 written at 10, 18, 24, 40 -> intervals 8, 6, 16.
    lbas = np.array([100, 100, 100, 100])
    timestamps = np.array([10, 18, 24, 40])
    result = rewrite_intervals(lbas, timestamps)

    assert result["interval"].tolist() == [8, 6, 16]
    # The timestamp of each interval is the write that closed it.
    assert result["timestamp"].tolist() == [18, 24, 40]
    assert result["position"].tolist() == [0, 1, 2]


def test_first_write_never_produces_a_target():
    lbas = np.array([1, 2, 3])
    timestamps = np.array([0, 5, 9])
    result = rewrite_intervals(lbas, timestamps)
    # Three distinct LBAs, each written once: no intervals exist at all. A
    # fabricated zero here would teach the model that new blocks are red hot.
    assert result["interval"].size == 0


def test_intervals_are_computed_per_lba_not_globally():
    # Interleaved writes to two LBAs: intervals must not mix across them.
    lbas = np.array([1, 2, 1, 2, 1])
    timestamps = np.array([0, 1, 10, 21, 30])
    result = rewrite_intervals(lbas, timestamps)

    lba1 = result["interval"][result["lba"] == 1]
    lba2 = result["interval"][result["lba"] == 2]
    assert lba1.tolist() == [10, 20]      # 0 -> 10 -> 30
    assert lba2.tolist() == [20]          # 1 -> 21


def test_zero_interval_is_kept_as_a_real_observation():
    lbas = np.array([7, 7, 7])
    timestamps = np.array([100, 100, 105])
    result = rewrite_intervals(lbas, timestamps)
    assert result["interval"].tolist() == [0, 5]


def test_sequences_never_cross_an_lba_boundary():
    lbas = np.array([1] * 6 + [2] * 6)
    timestamps = np.concatenate([np.arange(0, 60, 10), np.arange(5, 65, 10)])
    intervals = rewrite_intervals(lbas, timestamps)
    # 5 intervals per LBA; with L=3 each LBA yields 5 - 3 = 2 samples.
    samples = build_sequences(intervals, sequence_length=3)

    assert samples["inputs"].shape == (4, 3)
    assert samples["targets"].size == 4
    # Every sample belongs to exactly one LBA.
    assert sorted(samples["lba"].tolist()) == [1, 1, 2, 2]


def test_sequence_window_contents_are_the_preceding_intervals():
    lbas = np.array([9] * 6)
    timestamps = np.array([0, 1, 3, 6, 10, 15])          # intervals 1,2,3,4,5
    intervals = rewrite_intervals(lbas, timestamps)
    samples = build_sequences(intervals, sequence_length=3)

    assert samples["inputs"].tolist() == [[1, 2, 3], [2, 3, 4]]
    assert samples["targets"].tolist() == [4, 5]
    # The target's timestamp is when it became observable.
    assert samples["timestamp"].tolist() == [10, 15]


def test_too_short_history_yields_no_samples():
    intervals = rewrite_intervals(np.array([1, 1]), np.array([0, 5]))
    assert build_sequences(intervals, sequence_length=3)["targets"].size == 0
    assert build_sequences({"interval": np.array([]), "group_id": np.array([]),
                            "timestamp": np.array([]), "lba": np.array([])},
                           sequence_length=3)["targets"].size == 0


def test_build_sequences_rejects_invalid_length():
    intervals = rewrite_intervals(np.array([1, 1, 1]), np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        build_sequences(intervals, sequence_length=0)


def test_chronological_split_is_ordered_and_disjoint():
    timestamps = np.arange(100, dtype=np.int64)
    boundaries = chronological_split(timestamps, 0.70, 0.15)

    assert boundaries.train_size == 70
    assert boundaries.val_size == 15
    assert boundaries.test_size == 15
    assert boundaries.train_size + boundaries.val_size + boundaries.test_size == 100


def test_split_boundary_never_cuts_a_tied_timestamp():
    # 40 samples all sharing timestamp 5 straddle the 70% mark; the boundary
    # must move past them so no instant appears in two splits.
    timestamps = np.array([0] * 50 + [5] * 40 + [9] * 10, dtype=np.int64)
    boundaries = chronological_split(timestamps, 0.70, 0.15)

    train = timestamps[: boundaries.train_end]
    val = timestamps[boundaries.train_end: boundaries.val_end]
    test = timestamps[boundaries.val_end:]
    if train.size and val.size:
        assert train.max() < val.min()
    if val.size and test.size:
        assert val.max() < test.min()
    assert set(train) & set(val) == set()
    assert set(val) & set(test) == set()


def test_chronological_split_rejects_unsorted_input():
    with pytest.raises(ValueError):
        chronological_split(np.array([5, 1, 9]), 0.7, 0.15)


def test_split_samples_orders_globally_before_cutting():
    # Sequence building groups by LBA, so raw sample order is not chronological.
    lbas = np.array([1] * 8 + [2] * 8)
    timestamps = np.concatenate([np.arange(0, 800, 100), np.arange(50, 850, 100)])
    intervals = rewrite_intervals(lbas, timestamps)
    samples = build_sequences(intervals, sequence_length=2)

    train, val, test, boundaries = split_samples(samples, 0.6, 0.2)
    for part in (train, val, test):
        if part["timestamp"].size > 1:
            assert (np.diff(part["timestamp"]) >= 0).all()
    if train["timestamp"].size and val["timestamp"].size:
        assert train["timestamp"].max() < val["timestamp"].min()
    if val["timestamp"].size and test["timestamp"].size:
        assert val["timestamp"].max() < test["timestamp"].min()
    assert boundaries.total == samples["targets"].size


def test_order_chronologically_is_stable_and_complete():
    samples = {
        "inputs": np.arange(12, dtype=np.float64).reshape(4, 3),
        "targets": np.array([1.0, 2.0, 3.0, 4.0]),
        "timestamp": np.array([30, 10, 20, 10], dtype=np.int64),
        "lba": np.array([1, 2, 3, 4]),
    }
    ordered = order_chronologically(samples)
    assert ordered["timestamp"].tolist() == [10, 10, 20, 30]
    # Stability: the two ties keep their original relative order (LBA 2 then 4).
    assert ordered["lba"].tolist() == [2, 4, 3, 1]
    assert sorted(ordered["targets"].tolist()) == [1.0, 2.0, 3.0, 4.0]


def test_chronological_prefix_keeps_the_earliest_fraction():
    samples = {
        "inputs": np.zeros((10, 2)),
        "targets": np.arange(10, dtype=np.float64),
        "timestamp": np.arange(10, dtype=np.int64),
        "lba": np.zeros(10, dtype=np.int64),
    }
    prefix = chronological_prefix(samples, 0.5)
    assert prefix["timestamp"].tolist() == [0, 1, 2, 3, 4]


def test_interval_scaler_round_trips():
    train_inputs = np.array([[1.0, 10.0, 100.0], [5.0, 50.0, 500.0]])
    train_targets = np.array([20.0, 200.0])
    scaler = IntervalScaler.fit(train_inputs, train_targets)

    values = np.array([0.0, 1.0, 10.0, 1_000_000.0])
    recovered = scaler.inverse_transform(scaler.transform(values))
    assert np.allclose(recovered, values, rtol=1e-6, atol=1e-6)
    # Never produces a negative interval, whatever the model emits.
    assert (scaler.inverse_transform(np.array([-50.0, -1e3])) >= 0).all()


def test_interval_scaler_handles_degenerate_input():
    constant = np.full((4, 3), 7.0)
    scaler = IntervalScaler.fit(constant, np.full(4, 7.0))
    assert scaler.std == 1.0                      # no division by zero
    assert np.isfinite(scaler.transform(constant)).all()

    with pytest.raises(ValueError):
        IntervalScaler.fit(np.empty((0, 3)), np.empty(0))


def test_hot_threshold_is_a_training_percentile():
    train = np.arange(1, 101, dtype=np.float64)
    threshold = hot_threshold_from_training(train, 30.0)
    assert np.isclose(threshold, np.percentile(train, 30.0))

    labels = classify(np.array([1.0, threshold, threshold + 1]), threshold)
    assert labels.tolist() == [1, 1, 0]

    with pytest.raises(ValueError):
        hot_threshold_from_training(np.empty(0), 30.0)
    with pytest.raises(ValueError):
        hot_threshold_from_training(train, 0.0)
