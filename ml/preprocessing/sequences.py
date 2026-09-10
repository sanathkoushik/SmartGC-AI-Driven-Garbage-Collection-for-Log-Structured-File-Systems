"""Rewrite-interval extraction and leakage-safe chronological sequence building.

Definitions
-----------
For a logical block address, the **rewrite interval** of a write is the time
since the previous write *to the same LBA*::

    LBA 100 written at t = 10, 18, 24, 40
    intervals             =  8,  6, 16

The first write of an LBA has no interval and never becomes a target. A zero is
a real, observable interval (two writes in the same microsecond); a fabricated
zero for a first write would teach the model that every new block is maximally
hot, so first writes are excluded outright rather than filled in.

Reads never produce rewrite intervals. They may appear in a normalized trace and
are simply ignored here.

Sequence construction
---------------------
A training sample is ``L`` consecutive intervals of one LBA predicting the next::

    [i(t-L+1) ... i(t)]  ->  i(t+1)

Windows never cross an LBA boundary. Each sample is stamped with the timestamp
of the write that produced its **target**, which is the only time at which the
label becomes observable, and is what the chronological split orders by.

Leakage rules enforced here
---------------------------
* Splits are chronological by target timestamp, never random.
* A split boundary is advanced past ties so that no timestamp appears on both
  sides; ``max(train) < min(validation) < min(test)`` holds strictly.
* Feature windows only ever contain intervals whose target time is earlier than
  the sample's own target time, so no future information enters the input.
* Normalization statistics are fitted by the caller on the training split alone
  (see :class:`IntervalScaler`).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SplitBoundaries:
    """Index boundaries of a chronological three-way split."""

    train_end: int
    val_end: int
    total: int

    @property
    def train_size(self) -> int:
        return self.train_end

    @property
    def val_size(self) -> int:
        return self.val_end - self.train_end

    @property
    def test_size(self) -> int:
        return self.total - self.val_end

    def to_dict(self) -> dict[str, int]:
        return {"train_end": self.train_end, "val_end": self.val_end, "total": self.total,
                "train_size": self.train_size, "val_size": self.val_size,
                "test_size": self.test_size}


def rewrite_intervals(lbas: np.ndarray, timestamps: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-LBA rewrite intervals from a chronological write stream.

    Parameters
    ----------
    lbas, timestamps
        Equal-length arrays of write events in chronological order. Only writes
        may be passed; filter reads out first.

    Returns
    -------
    dict with equal-length arrays, ordered by (lba, time):
        ``lba``        LBA the interval belongs to
        ``timestamp``  time of the write that *closes* the interval
        ``interval``   timestamp minus the previous write time of that LBA
        ``group_id``   dense index of the LBA group
        ``position``   0-based index of this interval within its LBA's history
    """
    lbas = np.asarray(lbas)
    timestamps = np.asarray(timestamps)
    if lbas.shape != timestamps.shape:
        raise ValueError("lbas and timestamps must have the same shape")
    if lbas.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return {"lba": empty_i, "timestamp": empty_i, "interval": empty_i,
                "group_id": empty_i, "position": empty_i}

    # A stable sort by LBA groups each LBA's writes while preserving their
    # chronological order within the group.
    order = np.argsort(lbas, kind="stable")
    sorted_lba = lbas[order]
    sorted_ts = timestamps[order].astype(np.int64, copy=False)

    starts_group = np.empty(sorted_lba.size, dtype=bool)
    starts_group[0] = True
    starts_group[1:] = sorted_lba[1:] != sorted_lba[:-1]

    previous_ts = np.empty_like(sorted_ts)
    previous_ts[0] = 0
    previous_ts[1:] = sorted_ts[:-1]
    interval = sorted_ts - previous_ts

    # The first write of each LBA has no predecessor and is dropped entirely.
    has_interval = ~starts_group

    group_id_all = np.cumsum(starts_group) - 1
    group_starts = np.flatnonzero(starts_group)
    position_in_group = np.arange(sorted_lba.size) - group_starts[group_id_all]

    return {
        "lba": sorted_lba[has_interval],
        "timestamp": sorted_ts[has_interval],
        "interval": interval[has_interval],
        "group_id": group_id_all[has_interval],
        # position within the LBA's *interval* history: the first interval is 0
        "position": position_in_group[has_interval] - 1,
    }


def build_sequences(intervals: dict[str, np.ndarray],
                    sequence_length: int) -> dict[str, np.ndarray]:
    """Build ``(L intervals -> next interval)`` samples that never cross an LBA.

    Returns arrays of equal length:
        ``inputs``     (N, L) float64 windows of past intervals
        ``targets``    (N,)   the next interval
        ``timestamp``  (N,)   time at which the target became observable
        ``lba``        (N,)   LBA the sample belongs to
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")

    values = intervals["interval"]
    n = values.size
    window = sequence_length + 1
    if n < window:
        return {"inputs": np.empty((0, sequence_length), dtype=np.float64),
                "targets": np.empty(0, dtype=np.float64),
                "timestamp": np.empty(0, dtype=np.int64),
                "lba": np.empty(0, dtype=np.int64)}

    group_id = intervals["group_id"]
    # A window starting at p is valid only if p and p+L belong to the same LBA;
    # because the array is grouped and time-ordered, that single check also
    # guarantees every element in between belongs to it.
    starts = np.arange(n - window + 1)
    same_group = group_id[starts] == group_id[starts + sequence_length]
    starts = starts[same_group]
    if starts.size == 0:
        return {"inputs": np.empty((0, sequence_length), dtype=np.float64),
                "targets": np.empty(0, dtype=np.float64),
                "timestamp": np.empty(0, dtype=np.int64),
                "lba": np.empty(0, dtype=np.int64)}

    windows = np.lib.stride_tricks.sliding_window_view(values, sequence_length)[starts]
    target_index = starts + sequence_length
    return {
        "inputs": np.ascontiguousarray(windows, dtype=np.float64),
        "targets": values[target_index].astype(np.float64),
        "timestamp": intervals["timestamp"][target_index].astype(np.int64),
        "lba": intervals["lba"][target_index].astype(np.int64),
    }


def order_chronologically(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Sort samples by the time their target became observable.

    Sequence building groups by LBA, so the result is in LBA order; the split
    must see global chronological order instead.
    """
    if samples["targets"].size == 0:
        return samples
    order = np.argsort(samples["timestamp"], kind="stable")
    return {key: value[order] for key, value in samples.items()}


def chronological_split(timestamps: np.ndarray,
                        train_fraction: float,
                        val_fraction: float) -> SplitBoundaries:
    """Split an already time-ordered array into train / validation / test.

    Boundaries are advanced past tied timestamps so that no single instant
    appears in two splits. With heavy ties a split can therefore end up empty;
    callers must check, because silently returning an empty validation set would
    make model selection meaningless.
    """
    n = int(timestamps.size)
    if n == 0:
        return SplitBoundaries(0, 0, 0)
    if not 0.0 < train_fraction < 1.0 or not 0.0 < val_fraction < 1.0:
        raise ValueError("train_fraction and val_fraction must be in (0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must leave room for a test split")

    if np.any(np.diff(timestamps) < 0):
        raise ValueError("chronological_split requires timestamps in non-decreasing order")

    def snap(index: int) -> int:
        """Advance `index` to the first position with a new timestamp."""
        if index <= 0:
            return 0
        if index >= n:
            return n
        boundary_value = timestamps[index - 1]
        while index < n and timestamps[index] == boundary_value:
            index += 1
        return index

    train_end = snap(int(round(train_fraction * n)))
    val_end = snap(max(train_end, int(round((train_fraction + val_fraction) * n))))
    return SplitBoundaries(train_end=train_end, val_end=min(val_end, n), total=n)


def split_samples(samples: dict[str, np.ndarray],
                  train_fraction: float,
                  val_fraction: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
                                                dict[str, np.ndarray], SplitBoundaries]:
    """Order samples chronologically and cut them into train / val / test."""
    ordered = order_chronologically(samples)
    boundaries = chronological_split(ordered["timestamp"], train_fraction, val_fraction)

    def slice_range(start: int, stop: int) -> dict[str, np.ndarray]:
        return {key: value[start:stop] for key, value in ordered.items()}

    return (slice_range(0, boundaries.train_end),
            slice_range(boundaries.train_end, boundaries.val_end),
            slice_range(boundaries.val_end, boundaries.total),
            boundaries)


def chronological_prefix(samples: dict[str, np.ndarray], fraction: float) -> dict[str, np.ndarray]:
    """Keep the earliest `fraction` of samples. Used to fine-tune on early data only."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    ordered = order_chronologically(samples)
    cut = int(round(fraction * ordered["timestamp"].size))
    return {key: value[:cut] for key, value in ordered.items()}


@dataclass
class IntervalScaler:
    """log1p + standardisation, fitted on training data only.

    Rewrite intervals are strongly right-skewed and span many orders of
    magnitude, so the model works in ``log1p`` space; the mean and standard
    deviation are then taken from the *training* intervals alone. Fitting them
    on all data would leak the test distribution's scale into training.
    """

    mean: float = 0.0
    std: float = 1.0
    fitted: bool = False

    @classmethod
    def fit(cls, train_inputs: np.ndarray, train_targets: np.ndarray) -> "IntervalScaler":
        pool = np.concatenate([np.asarray(train_inputs).reshape(-1),
                               np.asarray(train_targets).reshape(-1)])
        if pool.size == 0:
            raise ValueError("cannot fit IntervalScaler on an empty training split")
        logged = np.log1p(np.clip(pool, 0.0, None))
        std = float(logged.std())
        # A degenerate split (every interval identical) would otherwise divide
        # by zero; 1.0 leaves the values centred but unscaled.
        return cls(mean=float(logged.mean()), std=std if std > 1e-12 else 1.0, fitted=True)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.log1p(np.clip(np.asarray(values, dtype=np.float64), 0.0, None))
                - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Map model output back to microseconds. Never returns a negative interval."""
        logged = np.asarray(values, dtype=np.float64) * self.std + self.mean
        return np.clip(np.expm1(np.clip(logged, -50.0, 50.0)), 0.0, None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntervalScaler":
        return cls(mean=float(data["mean"]), std=float(data["std"]),
                   fitted=bool(data.get("fitted", True)))


def hot_threshold_from_training(train_targets: np.ndarray, percentile: float) -> float:
    """Hot/cold cutoff in microseconds, derived from training targets only.

    An LBA is HOT when its predicted next rewrite interval is at or below this
    value. Deriving the cutoff from the test set would tune the classifier on
    the data it is evaluated against, so this function must never be handed test
    targets; ``tests/test_leakage.py`` asserts the pipeline honours that.
    """
    values = np.asarray(train_targets, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot derive a hot/cold threshold from an empty training split")
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be in (0, 100)")
    return float(np.percentile(values, percentile))


def classify(intervals: np.ndarray, threshold: float) -> np.ndarray:
    """1 = HOT (rewritten within `threshold`), 0 = COLD."""
    return (np.asarray(intervals, dtype=np.float64) <= threshold).astype(np.int8)
