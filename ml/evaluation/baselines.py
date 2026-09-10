"""Non-learned predictors of the next rewrite interval.

These exist so the LSTM has to earn its place. Each takes the same input window
the model sees -- the last ``L`` rewrite intervals of one LBA -- and returns a
prediction in the same unit, using no training and no parameters.

If a baseline matches the LSTM, that is the finding, and it is reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

Predictor = Callable[[np.ndarray], np.ndarray]


def last_interval(inputs: np.ndarray) -> np.ndarray:
    """Predict that the next interval equals the most recent one (persistence)."""
    return np.asarray(inputs, dtype=np.float64)[:, -1]


def mean_interval(inputs: np.ndarray) -> np.ndarray:
    """Predict the mean of the whole observed history in the window."""
    return np.asarray(inputs, dtype=np.float64).mean(axis=1)


def median_interval(inputs: np.ndarray) -> np.ndarray:
    """Median of the window: robust to a single very long idle gap."""
    return np.median(np.asarray(inputs, dtype=np.float64), axis=1)


def moving_average(window: int = 3) -> Predictor:
    """Mean of the `window` most recent intervals."""
    if window < 1:
        raise ValueError("window must be >= 1")

    def predict(inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64)
        k = min(window, values.shape[1])
        return values[:, -k:].mean(axis=1)

    return predict


def exponential_moving_average(alpha: float = 0.5) -> Predictor:
    """Recency-weighted mean; alpha is the weight of the most recent interval."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    def predict(inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64)
        length = values.shape[1]
        # Newest gets alpha, each older step decays by (1 - alpha); normalised so
        # the weights sum to one over the finite window.
        weights = alpha * (1.0 - alpha) ** np.arange(length - 1, -1, -1, dtype=np.float64)
        weights /= weights.sum()
        return values @ weights

    return predict


#: Registry used by the evaluation scripts. Keys become row labels in the
#: results tables, so they are stable identifiers rather than prose.
BASELINE_PREDICTORS: dict[str, Predictor] = {
    "last_interval": last_interval,
    "mean_interval": mean_interval,
    "median_interval": median_interval,
    "moving_average_3": moving_average(3),
    "ema_0.5": exponential_moving_average(0.5),
}


def rule_statistic_over_training(lbas: "np.ndarray",
                                 timestamps: "np.ndarray",
                                 min_history: int,
                                 train_fraction: float) -> "np.ndarray":
    """Values the rule-based classifier would compute during the training period.

    The C++ heuristic classifies on the *running mean* of an LBA's observed
    rewrite intervals. Its threshold must therefore be a percentile of that same
    statistic - not of the next-interval targets the LSTM is trained on.

    The two distributions are not interchangeable: a running mean over an LBA's
    whole history is dominated by its long idle gaps and sits orders of magnitude
    above a typical single interval. Using the model's threshold for the rule
    made it label 2 writes out of 550,987 as HOT, so it was really separating
    blocks by *how much history they had* rather than by temperature.

    Only the earliest `train_fraction` of the write stream is examined, so the
    resulting threshold carries no information from the evaluation period.
    """
    import numpy as np

    lbas = np.asarray(lbas)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    cutoff = int(round(train_fraction * lbas.size))
    if cutoff <= 0:
        return np.empty(0, dtype=np.float64)

    last_write: dict[int, int] = {}
    interval_sum: dict[int, float] = {}
    interval_count: dict[int, int] = {}
    observed: list[float] = []

    for index in range(cutoff):
        lba = int(lbas[index])
        timestamp = int(timestamps[index])
        count = interval_count.get(lba, 0)
        # Classification happens before the write is observed, mirroring the
        # simulator's order of operations.
        if count >= min_history:
            observed.append(interval_sum[lba] / count)
        previous = last_write.get(lba)
        if previous is not None and timestamp >= previous:
            interval_sum[lba] = interval_sum.get(lba, 0.0) + float(timestamp - previous)
            interval_count[lba] = count + 1
        last_write[lba] = timestamp

    return np.asarray(observed, dtype=np.float64)


def rule_threshold_from_training(trace_path: "Path",
                                 min_history: int,
                                 train_fraction: float,
                                 percentile: float,
                                 max_writes: int | None = None) -> float:
    """Hot/cold cutoff for the rule-based control, from its own statistic.

    Reads the write stream of a normalized trace, replays the heuristic's state
    machine over the earliest `train_fraction` of it, and returns the requested
    percentile of the running-mean values it would have classified on.

    Single implementation shared by the experiment runner and the demo: having
    two would let them drift, and it was exactly a mismatched threshold that
    made the heuristic degenerate once already.
    """
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(trace_path, usecols=["timestamp", "lba", "operation"])
    writes = frame[frame["operation"] == "W"]
    if max_writes:
        writes = writes.iloc[:max_writes]
    if writes.empty:
        raise ValueError(f"{trace_path} contains no writes")

    observed = rule_statistic_over_training(
        writes["lba"].to_numpy(), writes["timestamp"].to_numpy(),
        min_history=min_history, train_fraction=train_fraction)
    if observed.size == 0:
        raise ValueError(
            f"{trace_path}: no LBA reached {min_history} rewrite intervals during the "
            "training period, so the rule has no threshold to derive"
        )
    return float(np.percentile(observed, percentile))
