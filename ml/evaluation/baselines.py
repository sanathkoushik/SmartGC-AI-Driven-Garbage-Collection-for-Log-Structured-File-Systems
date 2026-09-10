"""Non-learned predictors of the next rewrite interval.

These exist so the LSTM has to earn its place. Each takes the same input window
the model sees -- the last ``L`` rewrite intervals of one LBA -- and returns a
prediction in the same unit, using no training and no parameters.

If a baseline matches the LSTM, that is the finding, and it is reported.
"""

from __future__ import annotations

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
