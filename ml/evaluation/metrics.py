"""Regression and hot/cold classification metrics.

All metrics are computed in the original unit (microseconds), not in the
normalized log space the model trains in, so they are directly interpretable and
comparable against the non-learned baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    median_absolute_error: float
    # Errors span orders of magnitude, so a log-space error is reported too:
    # it is the scale the model actually optimises and is far less dominated by
    # a handful of very long intervals.
    mae_log1p: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    actual_hot_fraction: float
    predicted_hot_fraction: float
    roc_auc: float | None
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> RegressionMetrics:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    if actual.size == 0:
        return RegressionMetrics(float("nan"), float("nan"), float("nan"), float("nan"), 0)

    error = predicted - actual
    log_error = (np.log1p(np.clip(predicted, 0.0, None))
                 - np.log1p(np.clip(actual, 0.0, None)))
    return RegressionMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error ** 2))),
        median_absolute_error=float(np.median(np.abs(error))),
        mae_log1p=float(np.mean(np.abs(log_error))),
        n=int(actual.size),
    )


def classification_metrics(actual_labels: np.ndarray,
                           predicted_labels: np.ndarray,
                           scores: np.ndarray | None = None) -> ClassificationMetrics:
    """HOT is the positive class (label 1).

    `scores` is an optional continuous "hotness" score used for ROC-AUC. Pass
    the negated predicted interval, since a *shorter* predicted interval means
    hotter. AUC is omitted when only one class is present, where it is undefined
    rather than zero.
    """
    actual_labels = np.asarray(actual_labels).astype(np.int8)
    predicted_labels = np.asarray(predicted_labels).astype(np.int8)
    if actual_labels.shape != predicted_labels.shape:
        raise ValueError("label arrays must have the same shape")
    n = int(actual_labels.size)
    if n == 0:
        return ClassificationMetrics(float("nan"), float("nan"), float("nan"), float("nan"),
                                     0, 0, 0, 0, float("nan"), float("nan"), None, 0)

    true_positive = int(np.sum((actual_labels == 1) & (predicted_labels == 1)))
    false_positive = int(np.sum((actual_labels == 0) & (predicted_labels == 1)))
    true_negative = int(np.sum((actual_labels == 0) & (predicted_labels == 0)))
    false_negative = int(np.sum((actual_labels == 1) & (predicted_labels == 0)))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    roc_auc: float | None = None
    if scores is not None and len(np.unique(actual_labels)) == 2:
        try:
            from sklearn.metrics import roc_auc_score

            roc_auc = float(roc_auc_score(actual_labels, np.asarray(scores, dtype=np.float64)))
        except (ImportError, ValueError):
            roc_auc = None

    return ClassificationMetrics(
        accuracy=(true_positive + true_negative) / n,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positive,
        false_positives=false_positive,
        true_negatives=true_negative,
        false_negatives=false_negative,
        actual_hot_fraction=float(np.mean(actual_labels == 1)),
        predicted_hot_fraction=float(np.mean(predicted_labels == 1)),
        roc_auc=roc_auc,
        n=n,
    )
