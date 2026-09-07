"""Phase 4b - predictive-confidence gate.

For the LSTM rungs, predictive confidence is estimated by Monte-Carlo dropout:
``k`` stochastic forward passes give a per-sample mean and spread, and the
coefficient of variation is squashed to a ``[0, 1]`` confidence.  For the
non-deep rungs a flat per-model confidence is used (they have no cheap
uncertainty signal).

When ``confidence < confidence_fallback_threshold`` the export layer emits the
RULE_BASED decision for that block instead of the model's, and the run's
fallback rate is reported (mirrors the C++ ``--confidence-threshold`` gate).
"""
from __future__ import annotations

import numpy as np

_FLAT_CONFIDENCE = {
    "RULE_BASED": 0.50,
    "SUP_LIKE": 1.00,
    "STAT_ML": 0.70,
}


def cv_to_confidence(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    cv = std / (np.abs(mean) + 1e-6)
    return np.clip(np.exp(-cv), 0.0, 1.0)


def mc_dropout_confidence(model, X: np.ndarray, k: int = 3):
    """Return (mean_interval_raw, confidence) using k MC-dropout passes.
    Falls back to a single deterministic pass + flat confidence if the model is
    not a torch model."""
    module = getattr(model, "module", None)
    if module is None:
        pred = model.predict_interval(X)
        flat = _FLAT_CONFIDENCE.get(getattr(model, "name", ""), 0.6)
        return pred, np.full(len(X), flat, dtype=np.float64)

    import torch

    was_training = module.training
    module.train()  # enable dropout
    passes = []
    with torch.no_grad():
        for _ in range(max(2, k)):
            outs = []
            for i in range(0, len(X), 512):
                xb = torch.tensor(X[i:i + 512], dtype=torch.float32)
                outs.append(module(xb).squeeze(-1).cpu().numpy())
            y_scaled = np.concatenate(outs) if outs else np.zeros((0,))
            passes.append(model._from_scaled_target(y_scaled))
    module.train(was_training)

    stack = np.stack(passes, axis=0) if passes else np.zeros((1, len(X)))
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    return mean, cv_to_confidence(mean, std)


class ConfidenceGate:
    def __init__(self, threshold: float):
        self.threshold = float(threshold)
        self.total = 0
        self.fallbacks = 0

    def decide(self, confidence: float) -> bool:
        """True => trust the model; False => fall back to RULE_BASED."""
        self.total += 1
        trust = confidence >= self.threshold
        if not trust:
            self.fallbacks += 1
        return trust

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / self.total if self.total else 0.0
