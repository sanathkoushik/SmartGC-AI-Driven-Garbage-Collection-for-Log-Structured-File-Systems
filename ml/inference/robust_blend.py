"""Phase 8 - learning-augmented robustness blend for HYBRID_ROBUST_SMARTGC.

Gap addressed
-------------
The Section-4 ladder found that on the real UMass SPC Financial1 trace the
zero-training ``RULE_BASED`` heuristic beats every learned rung, including
``LSTM_ATTN_SMARTGC`` -- and that the existing binary confidence gate
(``ml/inference/confidence_gate.py``) still trusts the model on ~69% of real
events yet is wrong on much of that trusted majority (docs/progress.md).  A
bigger model does not fix that; the gate itself is the blunt instrument.

This module instantiates the *consistency-robustness* framing of
learning-augmented online algorithms -- concretely, the SSD-management
setting of Lange, Naor & Yadgar, "Optimal SSD Management with Predictions"
(ACM SIGMETRICS / Proc. ACM Meas. Anal. Comput. Syst. 9(2), June 2025) -- as a
simple, empirically-tunable inference-time policy: the trained sequence
model's own prediction is trusted in proportion to its calibrated
(MC-dropout) confidence, but that trust is capped by a robustness parameter
``lam`` in [0, 1] so the policy can never lean on a possibly-wrong model more
than ``(1 - lam)`` of the way, regardless of confidence. ``lam = 0`` recovers
pure confidence-weighted trust in the model; ``lam = 1`` ignores the model
entirely and always uses the ``RULE_BASED`` interval -- the rung the ladder
already found to be the strongest zero-training baseline on real OLTP data.
Intermediate ``lam`` interpolates continuously, generalising the hard
threshold in ``ConfidenceGate`` into the kind of tunable consistency/
robustness tradeoff curve that paper's framing describes.

This is a heuristic instantiation of that framing for an empirical ladder
comparison, not a reproduction of the paper's formal worst-case algorithm or
its competitive-ratio guarantee.
"""
from __future__ import annotations

import numpy as np


def trust_weight(confidence: np.ndarray, lam: float) -> np.ndarray:
    """Weight given to the model's own prediction, in [0, 1 - lam]."""
    lam = float(np.clip(lam, 0.0, 1.0))
    return np.clip(confidence, 0.0, 1.0) * (1.0 - lam)


def blend_interval(model_interval: np.ndarray, rb_interval: np.ndarray,
                    confidence: np.ndarray, lam: float):
    """Elementwise confidence-weighted, robustness-capped blend.

    Returns ``(blended_interval, trust_weight)``. At ``lam=1`` the result is
    exactly ``rb_interval`` regardless of confidence; at ``lam=0`` it reduces
    to a continuous confidence-weighted average (no cap).
    """
    w = trust_weight(confidence, lam)
    blended = w * np.asarray(model_interval, dtype=np.float64) + \
        (1.0 - w) * np.asarray(rb_interval, dtype=np.float64)
    return blended, w
