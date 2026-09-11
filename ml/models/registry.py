"""Name -> model-class registry for the Phase 4a baseline ladder."""
from __future__ import annotations

from ml.models.base import BaseIntervalModel
from ml.models.rule_based import RuleBasedModel
from ml.models.sup_like import SupLikeModel
from ml.models.stat_ml import StatMLModel
from ml.models.lstm import LSTMModel
from ml.models.lstm_attention import LSTMAttentionModel

# The two C++-only rungs (MIXED, and the placement half of SUP_LIKE) do not need
# a Python model, but SUP_LIKE still gets one so it can emit predictions.csv.
#
# HYBRID_ROBUST_SMARTGC (Phase 8) intentionally maps to the *same*
# LSTMAttentionModel class as LSTM_ATTN_SMARTGC: it reuses that rung's trained
# checkpoint verbatim (BaseIntervalModel.load() keys off the class's fixed
# ``name`` attribute, not the registry key, so pointing it at a
# ``LSTM_ATTN_SMARTGC_<dataset>`` artefact dir just works). Only the Phase 4b
# inference-time confidence handling differs -- see ml/inference/export.py and
# ml/inference/robust_blend.py.
REGISTRY: dict[str, type[BaseIntervalModel]] = {
    "RULE_BASED": RuleBasedModel,
    "SUP_LIKE": SupLikeModel,
    "STAT_ML": StatMLModel,
    "LSTM_SMARTGC": LSTMModel,
    "LSTM_ATTN_SMARTGC": LSTMAttentionModel,
    "HYBRID_ROBUST_SMARTGC": LSTMAttentionModel,
}

# The original 6-rung falsifiable baseline ladder (docs/progress.md Phase 4a).
# Kept as-is so existing reports/plots that key off it are unaffected.
LADDER_ORDER = ["MIXED", "RULE_BASED", "SUP_LIKE", "STAT_ML",
                "LSTM_SMARTGC", "LSTM_ATTN_SMARTGC"]

# Phase 8 addition: the learning-augmented hybrid rung, evaluated alongside
# (not instead of) the original ladder.
EXTENDED_LADDER_ORDER = LADDER_ORDER + ["HYBRID_ROBUST_SMARTGC"]


def build(name: str, scaler: dict, config: dict) -> BaseIntervalModel:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](scaler, config)
