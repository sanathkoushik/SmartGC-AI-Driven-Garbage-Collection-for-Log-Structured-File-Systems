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
REGISTRY: dict[str, type[BaseIntervalModel]] = {
    "RULE_BASED": RuleBasedModel,
    "SUP_LIKE": SupLikeModel,
    "STAT_ML": StatMLModel,
    "LSTM_SMARTGC": LSTMModel,
    "LSTM_ATTN_SMARTGC": LSTMAttentionModel,
}

LADDER_ORDER = ["MIXED", "RULE_BASED", "SUP_LIKE", "STAT_ML",
                "LSTM_SMARTGC", "LSTM_ATTN_SMARTGC"]


def build(name: str, scaler: dict, config: dict) -> BaseIntervalModel:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](scaler, config)
