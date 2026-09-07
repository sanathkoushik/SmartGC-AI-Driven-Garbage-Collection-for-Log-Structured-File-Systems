"""Rung 5 - LSTM_SMARTGC: the vanilla 2-layer LSTM interval regressor.

This is the originally-planned SmartGC model: a lightweight stacked LSTM over
the multivariate ``sequence_length`` window, last hidden state -> linear head,
trained on the log1p-scaled next-rewrite-interval target.
"""
from __future__ import annotations

from ml.models.torch_common import TorchIntervalModel, _torch


def _make_lstm_module(n_features: int, hidden_dim: int, num_layers: int, dropout: float):
    torch = _torch()
    nn = torch.nn

    class LSTMRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            out, (h, _) = self.lstm(x)
            return self.head(h[-1])

    return LSTMRegressor()


class LSTMModel(TorchIntervalModel):
    name = "LSTM_SMARTGC"

    def _build_module(self):
        cfg = self.mlcfg
        return _make_lstm_module(
            n_features=self.n_features,
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
