"""Rung 6 - LSTM_ATTN_SMARTGC: LSTM backbone + lightweight additive attention.

Same stacked-LSTM backbone as ``LSTM_SMARTGC`` but the linear head reads a
context vector formed by additive (Bahdanau-style) attention over all
``sequence_length`` LSTM outputs instead of only the last hidden state
(closes gap #7).  ``ml.attention.num_heads`` heads share the additive scoring
mechanism over disjoint slices of the hidden dimension; keep it small - this is
a per-block model, not an LLM.  Whether attention actually helps at this
sequence length is an empirical question the benchmark answers; a negative
result is reported as-is.
"""
from __future__ import annotations

from ml.models.torch_common import TorchIntervalModel, _torch


def _make_attn_module(n_features, hidden_dim, num_layers, dropout, num_heads, attention_enabled=True):
    torch = _torch()
    nn = torch.nn

    heads = max(1, num_heads)
    while hidden_dim % heads != 0 and heads > 1:
        heads -= 1
    head_dim = hidden_dim // heads

    class AdditiveAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Linear(head_dim, head_dim, bias=True)
            self.v = nn.Linear(head_dim, 1, bias=False)

        def forward(self, h):  # h: [B, T, head_dim]
            scores = self.v(torch.tanh(self.W(h))).squeeze(-1)  # [B, T]
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
            return (weights * h).sum(dim=1)  # [B, head_dim]

    class LSTMAttnRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = heads
            self.head_dim = head_dim
            self.attention_enabled = attention_enabled
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            if attention_enabled:
                self.attn = nn.ModuleList([AdditiveAttention() for _ in range(heads)])
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)  # [B, T, H]
            if not self.attention_enabled:
                return self.head(out[:, -1, :])  # last-hidden-state pooling, like LSTM_SMARTGC
            ctx = []
            for i, attn in enumerate(self.attn):
                sl = out[:, :, i * self.head_dim:(i + 1) * self.head_dim]
                ctx.append(attn(sl))
            return self.head(torch.cat(ctx, dim=-1))

    return LSTMAttnRegressor()


class LSTMAttentionModel(TorchIntervalModel):
    name = "LSTM_ATTN_SMARTGC"

    def _build_module(self):
        cfg = self.mlcfg
        attn_cfg = cfg.get("attention") or {}
        return _make_attn_module(
            n_features=self.n_features,
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            num_heads=int(attn_cfg.get("num_heads", 2)),
            attention_enabled=bool(attn_cfg.get("enabled", True)),
        )
