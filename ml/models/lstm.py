"""Lightweight LSTM regressor over rewrite-interval sequences.

Architecture::

    (batch, sequence_length, 1)   normalized log1p rewrite intervals
        -> LSTM(hidden_dim, num_layers, dropout)
        -> final hidden state of the last layer
        -> Linear(hidden_dim, 1)
        -> predicted next interval, in normalized log1p space

The model is deliberately small: it must train on CPU in minutes, because the
research question is whether *any* sequence model beats a threshold heuristic,
not whether a large one can be fitted.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class LstmHyperparameters:
    sequence_length: int = 10
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LstmHyperparameters":
        return cls(
            sequence_length=int(data["sequence_length"]),
            hidden_dim=int(data["hidden_dim"]),
            num_layers=int(data["num_layers"]),
            dropout=float(data["dropout"]),
        )


class RewriteIntervalLSTM(nn.Module):
    """Predicts the next rewrite interval from the previous ones."""

    def __init__(self, hyperparameters: LstmHyperparameters) -> None:
        super().__init__()
        self.hyperparameters = hyperparameters
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hyperparameters.hidden_dim,
            num_layers=hyperparameters.num_layers,
            batch_first=True,
            # PyTorch applies dropout between stacked layers only, so it is
            # meaningless (and warns) with a single layer.
            dropout=hyperparameters.dropout if hyperparameters.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hyperparameters.hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: (batch, sequence_length) or (batch, sequence_length, 1) -> (batch,)."""
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(-1)
        output, _ = self.lstm(inputs)
        last_step = output[:, -1, :]
        return self.head(last_step).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(hyperparameters: LstmHyperparameters,
                seed: int | None = None) -> RewriteIntervalLSTM:
    """Construct the model, optionally seeding initialisation for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
    return RewriteIntervalLSTM(hyperparameters)


def make_loss(name: str) -> nn.Module:
    """Regression loss selected by ``ml.loss_function`` in config.yaml."""
    losses = {"SmoothL1": nn.SmoothL1Loss, "L1": nn.L1Loss, "MSE": nn.MSELoss}
    if name not in losses:
        raise ValueError(f"Unknown loss '{name}'; expected one of {sorted(losses)}")
    return losses[name]()
