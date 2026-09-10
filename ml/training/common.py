"""Dataset assembly, deterministic seeding, the training loop and checkpoints.

The split policy implemented here is the one the whole study depends on:

* Every trace is split **chronologically** into train / validation / test.
* Pooling across traces happens *after* splitting, so a trace's future never
  lands in another trace's training pool.
* The input scaler is fitted on the pooled **training** intervals only.
* The hot/cold threshold is derived from the pooled **training** targets only.
* Test data is touched exactly once, at final evaluation.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.config import REPO_ROOT, MlConfig, environment_metadata
from ml.models.lstm import LstmHyperparameters, RewriteIntervalLSTM, build_model, make_loss
from ml.preprocessing.sequences import (
    IntervalScaler,
    build_sequences,
    hot_threshold_from_training,
    order_chronologically,
    rewrite_intervals,
    split_samples,
)

NORMALIZED_DIR = REPO_ROOT / "data" / "processed" / "normalized"
SEQUENCE_CACHE_DIR = REPO_ROOT / "data" / "processed" / "sequences"
MODELS_DIR = REPO_ROOT / "models"

#: Chronological prefix kept per trace. Full MSR volumes yield tens of millions
#: of sequences, which is neither necessary for a small LSTM nor tractable on a
#: workstation. A prefix (never a random sample) preserves the causal ordering
#: the split relies on.
DEFAULT_MAX_SEQUENCES_PER_TRACE = 400_000


def set_deterministic_seed(seed: int) -> None:
    """Seed every generator the pipeline uses and disable nondeterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)   # LSTM has no deterministic CUDA kernel
    torch.backends.cudnn.benchmark = False


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def normalized_trace_path(trace_id: str) -> Path:
    return NORMALIZED_DIR / f"{trace_id}.csv"


def available_traces(dataset: str | None = None) -> list[str]:
    """Trace ids that have been normalized, optionally filtered by dataset prefix."""
    if not NORMALIZED_DIR.is_dir():
        return []
    names = sorted(path.stem for path in NORMALIZED_DIR.glob("*.csv"))
    if dataset:
        names = [name for name in names if name.startswith(f"{dataset}_")]
    return names


def load_trace_samples(trace_id: str,
                       sequence_length: int,
                       max_sequences: int | None = DEFAULT_MAX_SEQUENCES_PER_TRACE,
                       use_cache: bool = True) -> dict[str, np.ndarray]:
    """Build ``(window -> next interval)`` samples for one normalized trace.

    Cached as an .npz keyed by trace and sequence length, because rebuilding
    from a multi-million-row CSV dominates the runtime of the experiments.
    """
    cache_path = SEQUENCE_CACHE_DIR / f"{trace_id}__L{sequence_length}.npz"
    if use_cache and cache_path.is_file():
        with np.load(cache_path) as data:
            samples = {key: data[key] for key in ("inputs", "targets", "timestamp", "lba")}
    else:
        path = normalized_trace_path(trace_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"Normalized trace not found: {path}. "
                "Run: python -m ml.preprocessing.normalize --discover"
            )
        frame = pd.read_csv(path, usecols=["timestamp", "lba", "operation"])
        writes = frame[frame["operation"] == "W"]
        intervals = rewrite_intervals(writes["lba"].to_numpy(),
                                      writes["timestamp"].to_numpy())
        samples = build_sequences(intervals, sequence_length)
        samples = order_chronologically(samples)
        samples["inputs"] = samples["inputs"].astype(np.float32)
        samples["targets"] = samples["targets"].astype(np.float32)
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **samples)

    if max_sequences is not None and samples["targets"].size > max_sequences:
        # Chronological prefix, never a random sample.
        samples = {key: value[:max_sequences] for key, value in samples.items()}
    return samples


@dataclass
class SplitPool:
    """Pooled train/validation/test samples plus their per-trace provenance."""

    train: dict[str, np.ndarray]
    val: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    per_trace: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": int(self.train["targets"].size),
                "val": int(self.val["targets"].size),
                "test": int(self.test["targets"].size)}


def _concat(parts: Sequence[dict[str, np.ndarray]], sequence_length: int) -> dict[str, np.ndarray]:
    parts = [part for part in parts if part["targets"].size]
    if not parts:
        return {"inputs": np.empty((0, sequence_length), dtype=np.float32),
                "targets": np.empty(0, dtype=np.float32),
                "timestamp": np.empty(0, dtype=np.int64),
                "lba": np.empty(0, dtype=np.int64)}
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def build_split_pool(trace_ids: Iterable[str],
                     config: MlConfig,
                     max_sequences: int | None = DEFAULT_MAX_SEQUENCES_PER_TRACE,
                     sequence_length: int | None = None) -> SplitPool:
    """Split each trace chronologically, then pool the matching splits.

    Splitting before pooling is what keeps the experiment honest across
    workloads: pooling first and then cutting would place one trace's late
    behaviour in another trace's training set.
    """
    length = sequence_length or config.sequence_length
    trains, vals, tests = [], [], []
    per_trace: dict[str, dict[str, int]] = {}

    for trace_id in trace_ids:
        samples = load_trace_samples(trace_id, length, max_sequences=max_sequences)
        if samples["targets"].size == 0:
            per_trace[trace_id] = {"train": 0, "val": 0, "test": 0, "total": 0}
            continue
        train, val, test, boundaries = split_samples(
            samples, config.train_split, config.val_split)
        trains.append(train)
        vals.append(val)
        tests.append(test)
        per_trace[trace_id] = {"train": boundaries.train_size, "val": boundaries.val_size,
                               "test": boundaries.test_size, "total": boundaries.total}

    return SplitPool(train=_concat(trains, length), val=_concat(vals, length),
                     test=_concat(tests, length), per_trace=per_trace)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    best_epoch: int
    best_val_loss: float
    epochs_run: int
    training_seconds: float
    history: list[dict[str, float]]
    early_stopped: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _make_loader(samples: dict[str, np.ndarray],
                 scaler: IntervalScaler,
                 batch_size: int,
                 shuffle: bool,
                 generator: torch.Generator | None = None) -> DataLoader:
    inputs = torch.from_numpy(scaler.transform(samples["inputs"]).astype(np.float32))
    targets = torch.from_numpy(scaler.transform(samples["targets"]).astype(np.float32))
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size,
                      shuffle=shuffle, generator=generator, drop_last=False)


def train_model(model: RewriteIntervalLSTM,
                pool: SplitPool,
                scaler: IntervalScaler,
                config: MlConfig,
                seed: int,
                device: torch.device,
                learning_rate: float | None = None,
                epochs: int | None = None,
                progress_label: str = "") -> TrainingResult:
    """Train with early stopping, restoring the best-validation weights.

    Shuffling happens *within* the training split only. It reorders samples that
    are already all earlier than every validation sample, so it cannot leak;
    what would leak is shuffling before the split, which the pipeline never does.
    """
    if pool.train["targets"].size == 0:
        raise ValueError("training split is empty")
    if pool.val["targets"].size == 0:
        raise ValueError("validation split is empty; cannot select a checkpoint without it")

    set_deterministic_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = model.to(device)

    lr = learning_rate if learning_rate is not None else config.learning_rate
    max_epochs = epochs if epochs is not None else config.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = make_loss(config.loss_function)

    train_loader = _make_loader(pool.train, scaler, config.batch_size, True, generator)
    val_loader = _make_loader(pool.val, scaler, config.batch_size, False)

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    early_stopped = False
    started = time.monotonic()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            train_loss_sum += float(loss.item()) * batch_targets.size(0)
            train_count += batch_targets.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                predictions = model(batch_inputs)
                loss = criterion(predictions, batch_targets)
                val_loss_sum += float(loss.item()) * batch_targets.size(0)
                val_count += batch_targets.size(0)

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < best_val - 1e-9
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if progress_label:
            marker = " *" if improved else ""
            print(f"  [{progress_label}] epoch {epoch:3d}/{max_epochs}  "
                  f"train {train_loss:.5f}  val {val_loss:.5f}{marker}", flush=True)

        if epochs_without_improvement >= config.early_stopping_patience:
            early_stopped = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingResult(best_epoch=best_epoch, best_val_loss=best_val,
                          epochs_run=len(history),
                          training_seconds=time.monotonic() - started,
                          history=history, early_stopped=early_stopped)


@torch.no_grad()
def predict_intervals(model: RewriteIntervalLSTM,
                      inputs: np.ndarray,
                      scaler: IntervalScaler,
                      device: torch.device,
                      batch_size: int = 4096) -> np.ndarray:
    """Predict next rewrite intervals in microseconds."""
    model = model.to(device).eval()
    if inputs.size == 0:
        return np.empty(0, dtype=np.float64)
    scaled = torch.from_numpy(scaler.transform(inputs).astype(np.float32))
    outputs: list[np.ndarray] = []
    for start in range(0, scaled.shape[0], batch_size):
        batch = scaled[start:start + batch_size].to(device)
        outputs.append(model(batch).cpu().numpy())
    return scaler.inverse_transform(np.concatenate(outputs))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def repo_relative(path: Path) -> str:
    """Path relative to the repository root, or absolute if it lies outside it."""
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def save_checkpoint(model: RewriteIntervalLSTM,
                    scaler: IntervalScaler,
                    metadata: dict[str, Any],
                    checkpoint_path: Path) -> Path:
    """Write ``<name>.pt`` and the matching ``<name>.json`` metadata."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "hyperparameters": model.hyperparameters.to_dict(),
        "scaler": scaler.to_dict(),
    }, checkpoint_path)

    metadata = dict(metadata)
    metadata["checkpoint"] = repo_relative(checkpoint_path)
    metadata["hyperparameters"] = model.hyperparameters.to_dict()
    metadata["normalization"] = scaler.to_dict()
    metadata["parameter_count"] = model.parameter_count()
    metadata.setdefault("environment", environment_metadata())

    json_path = checkpoint_path.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
                         encoding="utf-8")
    return json_path


def load_checkpoint(checkpoint_path: Path,
                    device: torch.device | None = None
                    ) -> tuple[RewriteIntervalLSTM, IntervalScaler, dict[str, Any]]:
    """Restore a model, its scaler and its metadata."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
    hyperparameters = LstmHyperparameters.from_dict(payload["hyperparameters"])
    model = build_model(hyperparameters)
    model.load_state_dict(payload["state_dict"])
    scaler = IntervalScaler.from_dict(payload["scaler"])

    json_path = checkpoint_path.with_suffix(".json")
    metadata = json.loads(json_path.read_text(encoding="utf-8")) if json_path.is_file() else {}
    return model, scaler, metadata


def derive_hot_threshold(pool: SplitPool, config: MlConfig) -> float:
    """Hot/cold cutoff from the pooled TRAINING targets only."""
    return hot_threshold_from_training(pool.train["targets"], config.hot_percentile_cutoff)
