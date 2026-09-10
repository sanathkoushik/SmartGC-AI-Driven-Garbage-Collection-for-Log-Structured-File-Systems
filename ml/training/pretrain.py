"""Pretrain the general rewrite-interval LSTM on several MSR workloads.

    python -m ml.training.pretrain --traces msr_hm_0 msr_mds_0 msr_prn_0 ...

Each trace is split chronologically first and only the matching splits are
pooled, so no workload's future reaches another workload's training data. The
held-out target workloads must not appear in ``--traces``; the caller
(``experiments/run_all.py``) takes them from ``ml.dataset_selection``.

Writes ``models/pretrained/msr_pretrained.pt`` plus a ``.json`` recording the
traces used, the hyperparameters, the seed, the normalization parameters, the
sample counts, the best epoch, the training duration and the validation MAE and
RMSE in microseconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ml.config import REPO_ROOT, load_config
from ml.evaluation.metrics import regression_metrics
from ml.models.lstm import LstmHyperparameters, build_model
from ml.preprocessing.sequences import IntervalScaler
from ml.training.common import (
    DEFAULT_MAX_SEQUENCES_PER_TRACE,
    MODELS_DIR,
    SplitPool,
    available_traces,
    build_split_pool,
    derive_hot_threshold,
    predict_intervals,
    resolve_device,
    save_checkpoint,
    set_deterministic_seed,
    train_model,
)

DEFAULT_OUTPUT = MODELS_DIR / "pretrained" / "msr_pretrained.pt"


def pretrain(trace_ids: Sequence[str],
             config_path: Path | None = None,
             output_path: Path = DEFAULT_OUTPUT,
             hyperparameters: LstmHyperparameters | None = None,
             learning_rate: float | None = None,
             max_sequences: int | None = DEFAULT_MAX_SEQUENCES_PER_TRACE,
             seed: int | None = None,
             device_preference: str = "auto",
             verbose: bool = True) -> dict[str, Any]:
    """Train the general model and write its checkpoint plus metadata."""
    config = load_config(config_path)
    ml_config = config.ml
    seed = config.random_seed if seed is None else seed
    hyperparameters = hyperparameters or LstmHyperparameters(
        sequence_length=ml_config.sequence_length,
        hidden_dim=ml_config.hidden_dim,
        num_layers=ml_config.num_layers,
        dropout=ml_config.dropout,
    )

    if not trace_ids:
        raise ValueError("no pretraining traces given")

    set_deterministic_seed(seed)
    device = resolve_device(device_preference)

    if verbose:
        print(f"Pretraining on {len(trace_ids)} trace(s): {', '.join(trace_ids)}")
        print(f"  sequence_length={hyperparameters.sequence_length} "
              f"hidden_dim={hyperparameters.hidden_dim} "
              f"num_layers={hyperparameters.num_layers} device={device}")

    pool = build_split_pool(trace_ids, ml_config, max_sequences=max_sequences,
                            sequence_length=hyperparameters.sequence_length)
    sizes = pool.sizes
    if verbose:
        print(f"  samples: train {sizes['train']:,}  val {sizes['val']:,}  test {sizes['test']:,}")
    if sizes["train"] == 0 or sizes["val"] == 0:
        raise ValueError(
            f"pretraining pool is unusable (train={sizes['train']}, val={sizes['val']}). "
            "Check that the traces were normalized and contain rewrite intervals."
        )

    # Fitted on training intervals ONLY. See docs/methodology.md.
    scaler = IntervalScaler.fit(pool.train["inputs"], pool.train["targets"])
    hot_threshold = derive_hot_threshold(pool, ml_config)

    model = build_model(hyperparameters, seed=seed)
    result = train_model(model, pool, scaler, ml_config, seed=seed, device=device,
                         learning_rate=learning_rate,
                         progress_label="pretrain" if verbose else "")

    validation_predictions = predict_intervals(model, pool.val["inputs"], scaler, device)
    validation = regression_metrics(pool.val["targets"], validation_predictions)

    metadata: dict[str, Any] = {
        "stage": "pretrain",
        "traces": list(trace_ids),
        "seed": seed,
        "device": str(device),
        "optimizer": "AdamW",
        "learning_rate": learning_rate if learning_rate is not None else ml_config.learning_rate,
        "batch_size": ml_config.batch_size,
        "max_epochs": ml_config.epochs,
        "loss_function": ml_config.loss_function,
        "grad_clip_norm": ml_config.grad_clip_norm,
        "early_stopping_patience": ml_config.early_stopping_patience,
        "max_sequences_per_trace": max_sequences,
        "split_ratios": {"train": ml_config.train_split, "val": ml_config.val_split,
                         "test": ml_config.test_split},
        "samples": sizes,
        "per_trace_samples": pool.per_trace,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "epochs_run": result.epochs_run,
        "early_stopped": result.early_stopped,
        "training_seconds": result.training_seconds,
        "training_history": result.history,
        "validation_metrics_microseconds": validation.to_dict(),
        "hot_threshold_microseconds": hot_threshold,
        "hot_percentile_cutoff": ml_config.hot_percentile_cutoff,
        "config_source": str(config.source_path),
    }

    json_path = save_checkpoint(model, scaler, metadata, output_path)
    if verbose:
        print(f"  best epoch {result.best_epoch} (val loss {result.best_val_loss:.6f}), "
              f"{result.training_seconds:.1f}s")
        print(f"  validation MAE  {validation.mae:,.1f} us   RMSE {validation.rmse:,.1f} us")
        print(f"  hot/cold threshold (train p{ml_config.hot_percentile_cutoff:g}): "
              f"{hot_threshold:,.1f} us")
        print(f"  saved {output_path.relative_to(REPO_ROOT)} and {json_path.name}")
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", nargs="*", default=None,
                        help="normalized trace ids; defaults to every normalized MSR trace")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-sequences", type=int, default=DEFAULT_MAX_SEQUENCES_PER_TRACE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    traces = args.traces if args.traces else available_traces("msr")
    if not traces:
        print("No normalized MSR traces found. Run the preprocessing step first.")
        return 2

    hyperparameters = LstmHyperparameters(
        sequence_length=args.sequence_length or config.ml.sequence_length,
        hidden_dim=args.hidden_dim or config.ml.hidden_dim,
        num_layers=args.num_layers or config.ml.num_layers,
        dropout=config.ml.dropout if args.dropout is None else args.dropout,
    )
    pretrain(traces, config_path=args.config, output_path=args.output,
             hyperparameters=hyperparameters, learning_rate=args.learning_rate,
             max_sequences=args.max_sequences, seed=args.seed,
             device_preference=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
