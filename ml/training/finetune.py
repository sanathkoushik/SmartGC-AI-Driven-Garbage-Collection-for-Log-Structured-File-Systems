"""Adapt a model to one held-out target workload, or train one from scratch.

    python -m ml.training.finetune --trace msr_prxy_0 \
        --checkpoint models/pretrained/msr_pretrained.pt

    python -m ml.training.finetune --trace msr_prxy_0 --from-scratch

The three model variants this produces are the core comparison of the study:

  scratch               randomly initialised, trained on the target only
  pretrained            the general model, evaluated on the target unchanged
  pretrained_finetuned  the general model, adapted on the target's early data

Only the target's **early chronological training split** is ever used for
adaptation; the later test split is untouched until final evaluation.

Scaler policy
-------------
``scratch`` fits its own scaler on its own training split. ``pretrained`` and
``pretrained_finetuned`` keep the scaler that was fitted during pretraining, so
that "pretrained" and "pretrained + fine-tuned" differ *only* in whether the
weights were adapted. ``--refit-scaler`` overrides this and is reported in the
metadata when used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ml.config import REPO_ROOT, load_config
from ml.evaluation.metrics import regression_metrics
from ml.models.lstm import LstmHyperparameters, build_model
from ml.preprocessing.sequences import IntervalScaler, chronological_prefix
from ml.training.common import (
    repo_relative,
    DEFAULT_MAX_SEQUENCES_PER_TRACE,
    MODELS_DIR,
    SplitPool,
    build_split_pool,
    derive_hot_threshold,
    load_checkpoint,
    predict_intervals,
    resolve_device,
    save_checkpoint,
    set_deterministic_seed,
    train_model,
)

MODEL_TYPES = ("scratch", "pretrained", "pretrained_finetuned")


def default_output_path(model_type: str, trace_id: str) -> Path:
    return MODELS_DIR / "finetuned" / f"{model_type}__{trace_id}.pt"


def adapt_to_target(trace_id: str,
                    model_type: str,
                    checkpoint_path: Path | None = None,
                    config_path: Path | None = None,
                    output_path: Path | None = None,
                    finetune_fraction: float = 1.0,
                    hyperparameters: LstmHyperparameters | None = None,
                    learning_rate: float | None = None,
                    epochs: int | None = None,
                    max_sequences: int | None = DEFAULT_MAX_SEQUENCES_PER_TRACE,
                    refit_scaler: bool = False,
                    seed: int | None = None,
                    device_preference: str = "auto",
                    verbose: bool = True) -> dict[str, Any]:
    """Produce one target-workload model variant and evaluate it on held-out data."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}")
    if model_type != "scratch" and checkpoint_path is None:
        raise ValueError(f"model_type '{model_type}' requires --checkpoint")

    config = load_config(config_path)
    ml_config = config.ml
    seed = config.random_seed if seed is None else seed
    set_deterministic_seed(seed)
    device = resolve_device(device_preference)

    pretrained_metadata: dict[str, Any] = {}
    if checkpoint_path is not None:
        base_model, base_scaler, pretrained_metadata = load_checkpoint(checkpoint_path, device)
        hyperparameters = base_model.hyperparameters
        pretrain_traces = set(pretrained_metadata.get("traces", []))
        if trace_id in pretrain_traces:
            raise ValueError(
                f"leakage: target '{trace_id}' was used to pretrain {checkpoint_path.name}. "
                "The target workload must be held out of pretraining."
            )
    else:
        base_model, base_scaler = None, None
        # The scratch model must use the *selected* architecture, not the
        # config.yaml defaults. Training it at a different sequence length than
        # the pretrained variants would confound the comparison twice over: a
        # different model AND a different number of sequences, hence a different
        # chronological test split.
        hyperparameters = hyperparameters or LstmHyperparameters(
            sequence_length=ml_config.sequence_length,
            hidden_dim=ml_config.hidden_dim,
            num_layers=ml_config.num_layers,
            dropout=ml_config.dropout,
        )

    pool = build_split_pool([trace_id], ml_config, max_sequences=max_sequences,
                            sequence_length=hyperparameters.sequence_length)
    sizes = pool.sizes
    if verbose:
        print(f"\n[{model_type}] target {trace_id}: "
              f"train {sizes['train']:,}  val {sizes['val']:,}  test {sizes['test']:,}")
    if sizes["test"] == 0:
        raise ValueError(f"target '{trace_id}' has no test split; cannot evaluate")

    # Adaptation may use only an early prefix of the already-early training split.
    train_split = pool.train
    if finetune_fraction < 1.0:
        train_split = chronological_prefix(train_split, finetune_fraction)
    adaptation_pool = SplitPool(train=train_split, val=pool.val, test=pool.test,
                               per_trace=pool.per_trace)

    if model_type == "scratch" or refit_scaler:
        if adaptation_pool.train["targets"].size == 0:
            raise ValueError("target training split is empty; cannot fit a scaler")
        scaler = IntervalScaler.fit(adaptation_pool.train["inputs"],
                                    adaptation_pool.train["targets"])
    else:
        assert base_scaler is not None
        scaler = base_scaler

    training_result = None
    if model_type == "scratch":
        model = build_model(hyperparameters, seed=seed)
        training_result = train_model(model, adaptation_pool, scaler, ml_config, seed=seed,
                                      device=device, learning_rate=learning_rate,
                                      epochs=epochs,
                                      progress_label=f"scratch:{trace_id}" if verbose else "")
    elif model_type == "pretrained":
        # Applied unchanged: this is the ablation that isolates the value of
        # adaptation from the value of pretraining.
        assert base_model is not None
        model = base_model
    else:
        assert base_model is not None
        model = base_model
        training_result = train_model(model, adaptation_pool, scaler, ml_config, seed=seed,
                                      device=device,
                                      learning_rate=learning_rate if learning_rate is not None
                                      else ml_config.learning_rate / 10.0,
                                      epochs=epochs,
                                      progress_label=f"finetune:{trace_id}" if verbose else "")

    # Held-out evaluation. This is the only place test targets are read.
    test_predictions = predict_intervals(model, pool.test["inputs"], scaler, device)
    test_metrics = regression_metrics(pool.test["targets"], test_predictions)
    val_predictions = predict_intervals(model, pool.val["inputs"], scaler, device)
    val_metrics = regression_metrics(pool.val["targets"], val_predictions)

    # Threshold comes from the target's own training split, never from test.
    hot_threshold = derive_hot_threshold(adaptation_pool, ml_config)

    output_path = output_path or default_output_path(model_type, trace_id)
    metadata: dict[str, Any] = {
        "stage": model_type,
        "target_trace": trace_id,
        "source_checkpoint": repo_relative(checkpoint_path) if checkpoint_path else None,
        "pretrain_traces": pretrained_metadata.get("traces", []),
        "seed": seed,
        "device": str(device),
        "optimizer": "AdamW",
        "learning_rate": (learning_rate if learning_rate is not None else
                          (ml_config.learning_rate / 10.0 if model_type == "pretrained_finetuned"
                           else ml_config.learning_rate)),
        "batch_size": ml_config.batch_size,
        "max_epochs": epochs if epochs is not None else ml_config.epochs,
        "loss_function": ml_config.loss_function,
        "finetune_fraction_of_train_split": finetune_fraction,
        "refit_scaler": refit_scaler,
        "max_sequences_per_trace": max_sequences,
        "split_ratios": {"train": ml_config.train_split, "val": ml_config.val_split,
                         "test": ml_config.test_split},
        "samples": {**sizes, "used_for_adaptation": int(adaptation_pool.train["targets"].size)},
        "test_metrics_microseconds": test_metrics.to_dict(),
        "validation_metrics_microseconds": val_metrics.to_dict(),
        "hot_threshold_microseconds": hot_threshold,
        "hot_percentile_cutoff": ml_config.hot_percentile_cutoff,
        "config_source": str(config.source_path),
    }
    if training_result is not None:
        metadata.update({
            "best_epoch": training_result.best_epoch,
            "best_val_loss": training_result.best_val_loss,
            "epochs_run": training_result.epochs_run,
            "early_stopped": training_result.early_stopped,
            "training_seconds": training_result.training_seconds,
            "training_history": training_result.history,
        })
    else:
        metadata["training_seconds"] = 0.0
        metadata["note"] = "applied without adaptation to the target workload"

    save_checkpoint(model, scaler, metadata, output_path)
    if verbose:
        print(f"  test  MAE {test_metrics.mae:,.1f} us   RMSE {test_metrics.rmse:,.1f} us   "
              f"MAE(log1p) {test_metrics.mae_log1p:.4f}")
        print(f"  saved {output_path.relative_to(REPO_ROOT)}")
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", required=True, help="target workload trace id")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="pretrained checkpoint to adapt")
    parser.add_argument("--from-scratch", action="store_true",
                        help="train a fresh model on the target instead of adapting")
    parser.add_argument("--no-adapt", action="store_true",
                        help="evaluate the checkpoint on the target without fine-tuning")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--finetune-fraction", type=float, default=1.0,
                        help="fraction of the target's training split used for adaptation")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=DEFAULT_MAX_SEQUENCES_PER_TRACE)
    parser.add_argument("--refit-scaler", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.from_scratch:
        model_type = "scratch"
    elif args.no_adapt:
        model_type = "pretrained"
    else:
        model_type = "pretrained_finetuned"

    overrides = None
    if any(v is not None for v in (args.sequence_length, args.hidden_dim,
                                   args.num_layers, args.dropout)):
        from ml.config import load_config as _load
        defaults = _load(args.config).ml
        overrides = LstmHyperparameters(
            sequence_length=args.sequence_length or defaults.sequence_length,
            hidden_dim=args.hidden_dim or defaults.hidden_dim,
            num_layers=args.num_layers or defaults.num_layers,
            dropout=defaults.dropout if args.dropout is None else args.dropout,
        )

    adapt_to_target(args.trace, model_type, checkpoint_path=args.checkpoint,
                    config_path=args.config, output_path=args.output,
                    finetune_fraction=args.finetune_fraction,
                    hyperparameters=overrides,
                    learning_rate=args.learning_rate, epochs=args.epochs,
                    max_sequences=args.max_sequences, refit_scaler=args.refit_scaler,
                    seed=args.seed, device_preference=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
