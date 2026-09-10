"""Small controlled hyperparameter search, selected on validation MAE only.

    python -m ml.training.tune --traces msr_hm_0 msr_mds_0 ...

The grid is deliberately tiny (12 configurations by default). The research
question is whether a lightweight sequence model beats a threshold heuristic,
not how far a large search can push it, and every extra configuration is another
chance to overfit the selection.

**The test split is never touched here.** Configurations are ranked by
validation MAE in microseconds; the winner is then frozen and used for the final
pretraining run.

Writes ``results/ml/hyperparameter_search.csv``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from ml.config import REPO_ROOT, load_config
from ml.evaluation.metrics import regression_metrics
from ml.models.lstm import LstmHyperparameters, build_model
from ml.preprocessing.sequences import IntervalScaler
from ml.training.common import (
    DEFAULT_MAX_SEQUENCES_PER_TRACE,
    available_traces,
    build_split_pool,
    predict_intervals,
    resolve_device,
    set_deterministic_seed,
    train_model,
)

RESULTS_PATH = REPO_ROOT / "results" / "ml" / "hyperparameter_search.csv"

DEFAULT_SEQUENCE_LENGTHS = (5, 10, 20)
DEFAULT_HIDDEN_DIMS = (32, 64)
DEFAULT_LEARNING_RATES = (1e-3, 5e-4)

SEARCH_COLUMNS = [
    "rank", "sequence_length", "hidden_dim", "num_layers", "dropout", "learning_rate",
    "train_samples", "val_samples", "best_epoch", "epochs_run", "early_stopped",
    "val_loss", "val_mae_us", "val_rmse_us", "val_mae_log1p", "training_seconds", "selected",
]


def run_search(trace_ids: Sequence[str],
               sequence_lengths: Sequence[int] = DEFAULT_SEQUENCE_LENGTHS,
               hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
               learning_rates: Sequence[float] = DEFAULT_LEARNING_RATES,
               config_path: Path | None = None,
               max_sequences: int | None = DEFAULT_MAX_SEQUENCES_PER_TRACE,
               epochs: int | None = None,
               output_path: Path = RESULTS_PATH,
               seed: int | None = None,
               device_preference: str = "auto",
               verbose: bool = True) -> dict[str, Any]:
    """Evaluate the grid on validation data and return the winning configuration."""
    config = load_config(config_path)
    ml_config = config.ml
    seed = config.random_seed if seed is None else seed
    device = resolve_device(device_preference)

    grid = list(itertools.product(sequence_lengths, hidden_dims, learning_rates))
    if verbose:
        print(f"Hyperparameter search: {len(grid)} configurations on "
              f"{len(trace_ids)} trace(s), device {device}")

    # Sequence length changes the dataset, so pools are built once per length
    # rather than once per configuration.
    pools: dict[int, Any] = {}
    scalers: dict[int, IntervalScaler] = {}
    rows: list[dict[str, Any]] = []

    for sequence_length, hidden_dim, learning_rate in grid:
        if sequence_length not in pools:
            pool = build_split_pool(trace_ids, ml_config, max_sequences=max_sequences,
                                    sequence_length=sequence_length)
            if pool.train["targets"].size == 0 or pool.val["targets"].size == 0:
                if verbose:
                    print(f"  L={sequence_length}: no usable samples, skipping")
                pools[sequence_length] = None
                continue
            pools[sequence_length] = pool
            # Fitted on training data only, and refitted per sequence length
            # because the window changes which intervals are in the pool.
            scalers[sequence_length] = IntervalScaler.fit(pool.train["inputs"],
                                                          pool.train["targets"])
        pool = pools[sequence_length]
        if pool is None:
            continue
        scaler = scalers[sequence_length]

        hyperparameters = LstmHyperparameters(
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            num_layers=ml_config.num_layers,
            dropout=ml_config.dropout,
        )
        set_deterministic_seed(seed)
        model = build_model(hyperparameters, seed=seed)
        started = time.monotonic()
        result = train_model(model, pool, scaler, ml_config, seed=seed, device=device,
                             learning_rate=learning_rate, epochs=epochs)
        predictions = predict_intervals(model, pool.val["inputs"], scaler, device)
        metrics = regression_metrics(pool.val["targets"], predictions)

        row = {
            "sequence_length": sequence_length,
            "hidden_dim": hidden_dim,
            "num_layers": ml_config.num_layers,
            "dropout": ml_config.dropout,
            "learning_rate": learning_rate,
            "train_samples": int(pool.train["targets"].size),
            "val_samples": int(pool.val["targets"].size),
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "early_stopped": result.early_stopped,
            "val_loss": result.best_val_loss,
            "val_mae_us": metrics.mae,
            "val_rmse_us": metrics.rmse,
            "val_mae_log1p": metrics.mae_log1p,
            "training_seconds": time.monotonic() - started,
        }
        rows.append(row)
        if verbose:
            print(f"  L={sequence_length:<3} H={hidden_dim:<3} lr={learning_rate:<8g}  "
                  f"val MAE {metrics.mae:>14,.1f} us   RMSE {metrics.rmse:>14,.1f} us   "
                  f"({row['training_seconds']:.1f}s)")

    if not rows:
        raise ValueError("hyperparameter search produced no results; check the input traces")

    # Selection criterion: validation MAE. Ties break toward the smaller model,
    # which is the more conservative choice.
    rows.sort(key=lambda item: (item["val_mae_us"], item["hidden_dim"], item["sequence_length"]))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
        row["selected"] = index == 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEARCH_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    summary = {
        "selected": {
            "sequence_length": best["sequence_length"],
            "hidden_dim": best["hidden_dim"],
            "num_layers": best["num_layers"],
            "dropout": best["dropout"],
            "learning_rate": best["learning_rate"],
        },
        "selection_criterion": "lowest validation MAE (microseconds); test data not used",
        "val_mae_us": best["val_mae_us"],
        "val_rmse_us": best["val_rmse_us"],
        "configurations_evaluated": len(rows),
        "traces": list(trace_ids),
        "results_path": str(output_path.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    (output_path.parent / "hyperparameter_search_selected.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if verbose:
        print(f"\nSelected: sequence_length={best['sequence_length']} "
              f"hidden_dim={best['hidden_dim']} learning_rate={best['learning_rate']:g}"
              f"  (val MAE {best['val_mae_us']:,.1f} us)")
        print(f"Wrote {output_path.relative_to(REPO_ROOT)}")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", nargs="*", default=None)
    parser.add_argument("--sequence-lengths", type=int, nargs="*",
                        default=list(DEFAULT_SEQUENCE_LENGTHS))
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=list(DEFAULT_HIDDEN_DIMS))
    parser.add_argument("--learning-rates", type=float, nargs="*",
                        default=list(DEFAULT_LEARNING_RATES))
    parser.add_argument("--max-sequences", type=int, default=DEFAULT_MAX_SEQUENCES_PER_TRACE)
    parser.add_argument("--epochs", type=int, default=None,
                        help="cap epochs per configuration to keep the search short")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(list(argv) if argv is not None else None)

    traces = args.traces if args.traces else available_traces("msr")
    if not traces:
        print("No normalized MSR traces found.")
        return 2

    run_search(traces, sequence_lengths=args.sequence_lengths, hidden_dims=args.hidden_dims,
               learning_rates=args.learning_rates, config_path=args.config,
               max_sequences=args.max_sequences, epochs=args.epochs,
               output_path=args.output, seed=args.seed, device_preference=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
