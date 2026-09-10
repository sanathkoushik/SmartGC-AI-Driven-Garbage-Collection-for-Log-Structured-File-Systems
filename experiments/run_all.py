#!/usr/bin/env python3
"""End-to-end SmartGC pipeline: data -> model -> simulator -> results.

    python experiments/run_all.py                       # every stage
    python experiments/run_all.py --stages simulate analyze
    python experiments/run_all.py --quick               # small caps, for a smoke test

Stages, in order:

    normalize   raw traces           -> data/processed/normalized/
    inventory   statistics           -> results/dataset_inventory.csv, dataset_roles.json
    tune        validation-only grid -> results/ml/hyperparameter_search.csv
    pretrain    general MSR model    -> models/pretrained/
    models      per-target variants  -> models/finetuned/
    evaluate    models vs baselines  -> results/ml/model_comparison.csv
    predict     per-write labels     -> data/predictions/
    simulate    policy comparison    -> results/metrics/comparison.csv
    analyze     tables and plots     -> results/metrics/, results/plots/, results/final/

Everything is driven by ``config/config.yaml`` and by the frozen workload roles
in ``results/dataset_roles.json``; no stage chooses a workload by looking at a
result. A manifest recording the git commit, configuration, seed, environment
and per-stage timings is written to ``results/final/experiment_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.config import environment_metadata, load_config                 # noqa: E402
from ml.evaluation.baselines import rule_threshold_from_training        # noqa: E402
from ml.preprocessing.sequences import hot_threshold_from_training      # noqa: E402
from ml.training.common import (                                        # noqa: E402
    DEFAULT_MAX_SEQUENCES_PER_TRACE,
    MODELS_DIR,
    available_traces,
    build_split_pool,
    normalized_trace_path,
)
from ml.models.lstm import LstmHyperparameters                          # noqa: E402
from ml.training.finetune import MODEL_TYPES, adapt_to_target, default_output_path  # noqa: E402

ALL_STAGES = ("normalize", "inventory", "tune", "pretrain", "models",
              "evaluate", "predict", "simulate", "analyze")

RESULTS_DIR = REPO_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FINAL_DIR = RESULTS_DIR / "final"
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"
PRETRAINED_PATH = MODELS_DIR / "pretrained" / "msr_pretrained.pt"
COMPARISON_CSV = METRICS_DIR / "comparison.csv"
MANIFEST_PATH = FINAL_DIR / "experiment_manifest.json"

SIMULATOR_BINARIES = (
    REPO_ROOT / "simulator" / "build" / "smartgc_sim.exe",
    REPO_ROOT / "simulator" / "build" / "smartgc_sim",
    REPO_ROOT / "simulator" / "build" / "Release" / "smartgc_sim.exe",
)

DEFAULT_UTILIZATIONS = (0.70, 0.80, 0.85)


def find_simulator() -> Path:
    for candidate in SIMULATOR_BINARIES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "smartgc_sim not found. Build it first: python scripts/build_simulator.py"
    )


def distinct_written_lbas(trace_id: str, max_writes: int | None) -> tuple[int, int]:
    """(distinct LBAs written, write count) over the prefix the simulator replays."""
    path = normalized_trace_path(trace_id)
    frame = pd.read_csv(path, usecols=["lba", "operation"])
    writes = frame.loc[frame["operation"] == "W", "lba"]
    if max_writes:
        writes = writes.iloc[:max_writes]
    return int(writes.nunique()), int(len(writes))


def size_device(distinct_lbas: int, utilization: float, blocks_per_segment: int,
                gc_reserved_segments: int, gc_separate_stream: bool) -> int:
    """Segments needed so live data occupies `utilization` of raw capacity.

    Sized for the *most restrictive* policy (two user append points), because
    every policy in a comparison must run on identical geometry and the
    segregated policies lose one segment of usable capacity to the second log.
    """
    if not 0.0 < utilization < 1.0:
        raise ValueError("utilization must be in (0, 1)")
    segments = max(1, math.ceil(distinct_lbas / (utilization * blocks_per_segment)))
    unavailable = gc_reserved_segments + 2 + (1 if gc_separate_stream else 0)
    while (segments - unavailable) * blocks_per_segment < distinct_lbas:
        segments += 1
    return segments


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = load_config(args.config)
        self.config_path = args.config or (REPO_ROOT / "config" / "config.yaml")
        self.seed = args.seed if args.seed is not None else self.config.random_seed
        self.stage_times: dict[str, float] = {}
        self.notes: list[str] = []
        self.roles: dict[str, Any] = {}
        self.selected_hyperparameters: dict[str, Any] = {}
        self.max_sequences = args.max_sequences
        self.max_writes = args.max_writes
        self.utilizations = tuple(args.utilizations)

    # -- helpers ------------------------------------------------------------
    def run(self, command: Sequence[str], label: str) -> subprocess.CompletedProcess:
        print(f"\n$ {' '.join(str(part) for part in command)}", flush=True)
        result = subprocess.run([str(part) for part in command], cwd=str(REPO_ROOT))
        if result.returncode != 0:
            raise RuntimeError(f"{label} failed with exit code {result.returncode}")
        return result

    def load_roles(self) -> dict[str, Any]:
        path = RESULTS_DIR / "dataset_roles.json"
        if not path.is_file():
            raise FileNotFoundError(
                "results/dataset_roles.json is missing; run the 'inventory' stage first"
            )
        self.roles = json.loads(path.read_text(encoding="utf-8"))
        return self.roles

    def selected_targets(self) -> list[str]:
        """Workloads to model and simulate: the frozen targets plus external set.

        `--only-targets` narrows this without touching dataset_roles.json, which
        records the assignment as it was frozen. The restriction is written to
        the manifest so a partial run is never mistaken for the full one.
        """
        roles = self.roles or self.load_roles()
        available = list(roles["targets"]) + list(roles["external"])
        if not self.args.only_targets:
            return available

        chosen = [name for name in available if name in set(self.args.only_targets)]
        unknown = sorted(set(self.args.only_targets) - set(available))
        if unknown:
            raise ValueError(f"--only-targets names workloads that have no role: {unknown}")
        skipped = [name for name in available if name not in set(chosen)]
        if skipped:
            self.notes.append(
                "restricted to --only-targets " + ", ".join(chosen)
                + "; not run: " + ", ".join(skipped))
        return chosen

    # -- stages -------------------------------------------------------------
    def stage_normalize(self) -> None:
        command = [sys.executable, "-m", "ml.preprocessing.normalize", "--discover"]
        if self.args.max_records:
            command += ["--max-records", str(self.args.max_records)]
        self.run(command, "normalize")

    def stage_inventory(self) -> None:
        self.run([sys.executable, "experiments/analyze_datasets.py", "--sources"], "inventory")
        self.load_roles()

    def stage_tune(self) -> None:
        roles = self.roles or self.load_roles()
        traces = roles["pretrain"]
        if not traces:
            raise RuntimeError("no pretraining traces available for the hyperparameter search")
        command = [sys.executable, "-m", "ml.training.tune", "--traces", *traces,
                   "--max-sequences", str(self.args.tune_max_sequences),
                   "--seed", str(self.seed)]
        if self.args.tune_epochs:
            command += ["--epochs", str(self.args.tune_epochs)]
        self.run(command, "tune")

        selected_path = RESULTS_DIR / "ml" / "hyperparameter_search_selected.json"
        if selected_path.is_file():
            self.selected_hyperparameters = json.loads(
                selected_path.read_text(encoding="utf-8"))["selected"]

    def _resolve_hyperparameters(self) -> dict[str, Any]:
        if self.selected_hyperparameters:
            return self.selected_hyperparameters
        selected_path = RESULTS_DIR / "ml" / "hyperparameter_search_selected.json"
        if selected_path.is_file():
            self.selected_hyperparameters = json.loads(
                selected_path.read_text(encoding="utf-8"))["selected"]
            return self.selected_hyperparameters
        # No search was run: fall back to config.yaml and say so in the manifest.
        self.notes.append("hyperparameter search not run; using config.yaml values")
        return {"sequence_length": self.config.ml.sequence_length,
                "hidden_dim": self.config.ml.hidden_dim,
                "num_layers": self.config.ml.num_layers,
                "dropout": self.config.ml.dropout,
                "learning_rate": self.config.ml.learning_rate}

    def stage_pretrain(self) -> None:
        roles = self.roles or self.load_roles()
        traces = roles["pretrain"]
        if not traces:
            raise RuntimeError("no pretraining traces available")
        hyperparameters = self._resolve_hyperparameters()
        command = [sys.executable, "-m", "ml.training.pretrain",
                   "--traces", *traces,
                   "--output", str(PRETRAINED_PATH),
                   "--sequence-length", str(hyperparameters["sequence_length"]),
                   "--hidden-dim", str(hyperparameters["hidden_dim"]),
                   "--num-layers", str(hyperparameters["num_layers"]),
                   "--dropout", str(hyperparameters["dropout"]),
                   "--learning-rate", str(hyperparameters["learning_rate"]),
                   "--max-sequences", str(self.max_sequences),
                   "--seed", str(self.seed)]
        self.run(command, "pretrain")

    def stage_models(self) -> None:
        targets = self.selected_targets()
        if not targets:
            raise RuntimeError("no target or external workloads available")
        hyperparameters = self._resolve_hyperparameters()

        for trace_id in targets:
            for model_type in MODEL_TYPES:
                print(f"\n--- {model_type} on {trace_id} ---")
                adapt_to_target(
                    trace_id, model_type,
                    checkpoint_path=None if model_type == "scratch" else PRETRAINED_PATH,
                    config_path=self.args.config,
                    max_sequences=self.max_sequences,
                    # The scratch model gets the architecture the search chose,
                    # so all three variants share a sequence length and therefore
                    # an identical test split.
                    hyperparameters=(LstmHyperparameters(
                        sequence_length=hyperparameters["sequence_length"],
                        hidden_dim=hyperparameters["hidden_dim"],
                        num_layers=hyperparameters["num_layers"],
                        dropout=hyperparameters["dropout"],
                    ) if model_type == "scratch" else None),
                    learning_rate=(hyperparameters["learning_rate"]
                                   if model_type == "scratch" else None),
                    seed=self.seed,
                    device_preference=self.args.device,
                )

    def stage_evaluate(self) -> None:
        targets = self.selected_targets()
        self.run([sys.executable, "experiments/evaluate_models.py",
                  "--targets", *targets,
                  "--max-sequences", str(self.max_sequences)], "evaluate")
        self.run([sys.executable, "experiments/transfer_comparison.py",
                  "--targets", *targets,
                  "--max-sequences", str(self.max_sequences)], "transfer comparison")

    def stage_predict(self) -> None:
        targets = self.selected_targets()
        for trace_id in targets:
            for model_type in MODEL_TYPES:
                checkpoint = default_output_path(model_type, trace_id)
                if not checkpoint.is_file():
                    print(f"  skipping {model_type} on {trace_id}: no checkpoint")
                    continue
                self.run([sys.executable, "-m", "ml.inference.predict",
                          "--trace", trace_id,
                          "--checkpoint", str(checkpoint),
                          "--model-version", model_type], "predict")

    def _rule_threshold(self, trace_id: str, min_history: int) -> float:
        """Hot cutoff for the heuristic, from the target's TRAINING period only.

        Derived from the *running-mean* statistic the C++ rule computes, not from
        the next-interval targets the LSTM is trained on. Those distributions
        differ by orders of magnitude, and reusing the model's threshold made the
        rule label almost nothing HOT and degenerate into "separate by how much
        history a block has". Same percentile, same training-only rule, applied
        to the statistic actually being thresholded.
        """
        return rule_threshold_from_training(
            normalized_trace_path(trace_id),
            min_history=min_history,
            train_fraction=self.config.ml.train_split,
            percentile=self.config.ml.hot_percentile_cutoff,
            max_writes=self.max_writes)

    def stage_simulate(self) -> None:
        targets = self.selected_targets()
        simulator = find_simulator()
        simulator_config = self.config.simulator

        if COMPARISON_CSV.exists() and not self.args.append_metrics:
            COMPARISON_CSV.unlink()
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        reports_dir = METRICS_DIR / "runs"
        reports_dir.mkdir(parents=True, exist_ok=True)

        for trace_id in targets:
            trace_path = normalized_trace_path(trace_id)
            if not trace_path.is_file():
                print(f"  skipping {trace_id}: not normalized")
                continue
            distinct, write_count = distinct_written_lbas(trace_id, self.max_writes)
            dataset = trace_id.split("_", 1)[0]
            rule_min_history = self._resolve_hyperparameters()["sequence_length"]
            rule_threshold = self._rule_threshold(trace_id, rule_min_history)
            print(f"\n=== {trace_id}: {write_count:,} writes, {distinct:,} distinct LBAs, "
                  f"rule threshold {rule_threshold:,.0f} us ===")

            for utilization in self.utilizations:
                segments = size_device(distinct, utilization,
                                       simulator_config.blocks_per_segment,
                                       simulator_config.gc_reserved_segments,
                                       simulator_config.gc_separate_stream)
                base = [str(simulator),
                        "--config", str(self.config_path),
                        "--trace", str(trace_path),
                        "--total-segments", str(segments),
                        "--dataset", dataset,
                        "--trace-name", trace_id,
                        "--export-metrics", str(COMPARISON_CSV),
                        "--seed", str(self.seed),
                        "--quiet"]
                if self.max_writes:
                    base += ["--max-writes", str(self.max_writes)]

                print(f"\n-- utilization {utilization:.0%}: {segments} segments "
                      f"({segments * simulator_config.blocks_per_segment:,} blocks) --")

                tag = f"{trace_id}__u{int(round(utilization * 100))}"
                self.run(base + ["--placement", "MIXED", "--model-type", "none",
                                 "--report-json", str(reports_dir / f"{tag}__MIXED.json")],
                         "simulate MIXED")
                self.run(base + ["--placement", "RULE_BASED", "--model-type", "rule",
                                 "--rule-threshold-us", f"{rule_threshold:.6f}",
                                 "--rule-min-history", str(rule_min_history),
                                 "--report-json", str(reports_dir / f"{tag}__RULE_BASED.json")],
                         "simulate RULE_BASED")

                for model_type in MODEL_TYPES:
                    predictions = PREDICTIONS_DIR / f"{model_type}__{trace_id}.csv"
                    if not predictions.is_file():
                        print(f"  skipping LSTM_SMARTGC/{model_type}: no predictions file")
                        continue
                    self.run(base + ["--placement", "LSTM_SMARTGC",
                                     "--model-type", model_type,
                                     "--predictions", str(predictions),
                                     "--report-json",
                                     str(reports_dir / f"{tag}__LSTM_{model_type}.json")],
                             f"simulate LSTM_SMARTGC/{model_type}")

    def stage_analyze(self) -> None:
        self.run([sys.executable, "experiments/analyze_results.py"], "analyze")

    # -- driver -------------------------------------------------------------
    def execute(self, stages: Sequence[str]) -> int:
        started = time.monotonic()
        for stage in stages:
            method = getattr(self, f"stage_{stage}")
            print(f"\n{'=' * 72}\nSTAGE: {stage}\n{'=' * 72}")
            stage_started = time.monotonic()
            method()
            self.stage_times[stage] = time.monotonic() - stage_started
            print(f"\n[{stage}] completed in {self.stage_times[stage]:.1f}s")

        self.write_manifest(total_seconds=time.monotonic() - started, stages=stages)
        return 0

    def write_manifest(self, total_seconds: float, stages: Sequence[str]) -> None:
        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        manifest = {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stages_run": list(stages),
            "stage_seconds": self.stage_times,
            "total_seconds": total_seconds,
            "seed": self.seed,
            "config_path": str(self.config_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "config": self.config.to_dict(),
            "selected_hyperparameters": self.selected_hyperparameters,
            "workload_roles": self.roles,
            "utilizations": list(self.utilizations),
            "only_targets": list(self.args.only_targets) if self.args.only_targets else None,
            "max_writes_per_run": self.max_writes,
            "max_sequences_per_trace": self.max_sequences,
            "environment": environment_metadata(),
            "notes": self.notes,
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
                                 encoding="utf-8")
        print(f"\nManifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="*", default=list(ALL_STAGES), choices=list(ALL_STAGES))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-records", type=int, default=None,
                        help="cap raw records per trace during normalization")
    parser.add_argument("--max-sequences", type=int, default=DEFAULT_MAX_SEQUENCES_PER_TRACE,
                        help="cap training sequences per trace (chronological prefix)")
    parser.add_argument("--tune-max-sequences", type=int, default=100_000,
                        help="smaller cap for the hyperparameter search, which trains many models")
    parser.add_argument("--tune-epochs", type=int, default=8)
    parser.add_argument("--max-writes", type=int, default=2_000_000,
                        help="cap write requests replayed per simulation (0 = whole trace)")
    parser.add_argument("--utilizations", type=float, nargs="*", default=list(DEFAULT_UTILIZATIONS))
    parser.add_argument("--only-targets", nargs="*", default=None,
                        help="restrict modelling/simulation to these workloads; the frozen "
                             "role assignment is left untouched and the restriction is "
                             "recorded in the manifest")
    parser.add_argument("--append-metrics", action="store_true",
                        help="append to results/metrics/comparison.csv instead of replacing it")
    parser.add_argument("--quick", action="store_true",
                        help="tiny caps for a smoke test; NOT for reportable results")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.quick:
        args.max_records = args.max_records or 300_000
        args.max_sequences = min(args.max_sequences, 50_000)
        args.tune_max_sequences = min(args.tune_max_sequences, 20_000)
        args.tune_epochs = min(args.tune_epochs, 3)
        args.max_writes = min(args.max_writes or 300_000, 300_000)
        args.utilizations = [0.80]

    pipeline = Pipeline(args)
    try:
        return pipeline.execute(args.stages)
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"\nPIPELINE FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
