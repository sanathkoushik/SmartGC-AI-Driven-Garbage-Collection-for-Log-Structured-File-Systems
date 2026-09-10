#!/usr/bin/env python3
"""Turn the measured CSVs into the tables and figures for the write-up.

    python experiments/analyze_results.py

Reads only files produced by earlier stages and writes:

    results/metrics/ablation.csv     policy comparison with change vs MIXED
    results/final/*.csv              the conference tables
    results/plots/*.png              the figures
    results/final/README.md          what the pipeline produced, with real numbers

Every value plotted comes from a CSV on disk. Nothing here computes a result of
its own beyond arithmetic on measured columns, and a figure whose inputs are
missing is skipped with a message rather than drawn from partial data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.ticker import FuncFormatter                             # noqa: E402

from ml.preprocessing.sequences import (                                # noqa: E402
    order_chronologically, build_sequences, rewrite_intervals,
)

RESULTS_DIR = REPO_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
FINAL_DIR = RESULTS_DIR / "final"
ML_DIR = RESULTS_DIR / "ml"
STATS_DIR = RESULTS_DIR / "dataset_statistics"
NORMALIZED_DIR = REPO_ROOT / "data" / "processed" / "normalized"
MODELS_DIR = REPO_ROOT / "models"

COMPARISON_CSV = METRICS_DIR / "comparison.csv"
ABLATION_CSV = METRICS_DIR / "ablation.csv"
MODEL_COMPARISON_CSV = ML_DIR / "model_comparison.csv"
CLASSIFICATION_CSV = ML_DIR / "classification_metrics.csv"

# Validated categorical palette (dataviz skill reference instance), assigned in
# fixed slot order so a series keeps its colour across every figure.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                 "#008300", "#4a3aa7", "#e34948"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#dcdbd6"
SURFACE = "#fcfcfb"

# Fixed display order and colour slot for the five compared configurations.
POLICY_ORDER = ["MIXED", "RULE_BASED", "LSTM scratch", "LSTM pretrained",
                "LSTM pretrained+finetuned"]
POLICY_COLOR = {name: SERIES_COLORS[i] for i, name in enumerate(POLICY_ORDER)}


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.size": 9,
        "figure.dpi": 130,
        "lines.linewidth": 2.0,
    })


def save(fig: plt.Figure, name: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")
    return path


def label_bars(ax: plt.Axes, bars, values: Sequence[float], fmt: str = "{:.3f}") -> None:
    """Direct value labels. The palette's contrast WARN obliges visible labels."""
    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", va="bottom",
                    fontsize=7.5, color=TEXT_SECONDARY)


def policy_label(row: pd.Series) -> str:
    policy = str(row["policy"])
    if policy != "LSTM_SMARTGC":
        return policy
    model = str(row.get("model_type", ""))
    return {"scratch": "LSTM scratch",
            "pretrained": "LSTM pretrained",
            "pretrained_finetuned": "LSTM pretrained+finetuned"}.get(model, f"LSTM {model}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def build_ablation(comparison: pd.DataFrame) -> pd.DataFrame:
    """Add change-vs-MIXED columns within each (trace, utilization) group."""
    frame = comparison.copy()
    frame["configuration"] = frame.apply(policy_label, axis=1)
    frame["utilization_pct"] = (frame["target_utilization"] * 100).round(1)

    rows = []
    for (trace, utilization), group in frame.groupby(["trace", "utilization_pct"], sort=True):
        baseline = group[group["policy"] == "MIXED"]
        if baseline.empty:
            print(f"  no MIXED baseline for {trace} @ {utilization}%, skipping group")
            continue
        base = baseline.iloc[0]
        for _, row in group.iterrows():
            rows.append({
                "dataset": row["dataset"],
                "trace": trace,
                "utilization_pct": utilization,
                "configuration": row["configuration"],
                "policy": row["policy"],
                "model_type": row["model_type"],
                "total_segments": row["total_segments"],
                "blocks_per_segment": row["blocks_per_segment"],
                "seed": row["seed"],
                "total_write_requests": row["total_write_requests"],
                "logical_bytes_written": row["logical_bytes_written"],
                "physical_bytes_written": row["physical_bytes_written"],
                "gc_bytes_copied": row["gc_bytes_copied"],
                "valid_blocks_migrated": row["valid_blocks_migrated"],
                "gc_count": row["gc_count"],
                "main_stream_writes": row["main_stream_writes"],
                "cold_stream_writes": row["cold_stream_writes"],
                "unpredicted_writes": row["unpredicted_writes"],
                "prediction_coverage": (1.0 - row["unpredicted_writes"] / row["total_write_requests"]
                                        if row["total_write_requests"] else 0.0),
                "waf": row["waf"],
                "waf_vs_mixed_pct": (row["waf"] - base["waf"]) / base["waf"] * 100.0,
                "migrations_vs_mixed_pct": ((row["valid_blocks_migrated"] - base["valid_blocks_migrated"])
                                            / base["valid_blocks_migrated"] * 100.0
                                            if base["valid_blocks_migrated"] else float("nan")),
                "gc_bytes_vs_mixed_pct": ((row["gc_bytes_copied"] - base["gc_bytes_copied"])
                                          / base["gc_bytes_copied"] * 100.0
                                          if base["gc_bytes_copied"] else float("nan")),
            })
    ablation = pd.DataFrame(rows)
    if not ablation.empty:
        order = {name: i for i, name in enumerate(POLICY_ORDER)}
        ablation["_order"] = ablation["configuration"].map(lambda v: order.get(v, 99))
        ablation = ablation.sort_values(["trace", "utilization_pct", "_order"]).drop(columns="_order")
    return ablation


# ---------------------------------------------------------------------------
# Dataset figures
# ---------------------------------------------------------------------------

def load_dataset_statistics() -> pd.DataFrame:
    if not STATS_DIR.is_dir():
        return pd.DataFrame()
    records = []
    for path in sorted(STATS_DIR.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(records)


def plot_dataset_composition(statistics: pd.DataFrame) -> None:
    if statistics.empty:
        return
    frame = statistics.sort_values("trace_id")
    labels = frame["trace_id"].str.replace("msr_", "", regex=False).str.replace(
        "systor_", "", regex=False)
    writes = frame["write_blocks"].to_numpy() / 1e6
    reads = frame["read_blocks"].to_numpy() / 1e6

    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(frame) + 3), 3.4))
    x = np.arange(len(frame))
    # 2px surface gap between adjacent fills: rendered as a small bar gap.
    ax.bar(x - 0.21, writes, width=0.40, color=SERIES_COLORS[0], label="Writes")
    ax.bar(x + 0.21, reads, width=0.40, color=SERIES_COLORS[1], label="Reads")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Block accesses (millions)")
    ax.set_title("Write/read composition of each normalized workload")
    ax.legend()
    save(fig, "01_dataset_composition.png")


def plot_write_percentage(statistics: pd.DataFrame) -> None:
    if statistics.empty:
        return
    frame = statistics.sort_values("write_percentage", ascending=False)
    labels = frame["trace_id"].str.replace("msr_", "", regex=False)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(frame) + 3), 3.2))
    bars = ax.bar(labels, frame["write_percentage"], color=SERIES_COLORS[0], width=0.62)
    label_bars(ax, bars, frame["write_percentage"].tolist(), "{:.1f}%")
    ax.set_ylabel("Writes as % of block accesses")
    ax.set_ylim(0, 105)
    ax.set_title("Write intensity by workload")
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    save(fig, "02_write_percentage.png")


def _trace_write_frame(trace_id: str, max_rows: int = 3_000_000) -> pd.DataFrame | None:
    path = NORMALIZED_DIR / f"{trace_id}.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path, usecols=["timestamp", "lba", "operation"], nrows=max_rows)
    return frame[frame["operation"] == "W"]


def plot_rewrite_interval_distribution(trace_ids: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    plotted = 0
    for index, trace_id in enumerate(trace_ids[:6]):
        writes = _trace_write_frame(trace_id)
        if writes is None or writes.empty:
            continue
        intervals = rewrite_intervals(writes["lba"].to_numpy(),
                                      writes["timestamp"].to_numpy())["interval"]
        intervals = intervals[intervals > 0]
        if intervals.size < 100:
            continue
        # ECDF on a log axis: intervals span many orders of magnitude, so a
        # linear histogram would show one spike and nothing else.
        values = np.sort(intervals)
        ecdf = np.arange(1, values.size + 1) / values.size
        step = max(1, values.size // 4000)
        ax.plot(values[::step] / 1e6, ecdf[::step],
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
                label=trace_id.replace("msr_", "").replace("systor_", ""))
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("Rewrite interval (seconds, log scale)")
    ax.set_ylabel("Cumulative fraction of rewrites")
    ax.set_title("Rewrite-interval distribution")
    ax.legend()
    save(fig, "03_rewrite_interval_distribution.png")


def plot_lba_popularity(trace_ids: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    plotted = 0
    for index, trace_id in enumerate(trace_ids[:6]):
        writes = _trace_write_frame(trace_id)
        if writes is None or writes.empty:
            continue
        counts = np.sort(writes["lba"].value_counts().to_numpy())[::-1]
        ax.plot(np.arange(1, counts.size + 1), counts,
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
                label=trace_id.replace("msr_", "").replace("systor_", ""))
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("LBA rank (log scale)")
    ax.set_ylabel("Writes to that LBA (log scale)")
    ax.set_title("LBA write popularity")
    ax.legend()
    save(fig, "04_lba_popularity.png")


def plot_cumulative_writes(trace_ids: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    plotted = 0
    for index, trace_id in enumerate(trace_ids[:6]):
        writes = _trace_write_frame(trace_id)
        if writes is None or writes.empty:
            continue
        counts = np.sort(writes["lba"].value_counts().to_numpy())[::-1]
        cumulative = np.cumsum(counts) / counts.sum()
        fraction = np.arange(1, counts.size + 1) / counts.size
        ax.plot(fraction * 100, cumulative * 100,
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
                label=trace_id.replace("msr_", "").replace("systor_", ""))
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.plot([0, 100], [0, 100], color=TEXT_SECONDARY, linewidth=1.0,
            linestyle=":", label="uniform")
    ax.set_xlabel("Share of written LBAs (%, hottest first)")
    ax.set_ylabel("Share of write traffic (%)")
    ax.set_title("Concentration of write traffic")
    ax.legend()
    save(fig, "05_cumulative_writes_by_lba.png")


def plot_request_rate(trace_ids: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    plotted = 0
    for index, trace_id in enumerate(trace_ids[:6]):
        writes = _trace_write_frame(trace_id)
        if writes is None or writes.empty:
            continue
        seconds = writes["timestamp"].to_numpy() / 1e6
        span = seconds.max() - seconds.min()
        if span <= 0:
            continue
        bins = np.linspace(seconds.min(), seconds.max(), 120)
        counts, edges = np.histogram(seconds, bins=bins)
        width = np.diff(edges)
        ax.plot(edges[:-1] / 3600.0, counts / np.maximum(width, 1e-9),
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
                label=trace_id.replace("msr_", "").replace("systor_", ""),
                linewidth=1.4)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_xlabel("Time since trace start (hours)")
    ax.set_ylabel("Block writes per second (log scale)")
    ax.set_title("Write rate over time")
    ax.legend()
    save(fig, "06_request_rate_over_time.png")


def plot_hot_cold_distribution(model_comparison: pd.DataFrame) -> None:
    if model_comparison.empty:
        return
    frame = model_comparison[model_comparison["family"] == "lstm"]
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    traces = sorted(frame["trace"].unique())
    x = np.arange(len(traces))
    actual = [frame[frame["trace"] == t]["actual_hot_fraction"].iloc[0] * 100 for t in traces]
    predicted = [frame[(frame["trace"] == t)]["predicted_hot_fraction"].mean() * 100 for t in traces]
    bars_a = ax.bar(x - 0.21, actual, width=0.40, color=SERIES_COLORS[0], label="Actually HOT")
    bars_p = ax.bar(x + 0.21, predicted, width=0.40, color=SERIES_COLORS[1],
                    label="Predicted HOT (mean over models)")
    label_bars(ax, bars_a, actual, "{:.1f}%")
    label_bars(ax, bars_p, predicted, "{:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("msr_", "").replace("systor_", "") for t in traces])
    ax.set_ylabel("Share of test writes (%)")
    ax.set_title("Hot/cold class balance on the held-out test split")
    ax.legend()
    save(fig, "07_hot_cold_distribution.png")


# ---------------------------------------------------------------------------
# Model figures
# ---------------------------------------------------------------------------

def load_model_metadata() -> list[dict[str, Any]]:
    records = []
    for path in sorted(MODELS_DIR.rglob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return records


def plot_training_curves(metadata: list[dict[str, Any]]) -> None:
    curves = [m for m in metadata if m.get("training_history")]
    if not curves:
        return
    count = min(len(curves), 4)
    fig, axes = plt.subplots(1, count, figsize=(3.4 * count, 3.0), squeeze=False)
    for index, meta in enumerate(curves[:count]):
        ax = axes[0][index]
        history = pd.DataFrame(meta["training_history"])
        ax.plot(history["epoch"], history["train_loss"], color=SERIES_COLORS[0], label="train")
        ax.plot(history["epoch"], history["val_loss"], color=SERIES_COLORS[1], label="validation")
        best = meta.get("best_epoch")
        if best:
            ax.axvline(best, color=TEXT_SECONDARY, linewidth=1.0, linestyle=":")
            ax.annotate(f"best epoch {best}", (best, ax.get_ylim()[1]),
                        textcoords="offset points", xytext=(3, -10),
                        fontsize=7, color=TEXT_SECONDARY)
        stage = meta.get("stage", "?")
        target = meta.get("target_trace") or ",".join(meta.get("traces", []))[:24]
        ax.set_title(f"{stage}\n{target}", fontsize=9)
        ax.set_xlabel("Epoch")
        if index == 0:
            ax.set_ylabel(f"{meta.get('loss_function', 'loss')} (log1p space)")
            ax.legend()
    fig.suptitle("Training and validation loss", y=1.02, fontsize=11, color=TEXT_PRIMARY)
    save(fig, "08_training_curves.png")


def plot_prediction_quality(model_comparison: pd.DataFrame) -> None:
    if model_comparison.empty:
        return
    for metric, name, fmt in (("mae_us", "09_mae_comparison.png", "{:,.0f}"),
                              ("rmse_us", "10_rmse_comparison.png", "{:,.0f}"),
                              ("mae_log1p", "11_mae_log_comparison.png", "{:.4f}")):
        traces = sorted(model_comparison["trace"].unique())
        predictors = list(model_comparison["predictor"].unique())
        fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(predictors) + 3), 3.6))
        width = 0.8 / max(len(traces), 1)
        x = np.arange(len(predictors))
        for index, trace in enumerate(traces):
            subset = model_comparison[model_comparison["trace"] == trace]
            values = [subset[subset["predictor"] == p][metric].mean() for p in predictors]
            offset = (index - (len(traces) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width=width * 0.92,
                          color=SERIES_COLORS[index % len(SERIES_COLORS)],
                          label=trace.replace("msr_", "").replace("systor_", ""))
            if len(traces) <= 2:
                label_bars(ax, bars, values, fmt)
        ax.set_xticks(x)
        ax.set_xticklabels(predictors, rotation=30, ha="right")
        ax.set_ylabel({"mae_us": "MAE (microseconds)", "rmse_us": "RMSE (microseconds)",
                       "mae_log1p": "MAE in log1p space"}[metric])
        ax.set_title({"mae_us": "Rewrite-interval MAE on the held-out test split",
                      "rmse_us": "Rewrite-interval RMSE on the held-out test split",
                      "mae_log1p": "Log-space MAE (the scale the model optimises)"}[metric])
        if metric != "mae_log1p":
            ax.set_yscale("log")
        ax.legend()
        save(fig, name)


def plot_classification_comparison(model_comparison: pd.DataFrame) -> None:
    if model_comparison.empty:
        return
    traces = sorted(model_comparison["trace"].unique())
    predictors = list(model_comparison["predictor"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    for axis_index, metric in enumerate(("f1", "accuracy")):
        ax = axes[axis_index]
        x = np.arange(len(predictors))
        width = 0.8 / max(len(traces), 1)
        for index, trace in enumerate(traces):
            subset = model_comparison[model_comparison["trace"] == trace]
            values = [subset[subset["predictor"] == p][metric].mean() for p in predictors]
            offset = (index - (len(traces) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width=width * 0.92,
                          color=SERIES_COLORS[index % len(SERIES_COLORS)],
                          label=trace.replace("msr_", "").replace("systor_", ""))
            if len(traces) <= 2:
                label_bars(ax, bars, values, "{:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(predictors, rotation=30, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(metric.upper() if metric == "f1" else "Accuracy")
        ax.set_title(f"Hot/cold {metric.upper() if metric == 'f1' else 'accuracy'}")
        if axis_index == 0:
            ax.legend()
    save(fig, "12_classification_comparison.png")


def plot_confusion_matrices(classification: pd.DataFrame) -> None:
    if classification.empty:
        return
    frame = classification[classification["family"] == "lstm"]
    if frame.empty:
        return
    count = min(len(frame), 6)
    columns = min(count, 3)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.0 * rows), squeeze=False)
    for index in range(rows * columns):
        ax = axes[index // columns][index % columns]
        if index >= count:
            ax.axis("off")
            continue
        row = frame.iloc[index]
        matrix = np.array([[row["true_negatives"], row["false_positives"]],
                           [row["false_negatives"], row["true_positives"]]], dtype=float)
        share = matrix / max(matrix.sum(), 1)
        # Single-hue sequential ramp for magnitude, per the colour formula.
        ax.imshow(share, cmap="Blues", vmin=0, vmax=share.max() if share.max() else 1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(matrix[i, j]):,}\n{share[i, j]:.1%}",
                        ha="center", va="center", fontsize=8,
                        color=TEXT_PRIMARY if share[i, j] < share.max() * 0.6 else SURFACE)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred COLD", "pred HOT"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["actual COLD", "actual HOT"], fontsize=7)
        ax.grid(False)
        ax.set_title(f"{row['trace'].replace('msr_', '')}\n{row['predictor']}", fontsize=8)
    fig.suptitle("Hot/cold confusion matrices (held-out test split)", y=1.01,
                 fontsize=11, color=TEXT_PRIMARY)
    save(fig, "13_confusion_matrices.png")


def plot_transfer_comparison(model_comparison: pd.DataFrame) -> None:
    """scratch vs pretrained vs fine-tuned: the transfer-learning question."""
    frame = model_comparison[model_comparison["family"] == "lstm"]
    if frame.empty:
        return
    order = ["scratch", "pretrained", "pretrained_finetuned"]
    traces = sorted(frame["trace"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for axis_index, (metric, ylabel, fmt) in enumerate(
            (("mae_log1p", "MAE in log1p space (lower is better)", "{:.4f}"),
             ("f1", "Hot/cold F1 (higher is better)", "{:.3f}"))):
        ax = axes[axis_index]
        x = np.arange(len(order))
        width = 0.8 / max(len(traces), 1)
        for index, trace in enumerate(traces):
            subset = frame[frame["trace"] == trace]
            values = []
            for predictor in order:
                match = subset[subset["predictor"] == predictor]
                values.append(match[metric].iloc[0] if not match.empty else np.nan)
            offset = (index - (len(traces) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width=width * 0.92,
                          color=SERIES_COLORS[index % len(SERIES_COLORS)],
                          label=trace.replace("msr_", "").replace("systor_", ""))
            label_bars(ax, bars, values, fmt)
        ax.set_xticks(x)
        ax.set_xticklabels(["from scratch", "pretrained\n(no adaptation)",
                            "pretrained\n+ fine-tuned"], fontsize=8)
        ax.set_ylabel(ylabel)
        if axis_index == 0:
            ax.legend()
    fig.suptitle("Does pretraining help? Does fine-tuning help?", y=1.02,
                 fontsize=11, color=TEXT_PRIMARY)
    save(fig, "14_transfer_learning_comparison.png")


# ---------------------------------------------------------------------------
# Simulator figures
# ---------------------------------------------------------------------------

def plot_policy_metric(ablation: pd.DataFrame, column: str, ylabel: str,
                       title: str, filename: str, fmt: str = "{:,.0f}",
                       log_scale: bool = False) -> None:
    if ablation.empty or column not in ablation:
        return
    groups = sorted(ablation.groupby(["trace", "utilization_pct"]).groups.keys())
    if not groups:
        return
    configurations = [c for c in POLICY_ORDER if c in set(ablation["configuration"])]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(groups) + 3), 3.8))
    x = np.arange(len(groups))
    width = 0.8 / max(len(configurations), 1)
    for index, configuration in enumerate(configurations):
        values = []
        for trace, utilization in groups:
            match = ablation[(ablation["trace"] == trace)
                             & (ablation["utilization_pct"] == utilization)
                             & (ablation["configuration"] == configuration)]
            values.append(match[column].iloc[0] if not match.empty else np.nan)
        offset = (index - (len(configurations) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width * 0.92,
                      color=POLICY_COLOR.get(configuration, SERIES_COLORS[index]),
                      label=configuration)
        if len(groups) <= 3:
            label_bars(ax, bars, values, fmt)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t.replace('msr_', '').replace('systor_', '')}\n{u:.0f}%"
                        for t, u in groups], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Workload and target utilization")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(ncols=2)
    save(fig, filename)


def plot_waf_vs_utilization(ablation: pd.DataFrame) -> None:
    if ablation.empty:
        return
    traces = sorted(ablation["trace"].unique())
    fig, axes = plt.subplots(1, len(traces), figsize=(4.0 * len(traces), 3.4), squeeze=False)
    for index, trace in enumerate(traces):
        ax = axes[0][index]
        subset = ablation[ablation["trace"] == trace]
        for configuration in POLICY_ORDER:
            rows = subset[subset["configuration"] == configuration].sort_values("utilization_pct")
            if rows.empty:
                continue
            ax.plot(rows["utilization_pct"], rows["waf"], marker="o", markersize=5,
                    color=POLICY_COLOR[configuration], label=configuration)
        ax.set_xlabel("Target utilization (%)")
        if index == 0:
            ax.set_ylabel("Write amplification factor")
            ax.legend()
        ax.set_title(trace.replace("msr_", "").replace("systor_", ""))
    fig.suptitle("WAF against device utilization", y=1.02, fontsize=11, color=TEXT_PRIMARY)
    save(fig, "18_waf_vs_utilization.png")


def plot_improvement(ablation: pd.DataFrame) -> None:
    if ablation.empty:
        return
    frame = ablation[ablation["configuration"] != "MIXED"]
    if frame.empty:
        return
    groups = sorted(frame.groupby(["trace", "utilization_pct"]).groups.keys())
    configurations = [c for c in POLICY_ORDER[1:] if c in set(frame["configuration"])]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(groups) + 3), 3.8))
    x = np.arange(len(groups))
    width = 0.8 / max(len(configurations), 1)
    for index, configuration in enumerate(configurations):
        values = []
        for trace, utilization in groups:
            match = frame[(frame["trace"] == trace)
                          & (frame["utilization_pct"] == utilization)
                          & (frame["configuration"] == configuration)]
            values.append(match["waf_vs_mixed_pct"].iloc[0] if not match.empty else np.nan)
        offset = (index - (len(configurations) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width * 0.92,
                      color=POLICY_COLOR.get(configuration, SERIES_COLORS[index + 1]),
                      label=configuration)
        label_bars(ax, bars, values, "{:+.1f}%")
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t.replace('msr_', '').replace('systor_', '')}\n{u:.0f}%"
                        for t, u in groups], fontsize=8)
    ax.set_ylabel("Change in WAF vs MIXED (%)\nnegative is better")
    ax.set_title("SmartGC and the heuristic, relative to the MIXED baseline")
    ax.legend(ncols=2)
    save(fig, "19_waf_improvement_vs_mixed.png")


def plot_prediction_coverage(ablation: pd.DataFrame) -> None:
    frame = ablation[ablation["configuration"].str.startswith("LSTM")
                     | (ablation["configuration"] == "RULE_BASED")]
    if frame.empty:
        return
    groups = sorted(frame.groupby(["trace", "utilization_pct"]).groups.keys())
    configurations = [c for c in POLICY_ORDER if c in set(frame["configuration"])]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(groups) + 3), 3.4))
    x = np.arange(len(groups))
    width = 0.8 / max(len(configurations), 1)
    for index, configuration in enumerate(configurations):
        values = []
        for trace, utilization in groups:
            match = frame[(frame["trace"] == trace)
                          & (frame["utilization_pct"] == utilization)
                          & (frame["configuration"] == configuration)]
            values.append(match["prediction_coverage"].iloc[0] * 100 if not match.empty else np.nan)
        offset = (index - (len(configurations) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width * 0.92,
                      color=POLICY_COLOR.get(configuration, SERIES_COLORS[index]),
                      label=configuration)
        if len(groups) <= 3:
            label_bars(ax, bars, values, "{:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t.replace('msr_', '').replace('systor_', '')}\n{u:.0f}%"
                        for t, u in groups], fontsize=8)
    ax.set_ylabel("Writes with a temperature (%)")
    ax.set_title("Prediction coverage: writes the policy could classify")
    ax.legend(ncols=2)
    save(fig, "20_prediction_coverage.png")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def write_final_tables(ablation: pd.DataFrame, model_comparison: pd.DataFrame,
                       classification: pd.DataFrame) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    if not ablation.empty:
        ablation.to_csv(FINAL_DIR / "final_metrics.csv", index=False)
        print(f"  wrote {(FINAL_DIR / 'final_metrics.csv').relative_to(REPO_ROOT)}")
    if not model_comparison.empty:
        model_comparison.to_csv(FINAL_DIR / "model_metrics.csv", index=False)
        print(f"  wrote {(FINAL_DIR / 'model_metrics.csv').relative_to(REPO_ROOT)}")
    if not classification.empty:
        classification.to_csv(FINAL_DIR / "classification_metrics.csv", index=False)
        print(f"  wrote {(FINAL_DIR / 'classification_metrics.csv').relative_to(REPO_ROOT)}")
    inventory = RESULTS_DIR / "dataset_inventory.csv"
    if inventory.is_file():
        pd.read_csv(inventory).to_csv(FINAL_DIR / "dataset_inventory.csv", index=False)
        print(f"  wrote {(FINAL_DIR / 'dataset_inventory.csv').relative_to(REPO_ROOT)}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-dataset-plots", action="store_true",
                        help="skip the figures that re-read the normalized traces")
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_matplotlib()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    statistics = load_dataset_statistics()
    trace_ids = sorted(statistics["trace_id"]) if not statistics.empty else []

    print("\nDataset figures")
    plot_dataset_composition(statistics)
    plot_write_percentage(statistics)
    if not args.skip_dataset_plots and trace_ids:
        plot_rewrite_interval_distribution(trace_ids)
        plot_lba_popularity(trace_ids)
        plot_cumulative_writes(trace_ids)
        plot_request_rate(trace_ids)

    model_comparison = (pd.read_csv(MODEL_COMPARISON_CSV)
                        if MODEL_COMPARISON_CSV.is_file() else pd.DataFrame())
    classification = (pd.read_csv(CLASSIFICATION_CSV)
                      if CLASSIFICATION_CSV.is_file() else pd.DataFrame())

    print("\nModel figures")
    if model_comparison.empty:
        print("  results/ml/model_comparison.csv not found, skipping model figures")
    else:
        plot_hot_cold_distribution(model_comparison)
        plot_prediction_quality(model_comparison)
        plot_classification_comparison(model_comparison)
        plot_transfer_comparison(model_comparison)
    plot_training_curves(load_model_metadata())
    if not classification.empty:
        plot_confusion_matrices(classification)

    print("\nSimulator figures and tables")
    if not COMPARISON_CSV.is_file():
        print(f"  {COMPARISON_CSV.relative_to(REPO_ROOT)} not found; run the simulate stage")
        ablation = pd.DataFrame()
    else:
        comparison = pd.read_csv(COMPARISON_CSV)
        ablation = build_ablation(comparison)
        if not ablation.empty:
            METRICS_DIR.mkdir(parents=True, exist_ok=True)
            ablation.to_csv(ABLATION_CSV, index=False)
            print(f"  wrote {ABLATION_CSV.relative_to(REPO_ROOT)}  ({len(ablation)} rows)")
            plot_policy_metric(ablation, "waf", "Write amplification factor",
                               "WAF by placement policy", "15_waf_by_policy.png", "{:.3f}")
            plot_policy_metric(ablation, "valid_blocks_migrated", "Valid blocks migrated",
                               "Valid-block migrations by policy",
                               "16_valid_migrations.png", "{:,.0f}")
            plot_policy_metric(ablation, "gc_bytes_copied", "GC bytes copied",
                               "Bytes copied by garbage collection",
                               "17a_gc_bytes.png", "{:,.0f}")
            plot_policy_metric(ablation, "gc_count", "GC invocations",
                               "Garbage-collection invocations",
                               "17b_gc_count.png", "{:,.0f}")
            plot_policy_metric(ablation, "physical_bytes_written", "Physical bytes written",
                               "Physical bytes written", "17c_physical_bytes.png", "{:,.0f}")
            plot_waf_vs_utilization(ablation)
            plot_improvement(ablation)
            plot_prediction_coverage(ablation)

    write_final_tables(ablation, model_comparison, classification)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
