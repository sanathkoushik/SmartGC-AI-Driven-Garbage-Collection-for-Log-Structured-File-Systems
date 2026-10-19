#!/usr/bin/env python3
"""Figures for the classical-ML vs. LSTM vs. rule-based comparison.

    python experiments/plot_ml_comparison.py

Reads ``results/final/full_model_comparison.csv`` (prediction quality) and
``results/final/waf_full_comparison.csv`` (write amplification), both written by
``experiments/ml_comparison.py`` / ``experiments/simulate_classical.py``, and
writes ``results/plots/17_ml_prediction_comparison.png``,
``18_ml_waf_comparison.png`` and ``19_prediction_vs_waf.png``. Also writes the
summary tables ``results/final/ml_prediction_summary.csv`` and
``results/final/ml_waf_summary.csv`` that the README quotes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_results import (                                # noqa: E402
    SERIES_COLORS, TEXT_SECONDARY, configure_matplotlib, label_bars, plt, save,
)

FINAL = REPO_ROOT / "results" / "final"
PREDICTION_CSV = FINAL / "full_model_comparison.csv"
WAF_CSV = FINAL / "waf_full_comparison.csv"

#: (display name, prediction-table model_type, WAF-table configuration)
POLICIES = [
    ("Median baseline", "median_interval", None),
    ("Rule-based", None, "RULE_BASED"),
    ("LSTM scratch", "scratch", "LSTM scratch"),
    ("LSTM pretrained", "pretrained", "LSTM pretrained"),
    ("LSTM fine-tuned", "pretrained_finetuned", "LSTM pretrained_finetuned"),
    ("Linear regression", "linear_regression", "linear_regression"),
    ("Random forest", "random_forest", "random_forest"),
    ("XGBoost", "xgboost", "xgboost"),
]
COLOR = {name: SERIES_COLORS[i % len(SERIES_COLORS)] for i, (name, _, _) in enumerate(POLICIES)}
WORKLOADS = ["msr_rsrch_0", "msr_src2_0", "systor_lun0"]


def prediction_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, model_type, _ in POLICIES:
        if model_type is None:
            continue
        subset = frame[frame["model_type"] == model_type]
        for _, row in subset.iterrows():
            rows.append({"policy": name, "workload": row["target_trace"],
                         "mae_log1p": row["test_mae_log1p"], "f1": row["f1"],
                         "roc_auc": row["roc_auc"], "mae_us": row["test_mae_us"],
                         "rmse_us": row["test_rmse_us"],
                         "training_seconds": row["training_seconds"]})
    return pd.DataFrame(rows)


def waf_table(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["policy"] != "MIXED"].copy()
    frame["utilization_pct"] = (frame["target_utilization"] * 100).round(0).astype(int)
    rows = []
    for name, _, configuration in POLICIES:
        if configuration is None:
            continue
        for _, row in frame[frame["configuration"] == configuration].iterrows():
            rows.append({"policy": name, "workload": row["trace"],
                         "utilization_pct": row["utilization_pct"],
                         "waf": row["waf"], "change_vs_mixed_pct": row["waf_change_vs_mixed_pct"]})
    return pd.DataFrame(rows)


def label_below(ax, bars, values, fmt: str) -> None:
    """Value labels for bars that hang below zero: placed past the bar's far end."""
    for bar, value in zip(bars, values):
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, -3), ha="center", va="top",
                    fontsize=7.5, color=TEXT_SECONDARY)


def grouped_bars(ax, table: pd.DataFrame, value: str, policies: list[str], fmt: str) -> None:
    width = 0.8 / len(policies)
    for index, name in enumerate(policies):
        values = [table[(table["policy"] == name) & (table["workload"] == w)][value].mean()
                  for w in WORKLOADS]
        offset = (index - (len(policies) - 1) / 2) * width
        bars = ax.bar(np.arange(len(WORKLOADS)) + offset, values, width * 0.92,
                      color=COLOR[name], label=name)
        (label_below if max(values) <= 0 else label_bars)(ax, bars, values, fmt)
    ax.set_xticks(np.arange(len(WORKLOADS)))
    ax.set_xticklabels(WORKLOADS)


def plot_prediction(pred: pd.DataFrame) -> None:
    names = [p for p in pred["policy"].unique() if p != "Median baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    grouped_bars(axes[0], pred, "mae_log1p", names, "{:.2f}")
    axes[0].set_title("Prediction error (log1p MAE, lower is better)")
    axes[0].set_ylabel("MAE of log1p(interval)")
    axes[0].set_ylim(0, 2.1)
    grouped_bars(axes[1], pred, "f1", names, "{:.2f}")
    axes[1].set_title("Hot/cold F1 (higher is better)")
    axes[1].set_ylabel("F1 on the held-out test split")
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(ncol=2, loc="upper left")
    save(fig, "17_ml_prediction_comparison.png")


def plot_waf(waf: pd.DataFrame) -> None:
    names = list(waf["policy"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    grouped_bars(axes[0], waf, "change_vs_mixed_pct", names, "{:.1f}")
    axes[0].set_title("WAF change vs MIXED, mean of 70/80/85% utilization\n(more negative is better)")
    axes[0].set_ylabel("% change in WAF")
    axes[0].axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    axes[0].set_ylim(-13.5, 0.5)
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))

    overall = waf.groupby("policy")["change_vs_mixed_pct"].mean().sort_values()
    bars = axes[1].barh(overall.index, overall.values, color=[COLOR[n] for n in overall.index])
    for bar, value in zip(bars, overall.values):
        axes[1].annotate(f"{value:.2f}%", (value, bar.get_y() + bar.get_height() / 2),
                         textcoords="offset points", xytext=(-4, 0), ha="right", va="center",
                         fontsize=7.5, color=TEXT_SECONDARY)
    axes[1].set_xlim(overall.min() * 1.22, 0)
    axes[1].invert_yaxis()
    axes[1].set_title("Overall mean, all 9 workload x utilization cells")
    axes[1].set_xlabel("% change in WAF vs MIXED")
    save(fig, "18_ml_waf_comparison.png")


def plot_prediction_vs_waf(pred: pd.DataFrame, waf: pd.DataFrame) -> dict[str, float]:
    """Does a better predictor give lower WAF? One point per (policy, workload).

    Correlation is computed *within* each workload. Pooling across workloads is
    misleading: the MSR workloads differ from SYSTOR in both F1 and attainable WAF
    gain, so a pooled coefficient mostly measures which workload a point is from.
    A positive rho means higher F1 goes with a *higher* (worse) WAF change.
    """
    waf_mean = waf.groupby(["policy", "workload"], as_index=False)["change_vs_mixed_pct"].mean()
    joined = pred.merge(waf_mean, on=["policy", "workload"])
    rho = {workload: float(group["f1"].corr(group["change_vs_mixed_pct"], method="spearman"))
           for workload, group in joined.groupby("workload")}

    fig, ax = plt.subplots(figsize=(10.5, 5))
    markers = dict(zip(WORKLOADS, ["o", "s", "^"]))
    for _, row in joined.iterrows():
        ax.scatter(row["f1"], row["change_vs_mixed_pct"], s=70, color=COLOR[row["policy"]],
                   marker=markers[row["workload"]], edgecolor="white", linewidth=0.6)
    for name in joined["policy"].unique():
        ax.scatter([], [], color=COLOR[name], label=name, s=50)
    for workload, marker in markers.items():
        ax.scatter([], [], color=TEXT_SECONDARY, marker=marker, s=50,
                   label=f"{workload}  (rho {rho[workload]:+.2f}, n=6)")
    ax.set_xlabel("Hot/cold F1 on held-out test split (higher = better predictor)")
    ax.set_ylabel("Mean WAF change vs MIXED, % (lower = better placement)")
    ax.set_title("Prediction quality vs placement quality")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              title="Spearman rho of F1 vs WAF change")
    save(fig, "19_prediction_vs_waf.png")
    return rho


def main() -> int:
    for path in (PREDICTION_CSV, WAF_CSV):
        if not path.is_file():
            print(f"missing {path.relative_to(REPO_ROOT)}; run experiments/ml_comparison.py "
                  "and experiments/simulate_classical.py first", file=sys.stderr)
            return 1
    configure_matplotlib()
    pred = prediction_table(pd.read_csv(PREDICTION_CSV))
    waf = waf_table(pd.read_csv(WAF_CSV))
    waf = waf[waf["workload"].isin(WORKLOADS)]

    plot_prediction(pred)
    plot_waf(waf)
    rho = plot_prediction_vs_waf(pred, waf)

    pred.round(6).to_csv(FINAL / "ml_prediction_summary.csv", index=False)
    summary = (waf.groupby("policy")["change_vs_mixed_pct"].agg(["mean", "min", "max"])
               .rename(columns={"mean": "mean_change_pct", "min": "best_pct", "max": "worst_pct"}))
    wins = waf.pivot_table(index=["workload", "utilization_pct"], columns="policy",
                           values="change_vs_mixed_pct").idxmin(axis=1).value_counts()
    summary["cells_won"] = wins.reindex(summary.index).fillna(0).astype(int)
    summary.sort_values("mean_change_pct").round(3).to_csv(FINAL / "ml_waf_summary.csv")
    for workload, value in rho.items():
        print(f"Spearman(F1, WAF change) within {workload} (n=6 policies): {value:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
