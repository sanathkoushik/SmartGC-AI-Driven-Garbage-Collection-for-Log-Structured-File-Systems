"""Phase 7 - comparative plots for the writeup.

Generates, into ``results/plots/``:

* ``ladder_waf.png``          - WAF across the Phase 4a baseline ladder (synthetic, fixed trigger)
* ``waf_vs_op.png``           - WAF vs over-provisioning ratio, one line per policy
* ``gc_migration_tail.png``   - P50/P95/P99 blocks migrated per GC invocation, per policy
* ``drift_timeline.png``      - per-window KL divergence with the drift threshold and flags
* ``accuracy_vs_cost.png``    - WAF vs parameter count for the learning rungs

All inputs are the CSV/JSON artefacts written by the rest of the pipeline; this
script never re-runs a simulation.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.common.config import load_config, get, repo_path

# Okabe-Ito colourblind-safe categorical set, assigned to policies in ladder
# order and never cycled.
LADDER = ["MIXED", "RULE_BASED", "SUP_LIKE", "STAT_ML", "LSTM_SMARTGC", "LSTM_ATTN_SMARTGC"]
COLORS = dict(zip(LADDER, ["#999999", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00"]))
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
PLOTS = repo_path("results", "plots")


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_ladder_waf(matrix_csv: str) -> None:
    df = pd.read_csv(matrix_csv)
    df = df[(df["workload_name"] == "synthetic_zipf") & (df["learned_trigger_enabled"] == 0)]
    if df.empty:
        return
    df = df.set_index("placement_policy").reindex([p for p in LADDER if p in set(df["placement_policy"])])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = np.arange(len(df))
    ax.barh(y, df["waf"], color=[COLORS[p] for p in df.index], height=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle="--", zorder=2)
    for i, v in enumerate(df["waf"]):
        ax.text(v + 0.0008, i, f"{v:.4f}", va="center", color=INK, fontsize=9)
    _style(ax, "Write Amplification across the baseline ladder\n(synthetic Zipf, fixed GC trigger)",
           "WAF (physical / logical bytes)", "")
    ax.set_xlim(left=min(0.999, df["waf"].min() - 0.003))
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "ladder_waf.png"), dpi=150)
    plt.close(fig)


def plot_waf_vs_op(op_csv: str) -> None:
    if not os.path.exists(op_csv):
        return
    df = pd.read_csv(op_csv)
    # recover the OP ratio from the workload_name suffix "..._op<ratio>"
    df["op_ratio"] = df["workload_name"].str.extract(r"_op([0-9.]+)$").astype(float)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for p in LADDER:
        sub = df[df["placement_policy"] == p].sort_values("op_ratio")
        if sub.empty:
            continue
        ax.plot(sub["op_ratio"], sub["waf"], marker="o", markersize=5,
                color=COLORS[p], linewidth=2, label=p, zorder=3)
        ax.text(sub["op_ratio"].iloc[-1], sub["waf"].iloc[-1], f" {p}",
                color=COLORS[p], fontsize=8, va="center")
    _style(ax, "WAF vs over-provisioning (live-set / capacity)", "live-set / capacity ratio", "WAF")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "waf_vs_op.png"), dpi=150)
    plt.close(fig)


def plot_gc_tail(matrix_csv: str) -> None:
    df = pd.read_csv(matrix_csv)
    df = df[(df["workload_name"] == "synthetic_zipf") & (df["learned_trigger_enabled"] == 0)]
    if df.empty:
        return
    df = df.set_index("placement_policy").reindex([p for p in LADDER if p in set(df["placement_policy"])])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(df))
    w = 0.26
    for k, (pct, shade) in enumerate({"gc_migrated_p50": 0.9, "gc_migrated_p95": 0.6, "gc_migrated_p99": 0.35}.items()):
        ax.bar(x + (k - 1) * w, df[pct], width=w, zorder=3,
               color=[COLORS[p] for p in df.index], alpha=shade,
               label=pct.replace("gc_migrated_", "").upper())
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=20, ha="right")
    _style(ax, "GC-induced work: blocks migrated per GC invocation\n(synthetic Zipf, fixed trigger)",
           "", "blocks migrated / GC")
    ax.legend(frameon=False, fontsize=8, title="percentile", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "gc_migration_tail.png"), dpi=150)
    plt.close(fig)


def plot_drift_timeline(drift_json: str) -> None:
    if not os.path.exists(drift_json):
        return
    st = json.load(open(drift_json, encoding="utf-8"))
    tl = st.get("timeline", [])
    if not tl:
        return
    ev = [w["window_end_event"] for w in tl]
    kl = [w["kl_vs_prev"] for w in tl]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(ev, kl, marker="o", markersize=5, color="#0072B2", linewidth=2, zorder=3, label="symmetric KL")
    ax.axhline(st["kl_threshold"], color="#D55E00", linestyle="--", linewidth=1.2,
               label=f"threshold {st['kl_threshold']}", zorder=2)
    flg = [(e, k) for e, k, w in zip(ev, kl, tl) if w["flagged"]]
    if flg:
        ax.scatter(*zip(*flg), color="#D55E00", s=90, zorder=4, label="drift flagged")
    _style(ax, f"Predicted-interval drift timeline ({st.get('trace_name', '')})",
           "write event (window end)", "KL(prev window || this window)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "drift_timeline.png"), dpi=150)
    plt.close(fig)


def plot_accuracy_vs_cost(cost_csv: str) -> None:
    if not os.path.exists(cost_csv):
        return
    df = pd.read_csv(cost_csv)
    col = "waf" if "waf" in df.columns else "waf_synthetic_zipf"
    df = df[df.get("dataset", "synthetic_zipf") == "synthetic_zipf"] if "dataset" in df.columns else df
    if df.empty or df[col].isna().all():
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in df.iterrows():
        c = COLORS.get(r["model"], INK)
        ax.scatter(r["param_count"], r[col], s=110, color=c, zorder=3)
        ax.annotate(f"  {r['model']}\n  {r['latency_ms_per_batch']:.2f} ms/batch",
                    (r["param_count"], r[col]), fontsize=8, color=INK, va="center")
    ax.set_xscale("log")
    _style(ax, "Accuracy vs cost: WAF against parameter count\n(synthetic Zipf, fixed trigger)",
           "parameter count (log scale)", "WAF")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "accuracy_vs_cost.png"), dpi=150)
    plt.close(fig)


def plot_real_ladder(matrix_csv: str) -> None:
    """Before/after on real data: WAF for the ladder baselines vs the multi-stream
    ML policy on the UMass SPC Financial1 trace at a stressed (~1.5x) OP point,
    with the synthetic-Zipf equivalents alongside for context."""
    df = pd.read_csv(matrix_csv)
    df = df[df["learned_trigger_enabled"] == 0]
    real_pols = ["MIXED", "RULE_BASED", "LSTM_ATTN_SMARTGC"]
    real_key = next((k for k in ("financial1_op1.5", "financial1") if k in set(df["workload_name"])), None)
    if real_key is None:
        return
    groups = [("synthetic_zipf", "synthetic Zipf"), (real_key, "real (SPC Financial1)")]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(groups))
    w = 0.26
    ymax = 1.0
    for k, pol in enumerate(real_pols):
        vals = []
        for wl, _ in groups:
            sub = df[(df["workload_name"] == wl) & (df["placement_policy"] == pol)]
            v = float(sub["waf"].iloc[0]) if len(sub) else np.nan
            vals.append(v)
            if np.isfinite(v):
                ymax = max(ymax, v)
        bars = ax.bar(x + (k - 1) * w, vals, width=w, color=COLORS[pol], zorder=3, label=pol)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8, color=INK)
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in groups])
    ax.set_ylim(0.99, ymax * 1.03)
    _style(ax, "Before / after on real data: WAF, ladder baselines vs multi-stream ML\n"
               "(fixed GC trigger; real trace at ~1.5x over-provisioning)", "", "WAF")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "real_ladder_waf.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="SmartGC Phase 7 plot generation")
    ap.add_argument("--matrix", default=repo_path("results", "metrics", "matrix_results.csv"))
    ap.add_argument("--op-sweep", default=repo_path("results", "metrics", "op_sweep.csv"))
    ap.add_argument("--cost", default=repo_path("results", "metrics", "model_cost.csv"))
    ap.add_argument("--drift-name", default=str(get(cfg, "evaluation.drift_scenario_name", "synthetic_drift")))
    ap.add_argument("--real-name", default=str(get(cfg, "evaluation.real_trace_name", "financial1")))
    args = ap.parse_args()

    os.makedirs(PLOTS, exist_ok=True)
    drift_json = repo_path("results", "metrics", f"drift_status_{BEST_PREFIX}{args.drift_name}.json")
    made = []
    if os.path.exists(args.matrix):
        plot_ladder_waf(args.matrix); made.append("ladder_waf.png")
        plot_gc_tail(args.matrix); made.append("gc_migration_tail.png")
        plot_real_ladder(args.matrix); made.append("real_ladder_waf.png")
    plot_waf_vs_op(args.op_sweep); made.append("waf_vs_op.png")
    plot_accuracy_vs_cost(args.cost); made.append("accuracy_vs_cost.png")
    for cand in (drift_json,
                 repo_path("results", "metrics", f"drift_status_{args.drift_name}.json")):
        if os.path.exists(cand):
            plot_drift_timeline(cand); made.append("drift_timeline.png"); break

    print("[plots] wrote: " + ", ".join(f"results/plots/{m}" for m in made))


BEST_PREFIX = "LSTM_ATTN_SMARTGC_"

if __name__ == "__main__":
    main()
