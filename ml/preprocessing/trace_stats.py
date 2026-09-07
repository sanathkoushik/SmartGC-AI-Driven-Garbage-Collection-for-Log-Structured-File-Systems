"""Phase 2 - trace statistics report.

Consumes a normalized trace and writes
``results/metrics/trace_stats_<name>.csv`` (one summary row) plus a companion
``..._hist.csv`` rewrite-interval histogram.  These numbers are the evidence
base for the workload-drift discussion in Phase 4b and for choosing
over-provisioning points in Phase 7.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ml.common.config import repo_path

_HIST_EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, np.inf]


def rewrite_intervals(df: pd.DataFrame) -> np.ndarray:
    """Gap, in write events, between consecutive writes to the same LBA."""
    w = df[df["operation"] == "W"].reset_index(drop=True)
    w = w.assign(widx=np.arange(len(w)))
    gaps = w.groupby("lba")["widx"].diff().dropna().to_numpy()
    return gaps.astype(np.int64)


def compute_stats(df: pd.DataFrame, name: str) -> dict:
    n = len(df)
    writes = df[df["operation"] == "W"]
    reads = df[df["operation"] == "R"]
    nw, nr = len(writes), len(reads)

    per_lba = writes.groupby("lba").size().sort_values(ascending=False)
    top20 = max(1, int(round(0.20 * len(per_lba)))) if len(per_lba) else 0
    hot_share = float(per_lba.iloc[:top20].sum() / nw) if nw and top20 else 0.0

    gaps = rewrite_intervals(df)
    stats = {
        "trace_name": name,
        "events": n,
        "writes": nw,
        "reads": nr,
        "write_ratio": round(nw / n, 6) if n else 0.0,
        "unique_lbas": int(df["lba"].nunique()),
        "unique_written_lbas": int(writes["lba"].nunique()),
        "rewrites": int(len(gaps)),
        "rewrite_fraction": round(len(gaps) / nw, 6) if nw else 0.0,
        "rewrite_interval_mean": round(float(gaps.mean()), 4) if len(gaps) else 0.0,
        "rewrite_interval_median": float(np.median(gaps)) if len(gaps) else 0.0,
        "rewrite_interval_p90": float(np.percentile(gaps, 90)) if len(gaps) else 0.0,
        "top20pct_lba_write_share": round(hot_share, 6),
        "mean_writes_per_lba": round(float(per_lba.mean()), 4) if len(per_lba) else 0.0,
        "max_writes_per_lba": int(per_lba.iloc[0]) if len(per_lba) else 0,
    }
    return stats, gaps


def histogram(gaps: np.ndarray, name: str) -> pd.DataFrame:
    counts, _ = np.histogram(gaps, bins=_HIST_EDGES) if len(gaps) else (np.zeros(len(_HIST_EDGES) - 1, int), None)
    labels = []
    for lo, hi in zip(_HIST_EDGES[:-1], _HIST_EDGES[1:]):
        labels.append(f"[{int(lo)},{'inf' if np.isinf(hi) else int(hi)})")
    total = counts.sum() or 1
    return pd.DataFrame({
        "trace_name": name,
        "interval_bucket": labels,
        "count": counts.astype(int),
        "fraction": np.round(counts / total, 6),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description="SmartGC trace statistics report")
    ap.add_argument("--input", required=True, help="normalized trace CSV")
    ap.add_argument("--name", help="trace name (default: input stem)")
    ap.add_argument("--outdir", default=repo_path("results", "metrics"))
    args = ap.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.input))[0]
    df = pd.read_csv(args.input)
    df.columns = [c.strip().lower() for c in df.columns]

    stats, gaps = compute_stats(df, name)
    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, f"trace_stats_{name}.csv")
    hist_path = os.path.join(args.outdir, f"trace_stats_{name}_hist.csv")

    pd.DataFrame([stats]).to_csv(summary_path, index=False)
    histogram(gaps, name).to_csv(hist_path, index=False)

    print(f"[trace_stats] {name}: {stats['events']:,} events, "
          f"{stats['unique_lbas']:,} unique LBAs, "
          f"write_ratio={stats['write_ratio']}, "
          f"rewrite_interval_median={stats['rewrite_interval_median']}, "
          f"top20%_share={stats['top20pct_lba_write_share']}")
    print(f"[trace_stats] wrote {summary_path}")
    print(f"[trace_stats] wrote {hist_path}")


if __name__ == "__main__":
    main()
