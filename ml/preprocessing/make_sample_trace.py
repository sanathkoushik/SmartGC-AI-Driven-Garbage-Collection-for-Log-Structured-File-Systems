"""Generate an MSR-Cambridge-format *stand-in* trace.

The real MSR Cambridge / FIU block traces are large and distributed under the
SNIA IOTTA repository terms, so they are not vendored in this repo.  This
script emits a file in the exact MSR on-disk layout
(``Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime``, header-less,
Windows FILETIME timestamps, byte offsets) so the Phase 2 ``--source msr``
ingestion path, feature engineering and the benchmark matrix can run
end-to-end.  Replace ``data/raw/<name>.csv`` with a genuine trace of the same
name for a real-data result; nothing else in the pipeline changes.

The stand-in mixes a small skewed hot set, a long cold tail, and occasional
sequential runs, and includes a mid-stream shift in the hot set so the Phase 4b
drift detector has something to find.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from ml.common.config import load_config, get, repo_path

# 1601-01-01 -> 1970-01-01 in 100 ns ticks; MSR timestamps are FILETIME.
_FILETIME_EPOCH_OFFSET = 116_444_736_000_000_000


def generate(n_events: int, working_set: int, block_size: int, seed: int,
             read_ratio: float, hot_ratio: float, hot_traffic: float,
             drift_at: float) -> list[str]:
    rng = np.random.default_rng(seed)
    n_hot = max(1, int(round(hot_ratio * working_set)))

    hot_a = rng.choice(working_set, size=n_hot, replace=False)
    # Post-drift hot set: half the addresses rotate out.
    hot_b = rng.choice(working_set, size=n_hot, replace=False)
    drift_idx = int(drift_at * n_events)

    base_ft = _FILETIME_EPOCH_OFFSET + 130_000_000_000_000_000  # ~2013
    rows: list[str] = []
    t = base_ft
    run_left, run_lba = 0, 0

    for i in range(n_events):
        hot = hot_a if i < drift_idx else hot_b
        t += int(rng.integers(2_000, 60_000))  # 0.2-6 ms between IOs, in ticks

        if run_left > 0:
            lba = run_lba
            run_lba += 1
            run_left -= 1
        elif rng.random() < hot_traffic:
            lba = int(hot[rng.integers(0, len(hot))])
        else:
            lba = int(rng.integers(0, working_set))
            if rng.random() < 0.04:  # kick off a short sequential run
                run_left = int(rng.integers(3, 16))
                run_lba = lba + 1

        op = "Read" if rng.random() < read_ratio else "Write"
        size = block_size * int(rng.integers(1, 9))
        offset = lba * block_size
        resp = int(rng.integers(1_000, 200_000))
        rows.append(f"{t},smartgc-host,0,{op},{offset},{size},{resp}")

    return rows


def main() -> None:
    cfg = load_config()
    block_size = int(get(cfg, "simulator.block_size_bytes", 4096))
    seed = int(get(cfg, "random_seed", 42))

    ap = argparse.ArgumentParser(description="MSR-format stand-in trace generator")
    ap.add_argument("--name", default=str(get(cfg, "evaluation.real_trace_name", "msr_cambridge_src1")))
    ap.add_argument("--events", type=int, default=40_000)
    ap.add_argument("--working-set", type=int, default=1_200)
    ap.add_argument("--read-ratio", type=float, default=0.35)
    ap.add_argument("--hot-ratio", type=float, default=float(get(cfg, "synthetic_workload.hot_ratio", 0.20)))
    ap.add_argument("--hot-traffic", type=float, default=float(get(cfg, "synthetic_workload.hot_traffic_ratio", 0.80)))
    ap.add_argument("--drift-at", type=float, default=0.55, help="fraction of the stream where the hot set rotates")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    out = args.output or repo_path("data", "raw", f"{args.name}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = generate(args.events, args.working_set, block_size, seed,
                    args.read_ratio, args.hot_ratio, args.hot_traffic, args.drift_at)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"[make_sample_trace] STAND-IN trace ({len(rows):,} rows) -> {out}")
    print("[make_sample_trace] replace with a genuine SNIA IOTTA / MSR Cambridge "
          "file of the same name for a real-data result.")


if __name__ == "__main__":
    main()
