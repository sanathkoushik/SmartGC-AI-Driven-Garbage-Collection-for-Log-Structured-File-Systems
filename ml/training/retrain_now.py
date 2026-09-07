"""Phase 4b - manual "retrain now" trigger.

Polls a drift-status file written by ``ml/inference/export.py`` and, when it says
``needs_retrain`` (or ``--force``), rebuilds the Phase 3 sequence dataset for the
trace and retrains the given model.  Online/continuous retraining is explicitly
out of scope; this is the human-in-the-loop hook.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from ml.common.config import repo_path


def main() -> None:
    ap = argparse.ArgumentParser(description="SmartGC drift-triggered retrain")
    ap.add_argument("--status", required=True, help="results/metrics/drift_status_<name>.json")
    ap.add_argument("--trace", required=True, help="normalized trace CSV to retrain on")
    ap.add_argument("--model", required=True)
    ap.add_argument("--force", action="store_true", help="retrain regardless of the flag")
    args = ap.parse_args()

    needs = args.force
    if os.path.isfile(args.status):
        with open(args.status, "r", encoding="utf-8") as fh:
            st = json.load(fh)
        needs = needs or bool(st.get("needs_retrain"))
        print(f"[retrain_now] status: needs_retrain={st.get('needs_retrain')} "
              f"last_flag_event={st.get('last_flag_event')} windows={st.get('windows')}")
    else:
        print(f"[retrain_now] no status file at {args.status}")

    if not needs:
        print("[retrain_now] nothing to do (no drift flagged; pass --force to override)")
        return

    stem = os.path.splitext(os.path.basename(args.trace))[0]
    py = sys.executable
    print(f"[retrain_now] rebuilding features for {stem} and retraining {args.model} ...")
    subprocess.check_call([py, "-m", "ml.preprocessing.features", "--input", args.trace, "--name", stem])
    subprocess.check_call([py, "-m", "ml.training.train", "--dataset", stem, "--model", args.model])
    print(f"[retrain_now] done -> ml/models/{args.model}_{stem}/")


if __name__ == "__main__":
    main()
