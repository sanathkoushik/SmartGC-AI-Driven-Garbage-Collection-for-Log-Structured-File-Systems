#!/usr/bin/env python3
"""Scientific and leakage audit of the artefacts an actual run produced.

    python experiments/audit.py
    python experiments/audit.py --strict     # non-zero exit on any failure

`tests/test_leakage.py` checks the *code* obeys the methodology. This checks the
*outputs* do: it opens the CSVs, checkpoints and manifests on disk and verifies
the properties the write-up depends on. Both are needed - a rule can be correct
in the module and still be bypassed by the script that calls it, which is exactly
how two defects in this project were found.

Every check names what it examined, so a pass is evidence rather than assertion.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "results"
MODELS = REPO_ROOT / "models"


class Audit:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, bool, str]] = []

    def check(self, section: str, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((section, name, bool(ok), detail))
        return bool(ok)

    def skip(self, section: str, name: str, why: str) -> None:
        self.results.append((section, name, True, f"SKIPPED: {why}"))

    def report(self) -> int:
        current = None
        failures = 0
        for section, name, ok, detail in self.results:
            if section != current:
                print(f"\n{section}")
                current = section
            mark = "ok  " if ok else "FAIL"
            if not ok:
                failures += 1
            suffix = f"   {detail}" if detail else ""
            print(f"  [{mark}] {name}{suffix}")
        total = len(self.results)
        print(f"\n{total - failures}/{total} checks passed")
        return failures


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_provenance(audit: Audit) -> None:
    section = "Dataset provenance"
    for name in ("docs/dataset_source_decision.md", "docs/dataset.md", "data/README.md",
                 "docs/methodology.md", "results/dataset_source_comparison.csv",
                 "results/dataset_inventory.csv"):
        audit.check(section, f"{name} present", (REPO_ROOT / name).is_file())

    manifest = REPO_ROOT / "data" / "raw" / "download_manifest.json"
    if not manifest.is_file():
        audit.skip(section, "download checksums recorded", "no manifest (data not downloaded)")
        return
    files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    msr = [f for f in files if f["dataset"] == "msr" and f["name"].endswith(".csv.gz")]
    verified = [f for f in msr if f.get("md5_verified")]
    audit.check(section, "every MSR volume verified against the authors' MD5",
                bool(msr) and len(verified) == len(msr), f"{len(verified)}/{len(msr)}")
    audit.check(section, "every downloaded file has a SHA-256",
                all(f.get("sha256") for f in files), f"{len(files)} files")


def audit_selection(audit: Audit) -> None:
    section = "Workload selection"
    roles_path = RESULTS / "dataset_roles.json"
    if not roles_path.is_file():
        audit.skip(section, "roles recorded", "no dataset_roles.json")
        return
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    audit.check(section, "selection criteria recorded alongside the roles",
                "criteria" in roles)
    audit.check(section, "targets are disjoint from the pretraining set",
                set(roles["targets"]).isdisjoint(set(roles["pretrain"])),
                f"targets={roles['targets']}")
    audit.check(section, "external workloads never used for pretraining",
                set(roles["external"]).isdisjoint(set(roles["pretrain"])))
    audit.check(section, "at least one held-out target exists", bool(roles["targets"]))


def audit_models(audit: Audit) -> None:
    section = "Models"
    metadata_files = sorted(glob.glob(str(MODELS / "**" / "*.json"), recursive=True))
    if not metadata_files:
        audit.skip(section, "checkpoints present", "no models on disk")
        return

    by_target: dict[str, dict[str, dict[str, Any]]] = {}
    for path in metadata_files:
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
        name = Path(path).stem
        threshold = meta.get("hot_threshold_microseconds")
        audit.check(section, f"{name}: records a training-derived hot threshold",
                    bool(threshold and threshold > 0))
        target = meta.get("target_trace")
        pretrain = set(meta.get("pretrain_traces") or [])
        if target and pretrain:
            audit.check(section, f"{name}: target held out of pretraining",
                        target not in pretrain)
        if target:
            by_target.setdefault(target, {})[meta.get("stage", name)] = meta

    for target, variants in sorted(by_target.items()):
        lengths = {v["hyperparameters"]["sequence_length"] for v in variants.values()}
        # A variant trained at a different window length is scored on a
        # different number of sequences, hence a different chronological split.
        audit.check(section, f"{target}: all variants share one sequence length",
                    len(lengths) == 1, f"lengths={sorted(lengths)}, "
                                       f"variants={sorted(variants)}")


def audit_evaluation(audit: Audit) -> None:
    section = "Evaluation"
    path = RESULTS / "ml" / "transfer_learning_comparison.csv"
    if not path.is_file():
        audit.skip(section, "transfer comparison present", "not generated yet")
        return
    rows = read_csv(path)
    by_target: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_target.setdefault(row["target_trace"], []).append(row)

    for target, entries in sorted(by_target.items()):
        families = {row["family"] for row in entries}
        audit.check(section, f"{target}: baselines and models both present",
                    families == {"baseline", "lstm"}, f"families={sorted(families)}")
        sizes = {row["test_samples"] for row in entries}
        lengths = {row["sequence_length"] for row in entries}
        audit.check(section, f"{target}: every approach on one identical test split",
                    len(sizes) == 1 and len(lengths) == 1,
                    f"n={sorted(sizes)}, L={sorted(lengths)}")
        stages = {row["model_type"] for row in entries if row["family"] == "lstm"}
        audit.check(section, f"{target}: all three model variants scored",
                    stages == {"scratch", "pretrained", "pretrained_finetuned"},
                    f"present={sorted(stages)}")
        thresholds = {row["hot_threshold_source"] for row in entries}
        audit.check(section, f"{target}: no threshold derived from test data",
                    all("test" not in source.lower() for source in thresholds))


def audit_simulations(audit: Audit) -> None:
    section = "Simulator runs"
    path = RESULTS / "metrics" / "comparison.csv"
    if not path.is_file():
        audit.skip(section, "comparison.csv present", "not generated yet")
        return
    rows = read_csv(path)

    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["trace"], row["total_segments"]), []).append(row)

    for (trace, segments), entries in sorted(groups.items()):
        writes = {row["total_write_requests"] for row in entries}
        logical = {row["logical_bytes_written"] for row in entries}
        seeds = {row["seed"] for row in entries}
        audit.check(section,
                    f"{trace} @ {segments} segments: identical workload for all "
                    f"{len(entries)} policies",
                    len(writes) == 1 and len(logical) == 1 and len(seeds) == 1,
                    f"writes={sorted(writes)}")
        policies = {row["policy"] for row in entries}
        audit.check(section, f"{trace} @ {segments} segments: MIXED baseline present",
                    "MIXED" in policies)

    bad_identity = [row for row in rows
                    if int(row["physical_bytes_written"])
                    != int(row["logical_bytes_written"]) + int(row["gc_bytes_copied"])]
    audit.check(section, "physical == logical + gc_copied in every run",
                not bad_identity, f"{len(bad_identity)} violations of {len(rows)}")

    bad_waf = [row for row in rows
               if abs(float(row["waf"]) - int(row["physical_bytes_written"])
                      / int(row["logical_bytes_written"])) > 1e-6]
    audit.check(section, "WAF equals physical/logical in every run",
                not bad_waf, f"{len(bad_waf)} violations of {len(rows)}")

    mixed_cold = [row for row in rows
                  if row["policy"] == "MIXED" and int(row["cold_stream_writes"]) != 0]
    audit.check(section, "MIXED never opens the cold append point", not mixed_cold)

    smart_no_cold = [row for row in rows
                     if row["policy"] in ("RULE_BASED", "LSTM_SMARTGC")
                     and int(row["cold_stream_writes"]) == 0]
    audit.check(section, "every segregated policy actually used the cold log",
                not smart_no_cold, f"{len(smart_no_cold)} runs placed nothing cold")

    # A policy that classifies almost nothing is not doing hot/cold separation;
    # it is the failure mode that made the rule-based control degenerate once.
    degenerate = []
    for row in rows:
        if row["policy"] == "MIXED":
            continue
        classified = int(row["total_write_requests"]) - int(row["unpredicted_writes"])
        if classified == 0:
            continue
        cold = int(row["cold_stream_writes"])
        hot = classified - cold
        if classified and (hot / classified < 0.02 or hot / classified > 0.98):
            degenerate.append(f"{row['trace']}/{row['policy']}/{row['model_type']}"
                              f" hot={hot / classified:.1%}")
    audit.check(section, "no policy collapsed to an all-hot or all-cold labelling",
                not degenerate, "; ".join(degenerate[:4]))


def audit_manifest(audit: Audit) -> None:
    section = "Reproducibility"
    path = RESULTS / "final" / "experiment_manifest.json"
    if not path.is_file():
        audit.skip(section, "experiment manifest present", "not generated yet")
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    environment = manifest.get("environment", {})
    audit.check(section, "manifest records the git commit",
                bool(environment.get("git_commit")), environment.get("git_commit", "")[:16])
    audit.check(section, "manifest records the seed", manifest.get("seed") is not None)
    audit.check(section, "manifest records library versions",
                bool(environment.get("python_version")) and bool(environment.get("torch_version")),
                f"python {environment.get('python_version')}, torch {environment.get('torch_version')}")
    audit.check(section, "manifest records the truncation caps",
                manifest.get("max_writes_per_run") is not None
                and manifest.get("max_sequences_per_trace") is not None)
    audit.check(section, "manifest records the full configuration",
                "config" in manifest and "simulator" in manifest.get("config", {}))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if any check fails")
    args = parser.parse_args(list(argv) if argv is not None else None)

    audit = Audit()
    audit_provenance(audit)
    audit_selection(audit)
    audit_models(audit)
    audit_evaluation(audit)
    audit_simulations(audit)
    audit_manifest(audit)

    failures = audit.report()
    if failures and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
