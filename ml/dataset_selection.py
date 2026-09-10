"""Workload selection rules, frozen before any model is trained or evaluated.

The point of this module is that *nothing here may depend on a result*. The
eligibility thresholds and the rule that picks the held-out targets are
mechanical, are recorded in docs/methodology.md, and are applied by
``experiments/analyze_datasets.py`` before the first training run. Cherry-picking
the workload that flatters SmartGC afterwards would be the single easiest way to
produce a dishonest paper, so the choice is taken out of human hands.

Eligibility (a trace must satisfy all of these to be a candidate)
----------------------------------------------------------------
1. ``write_blocks >= MIN_WRITE_BLOCKS`` - enough write traffic to fill the
   simulated device several times over and actually trigger cleaning.
2. ``rewrite_ratio >= MIN_REWRITE_RATIO`` - a workload that mostly writes each
   block once has no rewrite intervals to predict, so the research question is
   undefined on it.
3. ``unique_written_lbas >= MIN_UNIQUE_WRITTEN_LBAS`` - a working set larger
   than a handful of segments, otherwise placement is trivial.
4. ``duration_seconds >= MIN_DURATION_SECONDS`` - long enough that a
   chronological 70/15/15 split leaves a meaningful test period.

Role assignment
---------------
Candidates are sorted by ``trace_id`` (a stable, content-independent key). The
two MSR candidates at the middle of that alphabetical order become the held-out
**targets**; every other MSR candidate becomes a **pretraining** trace. Targets
are excluded from pretraining entirely. SYSTOR candidates are never used for
pretraining or fine-tuning selection - they are the external-generalization set.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

MIN_WRITE_BLOCKS = 200_000
MIN_REWRITE_RATIO = 0.30
MIN_UNIQUE_WRITTEN_LBAS = 5_000
MIN_DURATION_SECONDS = 3_600.0

#: Number of MSR workloads held out as fine-tuning / evaluation targets.
NUM_TARGETS = 2

SELECTION_CRITERIA: dict[str, Any] = {
    "min_write_blocks": MIN_WRITE_BLOCKS,
    "min_rewrite_ratio": MIN_REWRITE_RATIO,
    "min_unique_written_lbas": MIN_UNIQUE_WRITTEN_LBAS,
    "min_duration_seconds": MIN_DURATION_SECONDS,
    "num_targets": NUM_TARGETS,
    "target_rule": ("MSR candidates sorted by trace_id; the NUM_TARGETS traces at the "
                    "middle of that order are held out. Independent of any measured "
                    "model or WAF result."),
}


@dataclass(frozen=True)
class EligibilityResult:
    trace_id: str
    dataset: str
    eligible: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = "; ".join(self.reasons)
        return data


@dataclass(frozen=True)
class WorkloadRoles:
    """Which trace plays which role in the experiment."""

    pretrain: tuple[str, ...]
    targets: tuple[str, ...]
    external: tuple[str, ...]
    excluded: tuple[EligibilityResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pretrain": list(self.pretrain),
            "targets": list(self.targets),
            "external": list(self.external),
            "excluded": [item.to_dict() for item in self.excluded],
            "criteria": SELECTION_CRITERIA,
        }


def check_eligibility(statistics: Mapping[str, Any]) -> EligibilityResult:
    """Apply the frozen eligibility thresholds to one trace's statistics."""
    trace_id = str(statistics.get("trace_id", "?"))
    dataset = str(statistics.get("dataset", "?"))
    reasons: list[str] = []

    write_blocks = int(statistics.get("write_blocks", 0) or 0)
    if write_blocks < MIN_WRITE_BLOCKS:
        reasons.append(f"write_blocks {write_blocks:,} < {MIN_WRITE_BLOCKS:,}")

    rewrite_ratio = float(statistics.get("rewrite_ratio", 0.0) or 0.0)
    if rewrite_ratio < MIN_REWRITE_RATIO:
        reasons.append(f"rewrite_ratio {rewrite_ratio:.3f} < {MIN_REWRITE_RATIO}")

    unique_written = int(statistics.get("unique_written_lbas", 0) or 0)
    if unique_written < MIN_UNIQUE_WRITTEN_LBAS:
        reasons.append(f"unique_written_lbas {unique_written:,} < {MIN_UNIQUE_WRITTEN_LBAS:,}")

    duration = float(statistics.get("duration_seconds", 0.0) or 0.0)
    if duration < MIN_DURATION_SECONDS:
        reasons.append(f"duration_seconds {duration:,.0f} < {MIN_DURATION_SECONDS:,.0f}")

    return EligibilityResult(trace_id=trace_id, dataset=dataset,
                             eligible=not reasons, reasons=tuple(reasons))


def assign_roles(all_statistics: Sequence[Mapping[str, Any]],
                 num_targets: int = NUM_TARGETS) -> WorkloadRoles:
    """Split eligible traces into pretraining, target and external sets."""
    results = [check_eligibility(statistics) for statistics in all_statistics]
    eligible = sorted((r for r in results if r.eligible), key=lambda r: r.trace_id)
    excluded = tuple(sorted((r for r in results if not r.eligible), key=lambda r: r.trace_id))

    msr = [r.trace_id for r in eligible if r.dataset == "msr"]
    external = tuple(r.trace_id for r in eligible if r.dataset != "msr")

    if not msr:
        return WorkloadRoles((), (), external, excluded)

    # Middle of the alphabetical order: a fixed, result-independent position.
    count = min(num_targets, max(1, len(msr) - 1)) if len(msr) > 1 else 0
    if count == 0:
        return WorkloadRoles(tuple(msr), (), external, excluded)

    start = max(0, (len(msr) - count) // 2)
    targets = tuple(msr[start:start + count])
    pretrain = tuple(name for name in msr if name not in set(targets))
    return WorkloadRoles(pretrain=pretrain, targets=targets, external=external, excluded=excluded)
