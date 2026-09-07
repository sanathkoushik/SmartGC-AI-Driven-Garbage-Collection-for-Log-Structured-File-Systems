"""CSV interface-contract constants shared by every producer/consumer.

Keep these in lock-step with ``docs/architecture.md`` section 2 and the C++
``CONTRACT_VERSION`` in ``simulator/include/types.hpp``.
"""
from __future__ import annotations

# Bumped from the implicit v1 (Phase 1) when Phase 4b/5 added
# predicted_stream_class / confidence to predictions.csv and the stream /
# learned-trigger / tail-latency counters to the metrics contract.
CONTRACT_VERSION = 2

# 2.1 Normalized trace (data/processed/*.csv)
#   size is expressed in 4 KiB blocks (integer >= 1), not bytes.
TRACE_COLUMNS = ["timestamp", "lba", "size", "operation"]

# 2.2 Predictions (data/predictions/*.csv) - v2
PREDICTION_COLUMNS = [
    "timestamp",
    "lba",
    "predicted_rewrite_interval",
    "predicted_class",           # HOT / COLD  (1 / 0)
    "predicted_stream_class",    # 0=SHORT 1=MEDIUM 2=LONG   (Phase 5)
    "confidence",                # [0, 1]                     (Phase 4b)
]

# 2.3 Simulator metrics (results/metrics/*.csv) - v2  (written by the C++ CLI;
# mirrored here so Python consumers can validate/observe the schema).
METRICS_COLUMNS = [
    "contract_version",
    "run_id",
    "workload_name",
    "placement_policy",
    "total_segments",
    "blocks_per_segment",
    "logical_bytes_written",
    "physical_bytes_written",
    "gc_bytes_copied",
    "valid_blocks_migrated",
    "gc_count",
    "waf",
    "migration_stream_count",
    "learned_trigger_enabled",
    "avg_gc_decision_latency_us",
    "gc_migrated_p50",
    "gc_migrated_p95",
    "gc_migrated_p99",
    "predicted_writes",
    "confidence_fallbacks",
    "confidence_fallback_rate",
]

# Stream-class vocabulary (must match simulator/include/types.hpp StreamClass).
STREAM_SHORT, STREAM_MEDIUM, STREAM_LONG = 0, 1, 2
STREAM_NAMES = {STREAM_SHORT: "SHORT", STREAM_MEDIUM: "MEDIUM", STREAM_LONG: "LONG"}

HOT, COLD = 1, 0
