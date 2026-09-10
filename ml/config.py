"""Typed access to config/config.yaml.

The same file drives the C++ simulator (via simulator/src/config.cpp) and the
Python subsystem, so a single edit changes both. Unknown keys are rejected here
for the same reason they are rejected in C++: a typo must never silently fall
back to a default in a research pipeline.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed or unrecognised."""


def _require_keys(section: dict[str, Any], known: set[str], name: str) -> None:
    unknown = set(section) - known
    if unknown:
        raise ConfigError(
            f"Unknown key(s) {sorted(unknown)} in section '{name}' of the SmartGC config; "
            "refusing to fall back to a default"
        )


@dataclass(frozen=True)
class MlConfig:
    """Hyperparameters and split ratios for the sequence model."""

    sequence_length: int = 10
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 25
    loss_function: str = "SmoothL1"
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    hot_percentile_cutoff: float = 30.0
    early_stopping_patience: int = 5
    grad_clip_norm: float = 1.0
    min_sequences_per_trace: int = 200

    def validate(self) -> None:
        if self.sequence_length < 1:
            raise ConfigError("ml.sequence_length must be >= 1")
        if self.hidden_dim < 1 or self.num_layers < 1:
            raise ConfigError("ml.hidden_dim and ml.num_layers must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("ml.dropout must be in [0, 1)")
        if self.batch_size < 1:
            raise ConfigError("ml.batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ConfigError("ml.learning_rate must be > 0")
        if self.epochs < 1:
            raise ConfigError("ml.epochs must be >= 1")
        if self.loss_function not in {"SmoothL1", "L1", "MSE"}:
            raise ConfigError("ml.loss_function must be one of SmoothL1, L1, MSE")
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(
                f"ml.train_split + val_split + test_split must sum to 1.0 (got {total})"
            )
        for name, value in (("train_split", self.train_split),
                            ("val_split", self.val_split),
                            ("test_split", self.test_split)):
            if not 0.0 < value < 1.0:
                raise ConfigError(f"ml.{name} must be in (0, 1)")
        if not 0.0 < self.hot_percentile_cutoff < 100.0:
            raise ConfigError("ml.hot_percentile_cutoff must be in (0, 100)")


@dataclass(frozen=True)
class SimulatorConfig:
    """Device geometry and cleaning parameters. Mirrors smartgc::SimulatorConfig."""

    block_size_bytes: int = 4096
    blocks_per_segment: int = 64
    total_segments: int = 32
    gc_free_segments_threshold: int = 2
    gc_reserved_segments: int = 2
    gc_separate_stream: bool = True
    gc_policy: str = "GREEDY"
    placement_policy: str = "MIXED"

    @property
    def total_capacity_blocks(self) -> int:
        return self.total_segments * self.blocks_per_segment

    def user_stream_count(self, policy: str | None = None) -> int:
        return 1 if (policy or self.placement_policy) == "MIXED" else 2

    def max_live_blocks(self, policy: str | None = None) -> int:
        unavailable = (self.gc_reserved_segments
                       + self.user_stream_count(policy)
                       + (1 if self.gc_separate_stream else 0))
        if self.total_segments <= unavailable:
            return 0
        return (self.total_segments - unavailable) * self.blocks_per_segment

    def max_utilization(self, policy: str | None = None) -> float:
        cap = self.total_capacity_blocks
        return 0.0 if cap == 0 else self.max_live_blocks(policy) / cap


@dataclass(frozen=True)
class PreprocessingConfig:
    """Trace normalization rules."""

    # A request longer than this many blocks is treated as malformed and dropped.
    # MSR requests reach ~1 MB (256 blocks at 4 KB); the cap catches corrupt rows
    # without discarding legitimate large I/O.
    max_blocks_per_request: int = 2048
    # Offsets beyond this are physically implausible for the traced volumes and
    # indicate a parse error rather than real I/O (1 PiB).
    max_offset_bytes: int = 1 << 50
    # Records are stable-sorted by timestamp; this reports how many were out of
    # order rather than silently reordering a badly broken trace.
    max_out_of_order_fraction: float = 0.01


@dataclass(frozen=True)
class SmartGcConfig:
    random_seed: int = 42
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    ml: MlConfig = field(default_factory=MlConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_path"] = str(self.source_path) if self.source_path else None
        return data


_SIM_KEYS = {
    "block_size_bytes", "blocks_per_segment", "total_segments",
    "gc_free_segments_threshold", "gc_reserved_segments", "gc_separate_stream",
    "gc_policy", "placement_policy",
}
_ML_KEYS = {f.name for f in MlConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
_PREP_KEYS = {f.name for f in PreprocessingConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]


def load_config(path: str | os.PathLike[str] | None = None) -> SmartGcConfig:
    """Load and validate config/config.yaml."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: top level must be a mapping")

    sim_raw = raw.get("simulator", {}) or {}
    ml_raw = raw.get("ml", {}) or {}
    prep_raw = raw.get("preprocessing", {}) or {}
    _require_keys(sim_raw, _SIM_KEYS, "simulator")
    _require_keys(ml_raw, _ML_KEYS, "ml")
    _require_keys(prep_raw, _PREP_KEYS, "preprocessing")

    ml = MlConfig(**ml_raw)
    ml.validate()

    config = SmartGcConfig(
        random_seed=int(raw.get("random_seed", 42)),
        simulator=SimulatorConfig(**sim_raw),
        ml=ml,
        preprocessing=PreprocessingConfig(**prep_raw),
        source_path=config_path,
    )
    if config.simulator.gc_reserved_segments < 1:
        raise ConfigError("simulator.gc_reserved_segments must be >= 1")
    return config


def git_commit_hash() -> str:
    """Best-effort git revision of the working tree, for experiment metadata."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                return f"{commit}-dirty"
            return commit
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def environment_metadata() -> dict[str, Any]:
    """Everything needed to reproduce a run's numerical environment."""
    meta: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "git_commit": git_commit_hash(),
    }
    try:
        import torch  # imported lazily so preprocessing does not require it

        meta["torch_version"] = torch.__version__
        meta["torch_device"] = "cuda" if torch.cuda.is_available() else "cpu"
        meta["torch_num_threads"] = torch.get_num_threads()
    except ImportError:
        meta["torch_version"] = None
        meta["torch_device"] = None
    return meta
