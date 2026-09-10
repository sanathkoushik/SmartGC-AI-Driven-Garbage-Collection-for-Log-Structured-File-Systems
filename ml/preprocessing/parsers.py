"""Registry of the raw block-I/O trace parsers.

Each dataset has its own reader module; they all share the validation, block
expansion and device-namespace rules in ``base_parser``.
"""

from __future__ import annotations

from typing import Any

from ml.preprocessing.base_parser import (
    NORMALIZED_COLUMNS,
    BaseTraceParser,
    MultipleDevicesError,
    ParseStats,
    open_maybe_compressed,
    ragged_arange,
)
from ml.preprocessing.fiu_parser import FiuTraceParser
from ml.preprocessing.msr_parser import MsrTraceParser
from ml.preprocessing.systor_parser import SystorTraceParser

PARSERS: dict[str, type[BaseTraceParser]] = {
    "msr": MsrTraceParser,
    "fiu": FiuTraceParser,
    "systor": SystorTraceParser,
}

DATASETS = tuple(sorted(PARSERS))

__all__ = [
    "PARSERS", "DATASETS", "get_parser", "BaseTraceParser", "ParseStats",
    "MultipleDevicesError", "NORMALIZED_COLUMNS", "open_maybe_compressed",
    "ragged_arange", "MsrTraceParser", "FiuTraceParser", "SystorTraceParser",
]


def get_parser(dataset: str, **kwargs: Any) -> BaseTraceParser:
    """Instantiate the parser for `dataset` ('msr', 'fiu' or 'systor')."""
    key = dataset.strip().lower()
    if key not in PARSERS:
        raise ValueError(f"Unknown dataset '{dataset}'; expected one of {sorted(PARSERS)}")
    return PARSERS[key](**kwargs)
