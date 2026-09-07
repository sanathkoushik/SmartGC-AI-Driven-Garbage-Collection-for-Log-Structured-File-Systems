"""Centralized access to ``config/config.yaml``.

Every ML script loads configuration through this module so that there are no
hard-coded hyperparameters anywhere in ``ml/`` (mirrors the "no hardcoded
parameters" rule already enforced in the C++ simulator).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import yaml

# Repo root = two levels up from this file (ml/common/config.py -> repo/).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(REPO_ROOT, "config", "config.yaml")


def repo_path(*parts: str) -> str:
    """Absolute path rooted at the repository directory."""
    return os.path.join(REPO_ROOT, *parts)


@lru_cache(maxsize=1)
def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Malformed config file: {path}")
    return cfg


def get(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch ``a.b.c`` from a nested dict, returning ``default`` if absent."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def require(cfg: Dict[str, Any], dotted_key: str) -> Any:
    sentinel = object()
    value = get(cfg, dotted_key, sentinel)
    if value is sentinel:
        raise KeyError(f"Required config key missing: {dotted_key}")
    return value


def seed(cfg: Dict[str, Any] | None = None) -> int:
    cfg = cfg or load_config()
    return int(get(cfg, "random_seed", 42))
