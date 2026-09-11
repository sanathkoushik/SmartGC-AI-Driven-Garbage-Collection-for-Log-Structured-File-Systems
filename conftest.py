"""Repo-root conftest: makes ``ml`` and ``experiments`` importable as packages
when running ``pytest`` from the repository root (no setup.py/pyproject.toml
in this project, so pytest's default rootdir insertion needs a nudge)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
