"""Configuration loading and merging."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "model": "mock",
    "max_iterations": 10,
    "max_depth": 3,
    "token_budget": 100000,
    "chunk_size": 4000,
    "forced_final": True,
    "context_loader": {
        "max_chunk_size": 4000,
        "overlap": 200,
    },
    "executor": {
        "timeout": 30,
        "max_output_lines": 500,
    },
    "recursion": {
        "max_depth": 3,
        "max_sub_queries": 5,
        "budget_fraction": 0.5,
    },
    "strategies": {
        "auto_detect": True,
    },
    "logging": {
        "level": "INFO",
        "trajectory": True,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge *override* into a copy of *base*."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration from a YAML file, merged on top of DEFAULT_CONFIG.

    If *path* is ``None``, looks for ``configs/default.yaml`` relative to the
    package root, falling back to DEFAULT_CONFIG if not found.
    """
    if path is None:
        candidate = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
        if candidate.exists():
            path = candidate
        else:
            return dict(DEFAULT_CONFIG)

    path = Path(path)
    if not path.exists():
        return dict(DEFAULT_CONFIG)

    with open(path, "r") as fh:
        file_cfg = yaml.safe_load(fh) or {}

    return merge_configs(DEFAULT_CONFIG, file_cfg)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Public helper: deep-merge *override* onto *base*."""
    return _deep_merge(base, override)
