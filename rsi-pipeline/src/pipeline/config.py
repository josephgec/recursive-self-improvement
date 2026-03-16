"""Pipeline configuration management."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional


def _load_yaml_simple(path: str) -> Dict[str, Any]:
    """Load YAML without external dependency — supports basic key: value and nesting."""
    result: Dict[str, Any] = {}
    stack: list = [(result, -1)]

    with open(path) as f:
        for raw_line in f:
            stripped = raw_line.rstrip("\n")
            # skip blank lines and comments
            if not stripped.strip() or stripped.strip().startswith("#"):
                continue
            indent = len(stripped) - len(stripped.lstrip())
            line = stripped.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()

            # pop stack to correct nesting level
            while len(stack) > 1 and stack[-1][1] >= indent:
                stack.pop()

            parent = stack[-1][0]

            if val == "" or val == "|":
                # nested mapping
                new_dict: Dict[str, Any] = {}
                parent[key] = new_dict
                stack.append((new_dict, indent))
            else:
                # parse value
                parent[key] = _parse_value(val)

    return result


def _parse_value(val: str) -> Any:
    """Parse a YAML scalar value."""
    # strip inline comments
    if " #" in val:
        val = val[:val.index(" #")].strip()
    # remove quotes
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    # boolean
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no"):
        return False
    # none
    if val.lower() in ("null", "none", "~"):
        return None
    # list (inline)
    if val.startswith("[") and val.endswith("]"):
        items = val[1:-1].split(",")
        return [_parse_value(i.strip()) for i in items if i.strip()]
    # number
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        pass
    # list item (- prefix) handled separately
    return val


class PipelineConfig:
    """Configuration for the RSI pipeline."""

    DEFAULTS: Dict[str, Any] = {
        "pipeline": {
            "max_iterations": 100,
            "candidates_per_iteration": 5,
            "checkpoint_interval": 10,
            "data_dir": "data",
        },
        "verification": {
            "empirical": {"test_timeout_seconds": 30, "min_pass_rate": 0.8},
            "compactness": {"max_bdm_score": 500, "complexity_weight": 0.3},
        },
        "modification": {
            "cooldown_iterations": 2,
            "complexity_ceiling": 1000,
            "allowed_targets": [
                "strategy_evolver", "candidate_pool",
                "empirical_gate", "compactness_gate", "pareto_filter",
            ],
            "forbidden_targets": [
                "emergency_stop", "constraint_enforcer", "gdi_monitor",
            ],
        },
        "safety": {
            "gdi": {"threshold": 0.3},
            "constraints": {
                "accuracy_floor": 0.6,
                "entropy_floor": 0.1,
                "drift_ceiling": 0.5,
            },
            "car": {"min_ratio": 0.5},
            "emergency": {"max_consecutive_rollbacks": 3, "check_interval": 1},
        },
        "scaling": {
            "rlm": {"max_context_tokens": 100000, "session_timeout_seconds": 300},
        },
        "tracking": {"log_level": "INFO", "export_format": "json"},
        "analysis": {
            "convergence": {"window_size": 10, "threshold": 0.01},
            "plateau_detection": {"min_iterations": 5, "tolerance": 0.005},
        },
    }

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = copy.deepcopy(data or self.DEFAULTS)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a config value using dotted key path, e.g. 'safety.gdi.threshold'."""
        keys = dotted_key.split(".")
        current: Any = self._data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def set(self, dotted_key: str, value: Any) -> None:
        """Set a config value using dotted key path."""
        keys = dotted_key.split(".")
        current = self._data
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load config from a YAML file, merged over defaults."""
        loaded = _load_yaml_simple(path)
        merged = cls._deep_merge(copy.deepcopy(cls.DEFAULTS), loaded)
        return cls(merged)

    @classmethod
    def merge_configs(cls, base: "PipelineConfig", override: "PipelineConfig") -> "PipelineConfig":
        """Merge two configs, with override taking precedence."""
        merged = cls._deep_merge(copy.deepcopy(base._data), override._data)
        return cls(merged)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                PipelineConfig._deep_merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
        return base
