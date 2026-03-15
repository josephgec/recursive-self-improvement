"""Environment-aware configuration for rsi-infra.

Loads from YAML files or environment variables and provides typed accessors
for each subsystem (REPL, symbolic, tracking).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from repl.src.sandbox import REPLConfig
from symbolic.src.sandbox import SymbolicConfig


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# InfraConfig
# ---------------------------------------------------------------------------

@dataclass
class InfraConfig:
    """Unified configuration container for all rsi-infra subsystems.

    Use :meth:`from_yaml` to load from a YAML file or :meth:`from_env` to
    auto-detect the environment (``RSI_ENV`` env-var or default to local).
    """

    project_name: str = "rsi-infra"
    environment: str = "local"  # "local" | "docker" | "cloud"

    # Raw section dicts (merged from default + overlay)
    _repl_raw: dict[str, Any] = field(default_factory=dict)
    _symbolic_raw: dict[str, Any] = field(default_factory=dict)
    _tracking_raw: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Typed accessors
    # ------------------------------------------------------------------

    @property
    def repl_config(self) -> REPLConfig:
        """Build a :class:`REPLConfig` from the merged REPL section."""
        return REPLConfig.from_dict(self._repl_raw)

    @property
    def repl_pool_size(self) -> int:
        return int(self._repl_raw.get("pool_size", 4))

    @property
    def repl_backend(self) -> str:
        return str(self._repl_raw.get("backend", "local"))

    @property
    def symbolic_config(self) -> SymbolicConfig:
        """Build a :class:`SymbolicConfig` from the merged symbolic section."""
        raw = dict(self._symbolic_raw)
        # Map sympy_timeout / z3_timeout to the generic ``timeout``
        if "sympy_timeout" in raw and "timeout" not in raw:
            raw["timeout"] = raw["sympy_timeout"]
        return SymbolicConfig.from_dict(raw)

    @property
    def symbolic_backend(self) -> str:
        return str(self._symbolic_raw.get("backend", "subprocess"))

    @property
    def tracking_config(self) -> dict[str, Any]:
        """Return the raw tracking configuration dict."""
        return dict(self._tracking_raw)

    @property
    def tracking_backend(self) -> str:
        return str(self._tracking_raw.get("backend", "local"))

    @property
    def safety_config(self) -> dict[str, Any]:
        """Return the ``safety`` sub-section of tracking config."""
        return dict(self._tracking_raw.get("safety", {}))

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> InfraConfig:
        """Load configuration from a YAML file.

        The file is first merged on top of ``configs/default.yaml`` so that
        any keys not specified in the overlay get sensible defaults.
        """
        # Load defaults
        defaults: dict[str, Any] = {}
        if _DEFAULT_YAML.exists():
            with open(_DEFAULT_YAML, encoding="utf-8") as f:
                defaults = yaml.safe_load(f) or {}

        # Load overlay
        overlay_path = Path(path)
        overlay: dict[str, Any] = {}
        if overlay_path.exists():
            with open(overlay_path, encoding="utf-8") as f:
                overlay = yaml.safe_load(f) or {}

        merged = _deep_merge(defaults, overlay)
        return cls._from_dict(merged)

    @classmethod
    def from_env(cls) -> InfraConfig:
        """Auto-detect configuration from environment.

        Looks for ``RSI_ENV`` (default ``"local"``) and loads
        ``configs/{env}.yaml`` over the defaults.  Individual values can be
        overridden with ``RSI_REPL_BACKEND``, ``RSI_TRACKING_BACKEND``, etc.
        """
        env = os.environ.get("RSI_ENV", "local")
        configs_dir = Path(__file__).resolve().parent.parent / "configs"
        env_yaml = configs_dir / f"{env}.yaml"

        if env_yaml.exists():
            config = cls.from_yaml(env_yaml)
        else:
            config = cls.from_yaml(_DEFAULT_YAML)

        # Override from env vars
        if "RSI_REPL_BACKEND" in os.environ:
            config._repl_raw["backend"] = os.environ["RSI_REPL_BACKEND"]
        if "RSI_SYMBOLIC_BACKEND" in os.environ:
            config._symbolic_raw["backend"] = os.environ["RSI_SYMBOLIC_BACKEND"]
        if "RSI_TRACKING_BACKEND" in os.environ:
            config._tracking_raw["backend"] = os.environ["RSI_TRACKING_BACKEND"]

        config.environment = env
        return config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> InfraConfig:
        project = d.get("project", {})
        return cls(
            project_name=project.get("name", "rsi-infra"),
            environment=project.get("environment", "local"),
            _repl_raw=d.get("repl", {}),
            _symbolic_raw=d.get("symbolic", {}),
            _tracking_raw=d.get("tracking", {}),
        )


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* on top of *base* (non-destructive)."""
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
