"""Weights & Biases experiment tracking backend.

Falls back gracefully when ``wandb`` is not installed — all calls become
no-ops and a warning is emitted once.
"""

from __future__ import annotations

import logging
from typing import Any

from tracking.src.tracker import ExperimentTracker

logger = logging.getLogger(__name__)

try:
    import wandb  # type: ignore[import-untyped]

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


class WandBTracker(ExperimentTracker):
    """W&B-backed experiment tracker.

    Parameters
    ----------
    project : str
        W&B project name.
    entity : str | None
        W&B entity (user / team).  ``None`` uses the default.
    """

    def __init__(self, project: str = "rsi-experiments", entity: str | None = None) -> None:
        self._project = project
        self._entity = entity
        self._run: Any = None
        if not _WANDB_AVAILABLE:
            logger.warning("wandb is not installed — WandBTracker will be a no-op.")

    # -----------------------------------------------------------------
    # ExperimentTracker interface
    # -----------------------------------------------------------------

    def init_run(
        self,
        run_name: str,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        if not _WANDB_AVAILABLE:
            return
        self._run = wandb.init(
            project=self._project,
            entity=self._entity,
            name=run_name,
            config=config or {},
            tags=tags or [],
        )

    def log_generation(self, generation: int, metrics: dict[str, Any]) -> None:
        if not _WANDB_AVAILABLE or self._run is None:
            return
        namespaced: dict[str, Any] = {"generation": generation}
        for key, value in metrics.items():
            namespaced[f"training/{key}"] = value
        wandb.log(namespaced, step=generation)

    def log_drift(self, generation: int, drift: dict[str, Any]) -> None:
        if not _WANDB_AVAILABLE or self._run is None:
            return
        namespaced: dict[str, Any] = {"generation": generation}
        for key, value in drift.items():
            namespaced[f"safety/{key}"] = value
        wandb.log(namespaced, step=generation)

    def log_constraints(self, generation: int, report: dict[str, Any]) -> None:
        if not _WANDB_AVAILABLE or self._run is None:
            return
        namespaced: dict[str, Any] = {"generation": generation}
        for key, value in report.items():
            namespaced[f"safety/{key}"] = value
        wandb.log(namespaced, step=generation)

    def log_alert(self, alert: dict[str, Any]) -> None:
        if not _WANDB_AVAILABLE or self._run is None:
            return
        title = alert.get("metric", "alert")
        text = alert.get("message", "")
        level_str = alert.get("severity", "warning").upper()
        # wandb.AlertLevel: INFO, WARN, ERROR
        level_map = {
            "WARNING": "WARN",
            "CRITICAL": "ERROR",
            "HALT": "ERROR",
        }
        wb_level = getattr(wandb.AlertLevel, level_map.get(level_str, "WARN"))
        wandb.alert(title=title, text=text, level=wb_level)

    def finish(self) -> None:
        if not _WANDB_AVAILABLE or self._run is None:
            return
        wandb.finish()
        self._run = None
