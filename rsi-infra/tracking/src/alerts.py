"""Alerting system — evaluates metrics, drift, and constraint results and
dispatches alerts through configurable channels."""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """A single alert event."""

    severity: str  # "warning" | "critical" | "halt"
    metric: str
    value: float
    threshold: float
    generation: int
    message: str
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

class AlertChannel(ABC):
    """Abstract channel for dispatching alerts."""

    @abstractmethod
    def send(self, alert: Alert) -> None:
        """Send an alert through this channel."""


class LogAlertChannel(AlertChannel):
    """Print alerts to stderr using rich-style formatting."""

    def send(self, alert: Alert) -> None:
        severity_markers = {
            "warning": "\033[33m[WARNING]\033[0m",
            "critical": "\033[31m[CRITICAL]\033[0m",
            "halt": "\033[1;31m[HALT]\033[0m",
        }
        marker = severity_markers.get(alert.severity.lower(), f"[{alert.severity.upper()}]")
        line = (
            f"{marker} gen={alert.generation} metric={alert.metric} "
            f"value={alert.value:.4f} threshold={alert.threshold:.4f} "
            f"| {alert.message}"
        )
        print(line, file=sys.stderr)


class WandBAlertChannel(AlertChannel):
    """Send alerts via ``wandb.alert``.  No-op if wandb unavailable."""

    def __init__(self) -> None:
        try:
            import wandb  # type: ignore[import-untyped]
            self._wandb = wandb
        except ImportError:
            self._wandb = None
            logger.warning("wandb not installed — WandBAlertChannel is a no-op.")

    def send(self, alert: Alert) -> None:
        if self._wandb is None:
            return
        level_map = {
            "warning": "WARN",
            "critical": "ERROR",
            "halt": "ERROR",
        }
        level_name = level_map.get(alert.severity.lower(), "WARN")
        level = getattr(self._wandb.AlertLevel, level_name)
        self._wandb.alert(
            title=f"{alert.metric} ({alert.severity})",
            text=alert.message,
            level=level,
        )


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """Evaluate metrics / drift / constraints and emit alerts.

    Parameters
    ----------
    config : dict
        Tracking config with ``safety`` sub-dict containing thresholds.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        safety = config.get("safety", config)
        self._drift_threshold: float = safety.get("alert_threshold_drift_cosine", 0.15)
        self._kl_threshold: float = safety.get("alert_threshold_kl", 0.1)
        self._entropy_drop_threshold: float = safety.get("alert_threshold_entropy_drop", 0.3)
        self._channels: list[AlertChannel] = []

    def add_channel(self, channel: AlertChannel) -> None:
        """Register an alert channel."""
        self._channels.append(channel)

    def _emit(self, alert: Alert) -> None:
        """Dispatch *alert* to all registered channels."""
        for ch in self._channels:
            ch.send(alert)

    def check_and_alert(
        self,
        generation: int,
        metrics: dict[str, Any] | None = None,
        drift: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> list[Alert]:
        """Evaluate metrics and return (and dispatch) any alerts triggered.

        Parameters
        ----------
        generation : int
            Current generation number.
        metrics : dict | None
            Metric values (e.g. loss, perplexity, entropy).
        drift : dict | None
            Goal-drift measurement dict (output of GoalDriftMeasurement).
        constraints : dict | None
            Constraint report dict (output of PreservationReport).

        Returns
        -------
        list[Alert]
            All alerts that fired.
        """
        alerts: list[Alert] = []
        metrics = metrics or {}
        drift = drift or {}
        constraints = constraints or {}

        # --- drift-based alerts ---
        gdi = drift.get("goal_drift_index")
        if gdi is not None and gdi > self._drift_threshold:
            a = Alert(
                severity="warning",
                metric="goal_drift_index",
                value=float(gdi),
                threshold=self._drift_threshold,
                generation=generation,
                message=f"Goal Drift Index ({gdi:.4f}) exceeds threshold ({self._drift_threshold}).",
            )
            alerts.append(a)

        kl = drift.get("distributional_drift")
        if kl is not None and kl > self._kl_threshold:
            a = Alert(
                severity="warning",
                metric="distributional_drift",
                value=float(kl),
                threshold=self._kl_threshold,
                generation=generation,
                message=f"KL divergence ({kl:.4f}) exceeds threshold ({self._kl_threshold}).",
            )
            alerts.append(a)

        # --- metric-based alerts ---
        entropy_drop = metrics.get("entropy_drop")
        if entropy_drop is not None and entropy_drop > self._entropy_drop_threshold:
            a = Alert(
                severity="warning",
                metric="entropy_drop",
                value=float(entropy_drop),
                threshold=self._entropy_drop_threshold,
                generation=generation,
                message=f"Entropy drop ({entropy_drop:.4f}) exceeds threshold ({self._entropy_drop_threshold}).",
            )
            alerts.append(a)

        # --- constraint-based alerts ---
        recommendation = constraints.get("recommendation")
        if recommendation in ("halt", "revert"):
            # Find failing constraints
            results = constraints.get("results", [])
            failing = [r for r in results if not r.get("passed", True)]
            for r in failing:
                sev = "halt" if r.get("violation_severity") == "halt" else "critical"
                a = Alert(
                    severity=sev,
                    metric=r.get("name", "constraint"),
                    value=float(r.get("value", 0)),
                    threshold=float(r.get("threshold", 0)),
                    generation=generation,
                    message=(
                        f"Constraint '{r.get('name')}' violated: "
                        f"value={r.get('value')}, threshold={r.get('threshold')}. "
                        f"Recommendation: {recommendation}."
                    ),
                )
                alerts.append(a)

        # Dispatch all
        for a in alerts:
            self._emit(a)

        return alerts
