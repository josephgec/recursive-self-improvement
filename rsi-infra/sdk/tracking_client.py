"""High-level tracking + safety client for rsi-infra.

Wraps :class:`ExperimentTracker`, :class:`GoalDriftComputer`,
:class:`ConstraintPreserver`, and :class:`AlertManager` into a single
facade with a simple lifecycle API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sdk.config import InfraConfig
from tracking.src.alerts import AlertManager, LogAlertChannel
from tracking.src.capability_alignment import CapabilityAlignmentTracker
from tracking.src.constraint import ConstraintPreserver, PreservationReport
from tracking.src.goal_drift import GoalDriftComputer, GoalDriftMeasurement
from tracking.src.local_backend import LocalTracker
from tracking.src.tracker import ExperimentTracker


@dataclass
class SafetyCheckResult:
    """Aggregate result from :meth:`TrackingClient.check_safety`."""

    drift: GoalDriftMeasurement | None = None
    constraints: PreservationReport | None = None
    alerts: list[dict[str, Any]] | None = None
    safe: bool = True


class TrackingClient:
    """Convenient tracking + safety facade.

    Usage::

        client = TrackingClient.from_config(config)
        client.start_run("my_experiment", {"lr": 1e-3})
        client.log_generation(0, {"loss": 1.5, "accuracy": 0.7})
        safety = client.check_safety(0, ["generated text ..."], {"accuracy": 0.7})
        client.finish()
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        drift_computer: GoalDriftComputer,
        constraint_preserver: ConstraintPreserver,
        alert_manager: AlertManager,
        *,
        capability_metrics: list[str] | None = None,
        alignment_metrics: list[str] | None = None,
    ) -> None:
        self._tracker = tracker
        self._drift = drift_computer
        self._constraints = constraint_preserver
        self._alerts = alert_manager
        self._car = CapabilityAlignmentTracker(
            capability_metrics=capability_metrics or ["accuracy"],
            alignment_metrics=alignment_metrics or ["safety_score"],
        )
        self._previous_metrics: dict[str, Any] | None = None
        self._reference_set = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: InfraConfig,
        *,
        capability_metrics: list[str] | None = None,
        alignment_metrics: list[str] | None = None,
    ) -> TrackingClient:
        """Create a fully-wired tracking client from *config*."""
        backend = config.tracking_backend
        tracking_cfg = config.tracking_config
        safety_cfg = config.safety_config

        # Select tracker backend
        if backend == "local":
            tracker: ExperimentTracker = LocalTracker()
        else:
            # Future: WandBTracker
            tracker = LocalTracker()

        drift_computer = GoalDriftComputer(tracking_cfg)
        constraint_preserver = ConstraintPreserver()
        alert_manager = AlertManager(tracking_cfg)
        alert_manager.add_channel(LogAlertChannel())

        return cls(
            tracker=tracker,
            drift_computer=drift_computer,
            constraint_preserver=constraint_preserver,
            alert_manager=alert_manager,
            capability_metrics=capability_metrics,
            alignment_metrics=alignment_metrics,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Initialise a new tracking run."""
        self._tracker.init_run(run_name=name, config=config, tags=tags)

    def finish(self) -> None:
        """Finalise the current run."""
        self._tracker.finish()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_generation(self, generation: int, metrics: dict[str, Any]) -> None:
        """Log metrics for a generation and update internal state."""
        self._tracker.log_generation(generation, metrics)

        # Compute CAR if we have a previous generation's metrics
        if self._previous_metrics is not None:
            car = self._car.compute(generation, metrics, self._previous_metrics)
            self._tracker.log_generation(generation, {"car": car})
        self._previous_metrics = dict(metrics)

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def set_reference(
        self,
        reference_texts: list[str],
        reference_distribution: np.ndarray | None = None,
    ) -> None:
        """Set the generation-0 reference for drift computation."""
        self._drift.set_reference(reference_texts, reference_distribution)
        self._reference_set = True

    def check_safety(
        self,
        generation: int,
        generated_texts: list[str],
        metrics: dict[str, Any],
        token_distribution: np.ndarray | None = None,
    ) -> SafetyCheckResult:
        """Run all safety checks for a generation.

        1. Compute goal drift (if reference has been set).
        2. Check constraint preservation.
        3. Run alert evaluation.

        Returns a :class:`SafetyCheckResult`.
        """
        result = SafetyCheckResult()

        # --- Goal Drift ---
        drift_measurement: GoalDriftMeasurement | None = None
        drift_dict: dict[str, Any] = {}
        if self._reference_set:
            drift_measurement = self._drift.compute(
                generation, generated_texts, token_distribution
            )
            drift_dict = _measurement_to_dict(drift_measurement)
            self._tracker.log_drift(generation, drift_dict)
            result.drift = drift_measurement

        # --- Constraints ---
        report = self._constraints.check(generation, metrics)
        report_dict = _report_to_dict(report)
        self._tracker.log_constraints(generation, report_dict)
        result.constraints = report

        # --- Alerts ---
        alerts = self._alerts.check_and_alert(
            generation=generation,
            metrics=metrics,
            drift=drift_dict,
            constraints=report_dict,
        )
        alert_dicts = [_alert_to_dict(a) for a in alerts]
        for ad in alert_dicts:
            self._tracker.log_alert(ad)
        result.alerts = alert_dicts

        # Overall safe?
        result.safe = report.all_passed and len(alerts) == 0
        return result

    # ------------------------------------------------------------------
    # Direct access
    # ------------------------------------------------------------------

    @property
    def tracker(self) -> ExperimentTracker:
        return self._tracker

    @property
    def drift_computer(self) -> GoalDriftComputer:
        return self._drift

    @property
    def constraint_preserver(self) -> ConstraintPreserver:
        return self._constraints

    @property
    def alert_manager(self) -> AlertManager:
        return self._alerts

    @property
    def car_tracker(self) -> CapabilityAlignmentTracker:
        return self._car


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _measurement_to_dict(m: GoalDriftMeasurement) -> dict[str, Any]:
    return asdict(m)


def _report_to_dict(r: PreservationReport) -> dict[str, Any]:
    return {
        "all_passed": r.all_passed,
        "generation": r.generation,
        "recommendation": r.recommendation,
        "results": [
            {
                "name": cr.name,
                "passed": cr.passed,
                "value": cr.value,
                "threshold": cr.threshold,
                "violation_severity": cr.violation_severity,
            }
            for cr in r.results
        ],
    }


def _alert_to_dict(a: Any) -> dict[str, Any]:
    return {
        "severity": a.severity,
        "metric": a.metric,
        "value": a.value,
        "threshold": a.threshold,
        "generation": a.generation,
        "message": a.message,
        "timestamp": a.timestamp,
    }
