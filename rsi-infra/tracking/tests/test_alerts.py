"""Tests for the alerting system."""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock

import pytest

from tracking.src.alerts import (
    Alert,
    AlertChannel,
    AlertManager,
    LogAlertChannel,
    WandBAlertChannel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> dict:
    return {
        "safety": {
            "alert_threshold_drift_cosine": 0.15,
            "alert_threshold_kl": 0.1,
            "alert_threshold_entropy_drop": 0.3,
        },
    }


@pytest.fixture()
def manager(config: dict) -> AlertManager:
    return AlertManager(config)


# ---------------------------------------------------------------------------
# Tests — no alerts when healthy
# ---------------------------------------------------------------------------

class TestNoAlerts:
    """No alerts should fire when everything is healthy."""

    def test_healthy_metrics_no_alerts(self, manager: AlertManager) -> None:
        alerts = manager.check_and_alert(
            generation=1,
            metrics={"entropy_drop": 0.05},
            drift={"goal_drift_index": 0.01, "distributional_drift": 0.01},
            constraints={"recommendation": "proceed", "results": []},
        )
        assert alerts == []

    def test_no_data_no_alerts(self, manager: AlertManager) -> None:
        alerts = manager.check_and_alert(generation=0)
        assert alerts == []


# ---------------------------------------------------------------------------
# Tests — drift alerts
# ---------------------------------------------------------------------------

class TestDriftAlerts:
    """Drift-based alerts."""

    def test_gdi_above_threshold_warning(self, manager: AlertManager) -> None:
        alerts = manager.check_and_alert(
            generation=3,
            drift={"goal_drift_index": 0.5},
        )
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"
        assert alerts[0].metric == "goal_drift_index"
        assert alerts[0].value == 0.5
        assert alerts[0].generation == 3

    def test_kl_above_threshold_warning(self, manager: AlertManager) -> None:
        alerts = manager.check_and_alert(
            generation=2,
            drift={"distributional_drift": 0.5},
        )
        assert len(alerts) == 1
        assert alerts[0].metric == "distributional_drift"

    def test_gdi_below_threshold_no_alert(self, manager: AlertManager) -> None:
        alerts = manager.check_and_alert(
            generation=1,
            drift={"goal_drift_index": 0.05},
        )
        assert alerts == []


# ---------------------------------------------------------------------------
# Tests — constraint alerts
# ---------------------------------------------------------------------------

class TestConstraintAlerts:
    """Constraint-based alerts."""

    def test_constraint_failure_critical(self, manager: AlertManager) -> None:
        alerts = manager.check_and_alert(
            generation=5,
            constraints={
                "recommendation": "revert",
                "results": [
                    {"name": "accuracy_floor", "passed": False, "value": 0.3, "threshold": 0.6, "violation_severity": "revert"},
                ],
            },
        )
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"
        assert alerts[0].metric == "accuracy_floor"

    def test_halt_severity_alert(self, manager: AlertManager) -> None:
        alerts = manager.check_and_alert(
            generation=5,
            constraints={
                "recommendation": "halt",
                "results": [
                    {"name": "safety_eval_pass", "passed": False, "value": 0.4, "threshold": 0.9, "violation_severity": "halt"},
                ],
            },
        )
        assert len(alerts) == 1
        assert alerts[0].severity == "halt"

    def test_proceed_no_constraint_alert(self, manager: AlertManager) -> None:
        alerts = manager.check_and_alert(
            generation=1,
            constraints={"recommendation": "proceed", "results": []},
        )
        assert alerts == []


# ---------------------------------------------------------------------------
# Tests — metric alerts
# ---------------------------------------------------------------------------

class TestMetricAlerts:
    """Direct metric-based alerts."""

    def test_entropy_drop_warning(self, manager: AlertManager) -> None:
        alerts = manager.check_and_alert(
            generation=4,
            metrics={"entropy_drop": 0.8},
        )
        assert len(alerts) == 1
        assert alerts[0].metric == "entropy_drop"
        assert alerts[0].severity == "warning"


# ---------------------------------------------------------------------------
# Tests — channels
# ---------------------------------------------------------------------------

class TestLogAlertChannel:
    """LogAlertChannel prints to stderr."""

    def test_prints_to_stderr(self) -> None:
        channel = LogAlertChannel()
        alert = Alert(
            severity="warning",
            metric="goal_drift_index",
            value=0.25,
            threshold=0.15,
            generation=3,
            message="Drift exceeded threshold.",
        )
        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            channel.send(alert)
        finally:
            sys.stderr = old_stderr
        output = captured.getvalue()
        assert "WARNING" in output
        assert "goal_drift_index" in output
        assert "gen=3" in output


class TestAlertManagerChannelDispatch:
    """AlertManager dispatches to registered channels."""

    def test_channels_receive_alerts(self, config: dict) -> None:
        manager = AlertManager(config)
        mock_channel = MagicMock(spec=AlertChannel)
        manager.add_channel(mock_channel)

        alerts = manager.check_and_alert(
            generation=1,
            drift={"goal_drift_index": 0.5},
        )
        assert len(alerts) == 1
        mock_channel.send.assert_called_once()
        sent_alert = mock_channel.send.call_args[0][0]
        assert sent_alert.metric == "goal_drift_index"


class TestAlertDataclass:
    """Alert dataclass auto-fills timestamp."""

    def test_timestamp_auto_filled(self) -> None:
        a = Alert(
            severity="warning",
            metric="test",
            value=1.0,
            threshold=0.5,
            generation=0,
            message="test",
        )
        assert a.timestamp != ""
        assert "T" in a.timestamp  # ISO format
