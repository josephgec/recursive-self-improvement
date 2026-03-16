"""Tests for GDI alerting system."""

import pytest

from src.composite.gdi import GDIResult
from src.alerting.thresholds import ThresholdConfig
from src.alerting.alert_manager import AlertManager, Alert
from src.alerting.escalation import EscalationPolicy
from src.alerting.channels import LogChannel, WandBChannel, WebhookChannel


def _make_result(score: float, level: str) -> GDIResult:
    """Create a GDIResult with given score and level."""
    return GDIResult(
        composite_score=score,
        alert_level=level,
        trend="stable",
        semantic_score=score,
        lexical_score=score,
        structural_score=score,
        distributional_score=score,
    )


class TestThresholdConfig:
    """Tests for ThresholdConfig."""

    def test_default_thresholds(self):
        """Default thresholds should be valid."""
        tc = ThresholdConfig()
        assert tc.green_max == 0.15
        assert tc.yellow_max == 0.40
        assert tc.orange_max == 0.70
        assert tc.red_min == 0.70

    def test_get_level(self):
        """get_level should return correct levels."""
        tc = ThresholdConfig()
        assert tc.get_level(0.10) == "green"
        assert tc.get_level(0.30) == "yellow"
        assert tc.get_level(0.55) == "orange"
        assert tc.get_level(0.80) == "red"

    def test_from_config(self):
        """Should create from config dict."""
        config = {
            "thresholds": {
                "green_max": 0.10,
                "yellow_max": 0.30,
                "orange_max": 0.60,
                "red_min": 0.60,
            }
        }
        tc = ThresholdConfig.from_config(config)
        assert tc.green_max == 0.10

    def test_from_config_defaults(self):
        """Empty config should use defaults."""
        tc = ThresholdConfig.from_config({})
        assert tc.green_max == 0.15

    def test_from_calibration(self):
        """Should create from calibrated thresholds."""
        from src.calibration.collapse_calibrator import CalibratedThresholds
        cal = CalibratedThresholds(
            green_max=0.12, yellow_max=0.35,
            orange_max=0.65, red_min=0.65,
        )
        tc = ThresholdConfig.from_calibration(cal)
        assert tc.green_max == 0.12


class TestAlertManager:
    """Tests for AlertManager."""

    def test_green_no_alert(self):
        """Green level should not generate alerts."""
        mgr = AlertManager()
        result = _make_result(0.10, "green")
        alert = mgr.process(result, 1)
        assert alert is None

    def test_yellow_alert(self):
        """Yellow level should generate an alert."""
        mgr = AlertManager()
        result = _make_result(0.30, "yellow")
        alert = mgr.process(result, 1)
        assert alert is not None
        assert alert.level == "yellow"
        assert alert.action == "log"

    def test_orange_alert(self):
        """Orange level should generate an alert."""
        mgr = AlertManager()
        result = _make_result(0.55, "orange")
        alert = mgr.process(result, 1)
        assert alert is not None
        assert alert.level == "orange"
        assert alert.action == "alert"

    def test_red_alert(self):
        """Red level should generate an alert."""
        mgr = AlertManager()
        result = _make_result(0.80, "red")
        alert = mgr.process(result, 1)
        assert alert is not None
        assert alert.level == "red"
        assert alert.action == "pause"

    def test_deduplication(self):
        """Same non-red level should be deduplicated."""
        mgr = AlertManager()
        result = _make_result(0.30, "yellow")
        alert1 = mgr.process(result, 1)
        alert2 = mgr.process(result, 2)
        assert alert1 is not None
        assert alert2 is None  # Deduplicated

    def test_red_not_deduplicated(self):
        """Red alerts should not be deduplicated."""
        mgr = AlertManager()
        result = _make_result(0.80, "red")
        alert1 = mgr.process(result, 1)
        alert2 = mgr.process(result, 2)
        assert alert1 is not None
        assert alert2 is not None

    def test_level_change_not_deduplicated(self):
        """Changing level should generate new alert."""
        mgr = AlertManager()
        yellow = _make_result(0.30, "yellow")
        orange = _make_result(0.55, "orange")
        alert1 = mgr.process(yellow, 1)
        alert2 = mgr.process(orange, 2)
        assert alert1 is not None
        assert alert2 is not None

    def test_alert_history(self):
        """Alert history should be maintained."""
        mgr = AlertManager()
        mgr.process(_make_result(0.30, "yellow"), 1)
        mgr.process(_make_result(0.55, "orange"), 2)
        assert len(mgr.alert_history) == 2

    def test_reset(self):
        """Reset should clear state."""
        mgr = AlertManager()
        mgr.process(_make_result(0.30, "yellow"), 1)
        mgr.reset()
        assert len(mgr.alert_history) == 0

    def test_channels_notified(self):
        """Alert channels should receive alerts."""
        channel = LogChannel()
        mgr = AlertManager(channels=[channel])
        mgr.process(_make_result(0.30, "yellow"), 1)
        assert len(channel.sent_alerts) == 1


class TestEscalationPolicy:
    """Tests for EscalationPolicy."""

    def test_green_action(self):
        """Green should yield no action."""
        ep = EscalationPolicy()
        assert ep.get_action("green") == "none"

    def test_yellow_action(self):
        """Yellow should yield log."""
        ep = EscalationPolicy()
        assert ep.get_action("yellow") == "log"

    def test_orange_action(self):
        """Orange should yield alert."""
        ep = EscalationPolicy()
        assert ep.get_action("orange") == "alert"

    def test_red_action(self):
        """Red should yield pause."""
        ep = EscalationPolicy()
        assert ep.get_action("red") == "pause"

    def test_consecutive_reds_emergency(self):
        """Three consecutive reds should trigger emergency stop."""
        ep = EscalationPolicy(consecutive_red_limit=3)
        ep.get_action("red")
        ep.get_action("red")
        action = ep.get_action("red")
        assert action == "emergency_stop"

    def test_consecutive_reds_reset_on_non_red(self):
        """Non-red should reset consecutive red counter."""
        ep = EscalationPolicy(consecutive_red_limit=3)
        ep.get_action("red")
        ep.get_action("red")
        ep.get_action("yellow")  # Reset
        ep.get_action("red")
        assert ep.get_action("red") == "pause"  # Only 2 consecutive

    def test_should_pause(self):
        """Should pause on red."""
        ep = EscalationPolicy()
        assert not ep.should_pause("green")
        assert not ep.should_pause("yellow")
        assert not ep.should_pause("orange")
        assert ep.should_pause("red")

    def test_should_rollback(self):
        """Should rollback on red with high score."""
        ep = EscalationPolicy()
        assert not ep.should_rollback("red", 0.75)
        assert ep.should_rollback("red", 0.90)
        assert not ep.should_rollback("orange", 0.90)

    def test_should_emergency_stop(self):
        """Emergency stop after consecutive reds."""
        ep = EscalationPolicy(consecutive_red_limit=2)
        ep.get_action("red")
        assert not ep.should_emergency_stop()
        ep.get_action("red")
        assert ep.should_emergency_stop()

    def test_reset(self):
        """Reset should clear state."""
        ep = EscalationPolicy()
        ep.get_action("red")
        ep.get_action("red")
        ep.reset()
        assert not ep.should_emergency_stop()


class TestAlertChannels:
    """Tests for alert channels."""

    def _make_alert(self) -> Alert:
        """Create a test alert."""
        return Alert(
            level="yellow",
            score=0.30,
            iteration=1,
            message="Test alert",
            action="log",
        )

    def test_log_channel(self):
        """LogChannel should log and record alerts."""
        channel = LogChannel()
        alert = self._make_alert()
        assert channel.send(alert)
        assert len(channel.sent_alerts) == 1

    def test_wandb_channel(self):
        """WandBChannel should record alerts."""
        channel = WandBChannel(project="test", run_name="test_run")
        alert = self._make_alert()
        assert channel.send(alert)
        assert len(channel.sent_alerts) == 1
        assert channel.project == "test"

    def test_webhook_channel(self):
        """WebhookChannel should record alerts."""
        channel = WebhookChannel(url="https://test.example.com")
        alert = self._make_alert()
        assert channel.send(alert)
        assert len(channel.sent_alerts) == 1
        assert channel.url == "https://test.example.com"

    def test_multiple_channels(self):
        """Multiple channels should all receive alerts."""
        channels = [LogChannel(), WandBChannel(), WebhookChannel()]
        mgr = AlertManager(channels=channels)
        mgr.process(_make_result(0.30, "yellow"), 1)

        for channel in channels:
            assert len(channel.sent_alerts) == 1
