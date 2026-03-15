"""Tests for the agent health monitor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.measurement.health_monitor import AgentHealthMonitor, HealthStatus


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_agent(functional: bool = True) -> MagicMock:
    agent = MagicMock()
    agent.is_functional.return_value = functional
    return agent


# ------------------------------------------------------------------ #
# HealthStatus
# ------------------------------------------------------------------ #


class TestHealthStatus:
    def test_defaults(self) -> None:
        status = HealthStatus()
        assert status.is_alive is True
        assert status.is_progressing is True
        assert status.resource_ok is True
        assert status.modification_rate_ok is True
        assert status.overall_healthy is True
        assert status.details == ""
        assert status.timestamp > 0


# ------------------------------------------------------------------ #
# AgentHealthMonitor.check
# ------------------------------------------------------------------ #


class TestCheck:
    def test_healthy_agent(self) -> None:
        monitor = AgentHealthMonitor()
        agent = _make_agent(functional=True)
        status = monitor.check(agent, iteration=0, accuracy=0.8)
        assert status.is_alive is True
        assert status.is_progressing is True
        assert status.resource_ok is True
        assert status.modification_rate_ok is True
        assert status.overall_healthy is True
        assert status.details == "healthy"

    def test_dead_agent(self) -> None:
        monitor = AgentHealthMonitor()
        agent = _make_agent(functional=False)
        status = monitor.check(agent, iteration=0, accuracy=0.0)
        assert status.is_alive is False
        assert status.overall_healthy is False
        assert "not responding" in status.details.lower()

    def test_agent_raises_exception(self) -> None:
        monitor = AgentHealthMonitor()
        agent = MagicMock()
        agent.is_functional.side_effect = RuntimeError("crash")
        status = monitor.check(agent, iteration=0, accuracy=0.5)
        assert status.is_alive is False

    def test_stagnant_agent(self) -> None:
        monitor = AgentHealthMonitor(max_stagnant_iterations=3)
        agent = _make_agent()
        # Feed 3 identical accuracies
        for i in range(3):
            status = monitor.check(agent, iteration=i, accuracy=0.50)
        assert status.is_progressing is False
        assert "stagnant" in status.details.lower()

    def test_progressing_agent(self) -> None:
        monitor = AgentHealthMonitor(max_stagnant_iterations=3)
        agent = _make_agent()
        for i, acc in enumerate([0.50, 0.55, 0.60]):
            status = monitor.check(agent, iteration=i, accuracy=acc)
        assert status.is_progressing is True

    def test_high_modification_rate(self) -> None:
        monitor = AgentHealthMonitor(max_modification_rate=2.0)
        agent = _make_agent()
        # 3 checks with modification_count=10 each
        for i in range(3):
            status = monitor.check(agent, iteration=i, accuracy=0.8, modification_count=10)
        assert status.modification_rate_ok is False
        assert "modification rate" in status.details.lower()

    def test_low_modification_rate_ok(self) -> None:
        monitor = AgentHealthMonitor(max_modification_rate=5.0)
        agent = _make_agent()
        for i in range(3):
            status = monitor.check(agent, iteration=i, accuracy=0.8, modification_count=2)
        assert status.modification_rate_ok is True

    def test_history_recorded(self) -> None:
        monitor = AgentHealthMonitor()
        agent = _make_agent()
        monitor.check(agent, iteration=0, accuracy=0.8)
        monitor.check(agent, iteration=1, accuracy=0.7)
        assert len(monitor.history) == 2


# ------------------------------------------------------------------ #
# should_terminate
# ------------------------------------------------------------------ #


class TestShouldTerminate:
    def test_no_history(self) -> None:
        monitor = AgentHealthMonitor()
        assert monitor.should_terminate() is False

    def test_terminate_on_dead_agent(self) -> None:
        monitor = AgentHealthMonitor()
        agent = _make_agent(functional=False)
        monitor.check(agent, iteration=0, accuracy=0.0)
        assert monitor.should_terminate() is True

    def test_terminate_on_prolonged_stagnation(self) -> None:
        monitor = AgentHealthMonitor(max_stagnant_iterations=3)
        agent = _make_agent()
        for i in range(5):
            monitor.check(agent, iteration=i, accuracy=0.50)
        assert monitor.should_terminate() is True

    def test_no_terminate_when_progressing(self) -> None:
        monitor = AgentHealthMonitor(max_stagnant_iterations=3)
        agent = _make_agent()
        for i, acc in enumerate([0.50, 0.55, 0.60, 0.65]):
            monitor.check(agent, iteration=i, accuracy=acc)
        assert monitor.should_terminate() is False

    def test_stagnation_resets_on_progress(self) -> None:
        monitor = AgentHealthMonitor(max_stagnant_iterations=3)
        agent = _make_agent()
        # Stagnant for 2 iterations
        monitor.check(agent, iteration=0, accuracy=0.50)
        monitor.check(agent, iteration=1, accuracy=0.50)
        # Then improve
        monitor.check(agent, iteration=2, accuracy=0.60)
        monitor.check(agent, iteration=3, accuracy=0.65)
        assert monitor.should_terminate() is False


# ------------------------------------------------------------------ #
# Internal checks
# ------------------------------------------------------------------ #


class TestInternalChecks:
    def test_check_progress_not_enough_data(self) -> None:
        monitor = AgentHealthMonitor(max_stagnant_iterations=5)
        agent = _make_agent()
        # Only 2 data points, fewer than window
        monitor.check(agent, iteration=0, accuracy=0.5)
        monitor.check(agent, iteration=1, accuracy=0.5)
        # Should still consider it progressing (not enough data to judge)
        assert monitor.history[-1].is_progressing is True

    def test_check_modification_rate_not_enough_data(self) -> None:
        monitor = AgentHealthMonitor(max_modification_rate=1.0)
        agent = _make_agent()
        # Only 2 data points, not enough for mod rate check
        monitor.check(agent, iteration=0, accuracy=0.8, modification_count=100)
        monitor.check(agent, iteration=1, accuracy=0.8, modification_count=100)
        # Should still be ok (< 3 data points)
        assert monitor.history[-1].modification_rate_ok is True

    def test_check_resources_always_true(self) -> None:
        monitor = AgentHealthMonitor()
        agent = _make_agent()
        status = monitor.check(agent, iteration=0, accuracy=0.8)
        assert status.resource_ok is True
