"""Tests for recovery pattern analysis."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pytest

from src.analysis.recovery_patterns import RecoveryPatternAnalyzer
from src.measurement.recovery_tracker import RecoveryEvent


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


def _make_event(
    name: str = "test",
    fault_type: str = "syntax",
    injected: int = 0,
    detected: int | None = None,
    recovered: int | None = None,
    recovery_method: str | None = None,
    recovery_quality: float = 0.0,
    recovery_latency: int | None = None,
    accuracies: list[float] | None = None,
) -> RecoveryEvent:
    event = RecoveryEvent(
        scenario_name=name,
        fault_type=fault_type,
        iteration_injected=injected,
        iteration_detected=detected,
        iteration_recovered=recovered,
        recovery_method=recovery_method,
        recovery_quality=recovery_quality,
        accuracies=accuracies or [],
    )
    return event


@pytest.fixture
def mixed_events() -> list[RecoveryEvent]:
    """A mix of quick, slow, and no recovery events."""
    return [
        # Quick recovery: detected at 1, recovered at 2
        _make_event("s1", "syntax", 0, detected=1, recovered=2,
                     recovery_method="rollback", recovery_quality=0.9,
                     accuracies=[0.8, 0.5, 0.85]),
        # Quick recovery: detected at 2, recovered at 3
        _make_event("s2", "logic", 0, detected=2, recovered=3,
                     recovery_method="rewrite", recovery_quality=0.8,
                     accuracies=[0.9, 0.6, 0.5, 0.88]),
        # Slow recovery: detected at 5, recovered at 8
        _make_event("s3", "runtime", 0, detected=5, recovered=8,
                     recovery_method="rollback", recovery_quality=0.7,
                     accuracies=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.4, 0.5, 0.75]),
        # No recovery
        _make_event("s4", "silent", 0, detected=3, recovered=None,
                     recovery_method=None, recovery_quality=0.0,
                     accuracies=[0.9, 0.8, 0.7, 0.5, 0.3]),
        # No recovery, not even detected
        _make_event("s5", "performance", 0, detected=None, recovered=None,
                     recovery_method=None, recovery_quality=0.0,
                     accuracies=[0.8, 0.75, 0.7, 0.65]),
    ]


@pytest.fixture
def analyzer() -> RecoveryPatternAnalyzer:
    return RecoveryPatternAnalyzer()


# ------------------------------------------------------------------ #
# cluster_recovery_patterns
# ------------------------------------------------------------------ #


class TestClusterRecoveryPatterns:
    def test_clusters_mixed(
        self, analyzer: RecoveryPatternAnalyzer, mixed_events: list[RecoveryEvent]
    ) -> None:
        clusters = analyzer.cluster_recovery_patterns(mixed_events)
        assert "quick_recovery" in clusters
        assert "slow_recovery" in clusters
        assert "no_recovery" in clusters
        # s1, s2 detected at <= 3 and recovered
        assert len(clusters["quick_recovery"]) == 2
        # s3 detected at 5 (> 3) and recovered
        assert len(clusters["slow_recovery"]) == 1
        # s4, s5 not recovered
        assert len(clusters["no_recovery"]) == 2

    def test_empty_events(self, analyzer: RecoveryPatternAnalyzer) -> None:
        clusters = analyzer.cluster_recovery_patterns([])
        assert clusters["quick_recovery"] == []
        assert clusters["slow_recovery"] == []
        assert clusters["no_recovery"] == []

    def test_all_quick(self, analyzer: RecoveryPatternAnalyzer) -> None:
        events = [
            _make_event("a", detected=1, recovered=2, recovery_method="r"),
            _make_event("b", detected=0, recovered=1, recovery_method="r"),
        ]
        clusters = analyzer.cluster_recovery_patterns(events)
        assert len(clusters["quick_recovery"]) == 2
        assert len(clusters["slow_recovery"]) == 0
        assert len(clusters["no_recovery"]) == 0

    def test_all_no_recovery(self, analyzer: RecoveryPatternAnalyzer) -> None:
        events = [
            _make_event("a"),
            _make_event("b"),
        ]
        clusters = analyzer.cluster_recovery_patterns(events)
        assert len(clusters["no_recovery"]) == 2


# ------------------------------------------------------------------ #
# recovery_strategy_effectiveness
# ------------------------------------------------------------------ #


class TestRecoveryStrategyEffectiveness:
    def test_effectiveness(
        self, analyzer: RecoveryPatternAnalyzer, mixed_events: list[RecoveryEvent]
    ) -> None:
        results = analyzer.recovery_strategy_effectiveness(mixed_events)
        # "rollback" used by s1, s3
        assert "rollback" in results
        assert results["rollback"]["count"] == 2
        assert results["rollback"]["recovery_rate"] == 1.0  # both recovered
        assert results["rollback"]["avg_quality"] == pytest.approx(0.8)

        # "rewrite" used by s2
        assert "rewrite" in results
        assert results["rewrite"]["count"] == 1
        assert results["rewrite"]["avg_quality"] == pytest.approx(0.8)

        # "none" used by s4, s5 (recovery_method is None)
        assert "none" in results
        assert results["none"]["count"] == 2
        assert results["none"]["recovery_rate"] == 0.0

    def test_empty_events(self, analyzer: RecoveryPatternAnalyzer) -> None:
        results = analyzer.recovery_strategy_effectiveness([])
        assert results == {}


# ------------------------------------------------------------------ #
# time_to_recovery_distribution
# ------------------------------------------------------------------ #


class TestTimeToRecoveryDistribution:
    def test_with_recoveries(
        self, analyzer: RecoveryPatternAnalyzer, mixed_events: list[RecoveryEvent]
    ) -> None:
        stats = analyzer.time_to_recovery_distribution(mixed_events)
        # s1: latency=2, s2: latency=3, s3: latency=8
        assert stats["count"] == 3
        assert stats["mean"] is not None
        assert stats["median"] is not None
        assert stats["mean"] == pytest.approx((2 + 3 + 8) / 3)
        assert stats["std"] is not None
        assert stats["p25"] is not None
        assert stats["p75"] is not None
        assert stats["p95"] is not None

    def test_no_recoveries(self, analyzer: RecoveryPatternAnalyzer) -> None:
        events = [_make_event("a"), _make_event("b")]
        stats = analyzer.time_to_recovery_distribution(events)
        assert stats["count"] == 0
        assert stats["mean"] is None
        assert stats["median"] is None

    def test_empty(self, analyzer: RecoveryPatternAnalyzer) -> None:
        stats = analyzer.time_to_recovery_distribution([])
        assert stats["count"] == 0


# ------------------------------------------------------------------ #
# plot_recovery_trajectories
# ------------------------------------------------------------------ #


class TestPlotRecoveryTrajectories:
    def test_plot_with_events(
        self, analyzer: RecoveryPatternAnalyzer, mixed_events: list[RecoveryEvent]
    ) -> None:
        import matplotlib.pyplot as plt

        fig = analyzer.plot_recovery_trajectories(mixed_events)
        assert fig is not None
        plt.close(fig)

    def test_plot_empty_events(self, analyzer: RecoveryPatternAnalyzer) -> None:
        fig = analyzer.plot_recovery_trajectories([])
        assert fig is None

    def test_plot_saves_to_file(
        self, analyzer: RecoveryPatternAnalyzer, mixed_events: list[RecoveryEvent], tmp_path
    ) -> None:
        import matplotlib.pyplot as plt
        import os

        output_path = str(tmp_path / "trajectories.png")
        fig = analyzer.plot_recovery_trajectories(mixed_events, output_path=output_path)
        assert fig is not None
        assert os.path.exists(output_path)
        plt.close(fig)
