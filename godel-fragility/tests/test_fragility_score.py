"""Tests for the fragility scoring module."""

from __future__ import annotations

import pytest

from src.analysis.fragility_score import FragilityReport, FragilityScorer
from src.harness.stress_runner import ScenarioResult, StressTestResults
from src.measurement.failure_classifier import FailureMode
from src.measurement.recovery_tracker import RecoveryTracker


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_results(
    pass_rate: float = 0.5,
    catastrophic: int = 0,
    total: int = 10,
) -> StressTestResults:
    passed = int(total * pass_rate)
    failed = total - passed
    results = []
    for i in range(passed):
        results.append(
            ScenarioResult(
                scenario_name=f"pass_{i}",
                category="test",
                severity=2,
                success=True,
                failure_mode=FailureMode.VALIDATION_CAUGHT,
                iterations_run=5,
                duration_seconds=1.0,
            )
        )
    for i in range(failed):
        mode = FailureMode.SELF_LOBOTOMY if i < catastrophic else FailureMode.STAGNATION
        results.append(
            ScenarioResult(
                scenario_name=f"fail_{i}",
                category="test",
                severity=3,
                success=False,
                failure_mode=mode,
                iterations_run=5,
                duration_seconds=1.0,
            )
        )
    return StressTestResults(
        results=results,
        total_scenarios=total,
        passed=passed,
        failed=failed,
        duration_seconds=float(total),
    )


def _make_tracker(
    total: int = 5,
    recovered: int = 3,
    detected: int = 4,
) -> RecoveryTracker:
    tracker = RecoveryTracker()
    for i in range(total):
        tracker.track_injection(f"event_{i}", "syntax", iteration=0)
        if i < detected:
            tracker.update(1, 0.5, detected=True, detection_method="accuracy_drop")
        if i < recovered:
            tracker.update(
                2, 0.85,
                recovered=True,
                recovery_method="rollback",
                recovery_quality=0.9,
            )
        tracker.finalize_event()
    return tracker


@pytest.fixture
def scorer() -> FragilityScorer:
    return FragilityScorer()


# ------------------------------------------------------------------ #
# compute
# ------------------------------------------------------------------ #


class TestCompute:
    def test_basic_compute(self, scorer: FragilityScorer) -> None:
        results = _make_results(pass_rate=0.5, total=10)
        report = scorer.compute(results)
        assert isinstance(report, FragilityReport)
        assert 0.0 <= report.overall_score <= 1.0
        assert report.interpretation
        assert report.grade
        assert report.recommendations

    def test_robust_agent(self, scorer: FragilityScorer) -> None:
        results = _make_results(pass_rate=1.0, total=10)
        tracker = _make_tracker(total=5, recovered=5, detected=5)
        report = scorer.compute(
            results,
            recovery_tracker=tracker,
            complexity_ceiling=200.0,
            max_complexity_tested=100.0,
        )
        # Low fragility score for robust agent
        assert report.overall_score < 0.3
        assert report.grade in ("A+", "A", "B")

    def test_fragile_agent(self, scorer: FragilityScorer) -> None:
        results = _make_results(pass_rate=0.1, catastrophic=5, total=10)
        tracker = _make_tracker(total=5, recovered=0, detected=1)
        report = scorer.compute(
            results,
            recovery_tracker=tracker,
            complexity_ceiling=30.0,
            max_complexity_tested=200.0,
        )
        # High fragility score
        assert report.overall_score > 0.5

    def test_no_recovery_tracker(self, scorer: FragilityScorer) -> None:
        results = _make_results(pass_rate=0.5, total=10)
        report = scorer.compute(results)
        # Should use pass_rate as proxy for recovery
        assert "recovery_rate" in report.components

    def test_no_ceiling_data(self, scorer: FragilityScorer) -> None:
        results = _make_results(pass_rate=0.5, total=10)
        report = scorer.compute(results)
        # Ceiling ratio should default to 0.5
        assert report.components["ceiling_ratio"] == pytest.approx(0.5)

    def test_with_ceiling_data(self, scorer: FragilityScorer) -> None:
        results = _make_results(pass_rate=0.5, total=10)
        report = scorer.compute(
            results,
            complexity_ceiling=50.0,
            max_complexity_tested=100.0,
        )
        # ceiling/max = 0.5, fragility = 1 - 0.5 = 0.5
        assert report.components["ceiling_ratio"] == pytest.approx(0.5)


# ------------------------------------------------------------------ #
# component_breakdown
# ------------------------------------------------------------------ #


class TestComponentBreakdown:
    def test_recovery_rate_from_tracker(self, scorer: FragilityScorer) -> None:
        results = _make_results(pass_rate=0.5, total=10)
        tracker = _make_tracker(total=5, recovered=3, detected=5)
        components = scorer.component_breakdown(results, recovery_tracker=tracker)
        # Recovery rate = 3/5 = 0.6, fragility = 1 - 0.6 = 0.4
        assert components["recovery_rate"] == pytest.approx(0.4)

    def test_detection_rate_from_tracker(self, scorer: FragilityScorer) -> None:
        results = _make_results(pass_rate=0.5, total=10)
        tracker = _make_tracker(total=5, recovered=0, detected=4)
        components = scorer.component_breakdown(results, recovery_tracker=tracker)
        # Detection rate = 4/5 = 0.8, fragility = 1 - 0.8 = 0.2
        assert components["detection_rate"] == pytest.approx(0.2)

    def test_catastrophic_rate(self, scorer: FragilityScorer) -> None:
        results = _make_results(pass_rate=0.5, catastrophic=3, total=10)
        components = scorer.component_breakdown(results)
        assert components["catastrophic_rate"] == pytest.approx(3 / 10)

    def test_detection_from_validation(self, scorer: FragilityScorer) -> None:
        """When no tracker, detection uses validation_caught count."""
        results = _make_results(pass_rate=0.5, total=10)
        # The passing ones have VALIDATION_CAUGHT failure mode
        components = scorer.component_breakdown(results)
        # 5 passing have VALIDATION_CAUGHT mode, so 5/10 = 0.5
        assert components["detection_rate"] == pytest.approx(0.5)


# ------------------------------------------------------------------ #
# interpret
# ------------------------------------------------------------------ #


class TestInterpret:
    def test_all_ranges(self, scorer: FragilityScorer) -> None:
        for score, expected_substring in [
            (0.1, "strong robustness"),
            (0.3, "moderate robustness"),
            (0.5, "notable fragility"),
            (0.7, "highly fragile"),
            (0.9, "critically fragile"),
        ]:
            interpretation = scorer.interpret(score)
            assert expected_substring in interpretation.lower()


# ------------------------------------------------------------------ #
# _grade
# ------------------------------------------------------------------ #


class TestGrade:
    def test_grade_mapping(self, scorer: FragilityScorer) -> None:
        assert scorer._grade(0.05) == "A+"
        assert scorer._grade(0.15) == "A"
        assert scorer._grade(0.25) == "B"
        assert scorer._grade(0.35) == "B-"
        assert scorer._grade(0.45) == "C"
        assert scorer._grade(0.55) == "C-"
        assert scorer._grade(0.65) == "D"
        assert scorer._grade(0.75) == "D-"
        assert scorer._grade(0.85) == "F"


# ------------------------------------------------------------------ #
# _recommendations
# ------------------------------------------------------------------ #


class TestRecommendations:
    def test_recovery_rec(self, scorer: FragilityScorer) -> None:
        recs = scorer._recommendations({"recovery_rate": 0.6})
        assert any("recovery" in r.lower() for r in recs)

    def test_ceiling_rec(self, scorer: FragilityScorer) -> None:
        recs = scorer._recommendations({"ceiling_ratio": 0.6})
        assert any("ceiling" in r.lower() for r in recs)

    def test_catastrophic_rec(self, scorer: FragilityScorer) -> None:
        recs = scorer._recommendations({"catastrophic_rate": 0.4})
        assert any("catastrophic" in r.lower() for r in recs)

    def test_detection_rec(self, scorer: FragilityScorer) -> None:
        recs = scorer._recommendations({"detection_rate": 0.6})
        assert any("detection" in r.lower() for r in recs)

    def test_no_issues(self, scorer: FragilityScorer) -> None:
        recs = scorer._recommendations({
            "recovery_rate": 0.1,
            "ceiling_ratio": 0.1,
            "catastrophic_rate": 0.1,
            "detection_rate": 0.1,
        })
        assert any("adequate" in r.lower() for r in recs)
