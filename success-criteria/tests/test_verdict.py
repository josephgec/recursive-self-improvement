"""Tests for verdict determination, partial analysis, and recommendations."""

import pytest

from src.criteria.base import CriterionResult, Evidence
from src.evaluation.evaluator import CriteriaEvaluator
from src.verdict.verdict import (
    FinalVerdict,
    SuccessVerdict,
    VerdictCategory,
)
from src.verdict.partial_success import PartialSuccessAnalyzer
from src.verdict.recommendations import RecommendationGenerator
from tests.conftest import (
    build_passing_evidence,
    build_partial_evidence,
    build_failing_evidence,
)


def _make_result(passed: bool, name: str = "", margin: float = 0.0) -> CriterionResult:
    """Helper to create a CriterionResult."""
    return CriterionResult(
        passed=passed,
        confidence=0.9 if passed else 0.4,
        measured_value=0,
        threshold=0,
        margin=margin,
        criterion_name=name or ("Passed" if passed else "Failed"),
    )


class TestSuccessVerdict:
    """Test verdict determination."""

    def test_all_pass_is_success(self):
        """5/5 passed should be SUCCESS."""
        results = [_make_result(True, f"C{i}") for i in range(5)]
        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)

        assert verdict.category == VerdictCategory.SUCCESS
        assert verdict.n_passed == 5
        assert verdict.is_go is True

    def test_three_pass_is_partial(self):
        """3/5 passed should be PARTIAL."""
        results = [
            _make_result(True, "C1"),
            _make_result(True, "C2"),
            _make_result(True, "C3"),
            _make_result(False, "C4", margin=-2.0),
            _make_result(False, "C5", margin=-3.0),
        ]
        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)

        assert verdict.category == VerdictCategory.PARTIAL
        assert verdict.n_passed == 3
        assert verdict.is_go is False

    def test_four_pass_is_partial(self):
        """4/5 passed should be PARTIAL."""
        results = [_make_result(True, f"C{i}") for i in range(4)]
        results.append(_make_result(False, "C5", margin=-1.0))
        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)

        assert verdict.category == VerdictCategory.PARTIAL
        assert verdict.n_passed == 4

    def test_one_pass_is_not_met(self):
        """1/5 passed should be NOT_MET."""
        results = [_make_result(False, f"C{i}", margin=-5.0) for i in range(4)]
        results.append(_make_result(True, "C5"))
        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)

        assert verdict.category == VerdictCategory.NOT_MET
        assert verdict.n_passed == 1
        assert verdict.is_go is False

    def test_two_pass_is_not_met(self):
        """2/5 passed should be NOT_MET."""
        results = [
            _make_result(True, "C1"),
            _make_result(True, "C2"),
            _make_result(False, "C3", margin=-3.0),
            _make_result(False, "C4", margin=-4.0),
            _make_result(False, "C5", margin=-5.0),
        ]
        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)

        assert verdict.category == VerdictCategory.NOT_MET
        assert verdict.n_passed == 2

    def test_summary_format(self):
        """Summary should contain key information."""
        results = [_make_result(True, f"C{i}") for i in range(5)]
        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)
        summary = verdict.summary()

        assert "SUCCESS" in summary
        assert "5/5" in summary

    def test_passed_and_failed_criteria(self):
        """Check passed_criteria and failed_criteria properties."""
        results = [
            _make_result(True, "C1"),
            _make_result(False, "C2", margin=-1.0),
            _make_result(True, "C3"),
        ]
        verdict_engine = SuccessVerdict(
            success_threshold=3, partial_min=2, partial_max=2
        )
        verdict = verdict_engine.evaluate(results)

        assert len(verdict.passed_criteria) == 2
        assert len(verdict.failed_criteria) == 1

    def test_overall_confidence(self):
        """Overall confidence should be average of individual confidences."""
        results = [
            _make_result(True, "C1"),  # conf=0.9
            _make_result(False, "C2", margin=-1.0),  # conf=0.4
        ]
        verdict_engine = SuccessVerdict(
            success_threshold=2, partial_min=1, partial_max=1
        )
        verdict = verdict_engine.evaluate(results)

        expected = (0.9 + 0.4) / 2.0
        assert abs(verdict.overall_confidence - expected) < 1e-6

    def test_empty_results(self):
        """Empty results list should be NOT_MET."""
        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate([])

        assert verdict.category == VerdictCategory.NOT_MET
        assert verdict.n_passed == 0
        assert verdict.overall_confidence == 0.0


class TestPartialSuccessAnalyzer:
    """Test partial success analysis."""

    def test_analyze_partial(self):
        """Analyze a PARTIAL verdict."""
        results = [
            _make_result(True, "C1"),
            _make_result(True, "C2"),
            _make_result(True, "C3"),
            _make_result(False, "C4", margin=-2.0),
            _make_result(False, "C5", margin=-5.0),
        ]
        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)

        analyzer = PartialSuccessAnalyzer()
        analysis = analyzer.analyze(verdict)

        assert analysis["n_passed"] == 3
        assert analysis["gap"] == 2
        assert len(analysis["failed_criteria"]) == 2

    def test_closest_to_passing(self):
        """Should identify the criterion closest to passing."""
        results = [
            _make_result(False, "C1", margin=-1.0),  # closest
            _make_result(False, "C2", margin=-5.0),
            _make_result(False, "C3", margin=-10.0),
        ]
        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)

        analyzer = PartialSuccessAnalyzer()
        closest = analyzer.closest_to_passing(verdict.failed_criteria)

        assert closest is not None
        assert closest["criterion"] == "C1"
        assert closest["shortfall"] == 1.0

    def test_no_failed_criteria(self):
        """Closest to passing should return None when nothing failed."""
        analyzer = PartialSuccessAnalyzer()
        assert analyzer.closest_to_passing([]) is None

    def test_estimate_remaining_work(self):
        """Remaining work estimates should have required fields."""
        results = [
            _make_result(False, "Sustained Improvement", margin=-3.0),
            _make_result(False, "Publication Acceptance", margin=-1.0),
        ]
        analyzer = PartialSuccessAnalyzer()
        estimates = analyzer.estimate_remaining_work(results)

        assert len(estimates) == 2
        for est in estimates:
            assert "criterion" in est
            assert "deficit" in est
            assert "difficulty" in est
            assert "estimated_phases_needed" in est
            assert "recommendation" in est


class TestRecommendationGenerator:
    """Test recommendation generation."""

    def test_success_recommendations(self):
        """SUCCESS verdict should generate proceed recommendations."""
        results = [_make_result(True, f"C{i}") for i in range(5)]
        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)

        gen = RecommendationGenerator()
        recs = gen.generate(verdict)

        assert len(recs) > 0
        assert any("PROCEED" in r for r in recs)

    def test_partial_recommendations(self):
        """PARTIAL verdict should generate action items."""
        results = [
            _make_result(True, "C1"),
            _make_result(True, "C2"),
            _make_result(True, "C3"),
            _make_result(False, "C4", margin=-2.0),
            _make_result(False, "C5", margin=-3.0),
        ]
        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)

        gen = RecommendationGenerator()
        recs = gen.generate(verdict)

        assert any("CONDITIONAL" in r or "ACTION" in r for r in recs)

    def test_not_met_recommendations(self):
        """NOT_MET verdict should generate remediation recommendations."""
        results = [
            _make_result(True, "C1"),
            _make_result(False, "C2", margin=-5.0),
            _make_result(False, "C3", margin=-5.0),
            _make_result(False, "C4", margin=-5.0),
            _make_result(False, "C5", margin=-5.0),
        ]
        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)

        gen = RecommendationGenerator()
        recs = gen.generate(verdict)

        assert any("NO-GO" in r for r in recs)
        assert any("REMEDIATE" in r for r in recs)

    def test_low_confidence_warning(self):
        """Low overall confidence should trigger a warning."""
        results = [
            CriterionResult(
                passed=True, confidence=0.3, measured_value=0,
                threshold=0, margin=0, criterion_name=f"C{i}"
            )
            for i in range(5)
        ]
        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)

        gen = RecommendationGenerator()
        recs = gen.generate(verdict)

        assert any("WARNING" in r or "confidence" in r.lower() for r in recs)
