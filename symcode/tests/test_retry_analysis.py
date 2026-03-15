"""Tests for retry / self-correction analysis."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.analysis.retry_analysis import RetryAnalyzer
from src.verification.result_types import (
    AttemptRecord,
    CodeError,
    CodeExecutionResult,
    SolveResult,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_attempt(
    num: int,
    success: bool = True,
    answer_correct: bool | None = None,
    error_type: str | None = None,
    error_msg: str = "",
) -> AttemptRecord:
    """Build an AttemptRecord for testing."""
    error = None
    if error_type:
        error = CodeError(error_type=error_type, message=error_msg)
    return AttemptRecord(
        attempt_number=num,
        code=f"answer = {num}",
        execution_result=CodeExecutionResult(
            success=success,
            answer=str(num) if success else None,
            error=error,
        ),
        extracted_answer=str(num) if success else None,
        answer_correct=answer_correct,
    )


def _make_solve(
    correct: bool,
    attempts: list[AttemptRecord],
) -> SolveResult:
    return SolveResult(
        problem="test problem",
        expected_answer="42",
        final_answer="42" if correct else "wrong",
        correct=correct,
        num_attempts=len(attempts),
        attempts=attempts,
    )


# ── correction_success_by_error_type ────────────────────────────────


class TestCorrectionSuccessByErrorType:
    def test_syntax_error_corrected(self):
        """SyntaxError followed by success should count as correction."""
        a1 = _make_attempt(1, success=False, error_type="SyntaxError")
        a2 = _make_attempt(2, success=True, answer_correct=True)
        results = [_make_solve(correct=True, attempts=[a1, a2])]

        stats = RetryAnalyzer.correction_success_by_error_type(results)
        assert "SyntaxError" in stats
        assert stats["SyntaxError"]["attempts"] == 1
        assert stats["SyntaxError"]["successes"] == 1
        assert stats["SyntaxError"]["rate"] == pytest.approx(1.0)

    def test_multiple_error_types(self):
        """Multiple error types tracked separately."""
        a1 = _make_attempt(1, success=False, error_type="SyntaxError")
        a2 = _make_attempt(2, success=False, error_type="NameError")
        a3 = _make_attempt(3, success=True, answer_correct=True)
        results = [_make_solve(correct=True, attempts=[a1, a2, a3])]

        stats = RetryAnalyzer.correction_success_by_error_type(results)
        assert "SyntaxError" in stats
        assert "NameError" in stats
        # SyntaxError -> NameError (failed correction)
        assert stats["SyntaxError"]["successes"] == 0
        # NameError -> success
        assert stats["NameError"]["successes"] == 1

    def test_wrong_answer_tracked(self):
        """Wrong answer (no error) should be tracked as WrongAnswer."""
        a1 = _make_attempt(1, success=True, answer_correct=False)
        a2 = _make_attempt(2, success=True, answer_correct=True)
        results = [_make_solve(correct=True, attempts=[a1, a2])]

        stats = RetryAnalyzer.correction_success_by_error_type(results)
        assert "WrongAnswer" in stats
        assert stats["WrongAnswer"]["attempts"] == 1
        assert stats["WrongAnswer"]["successes"] == 1

    def test_no_correction_opportunities(self):
        """Single-attempt results should produce empty stats."""
        a1 = _make_attempt(1, success=True, answer_correct=True)
        results = [_make_solve(correct=True, attempts=[a1])]

        stats = RetryAnalyzer.correction_success_by_error_type(results)
        assert len(stats) == 0

    def test_empty_results(self):
        stats = RetryAnalyzer.correction_success_by_error_type([])
        assert len(stats) == 0

    def test_successful_attempt_skipped(self):
        """If attempt succeeded and answer is not False, skip it."""
        a1 = _make_attempt(1, success=True, answer_correct=True)
        a2 = _make_attempt(2, success=True, answer_correct=True)
        results = [_make_solve(correct=True, attempts=[a1, a2])]

        stats = RetryAnalyzer.correction_success_by_error_type(results)
        # a1 had no error and answer_correct is True, so skip
        assert len(stats) == 0


# ── optimal_k ────────────────────────────────────────────────────────


class TestOptimalK:
    def test_optimal_k_single_attempt(self):
        """All problems solved on first attempt; optimal k should be 1."""
        a1 = _make_attempt(1, success=True, answer_correct=True)
        results = [_make_solve(correct=True, attempts=[a1])] * 5

        opt = RetryAnalyzer.optimal_k(results, max_k=5)
        assert opt["optimal_k"] == 1
        assert opt["accuracy_at_optimal"] == pytest.approx(1.0)
        assert len(opt["analysis"]) == 5

    def test_optimal_k_benefits_from_retries(self):
        """Some problems benefit from retries."""
        # Problem 1: solved on attempt 1
        a1_ok = _make_attempt(1, success=True, answer_correct=True)
        r1 = _make_solve(correct=True, attempts=[a1_ok])

        # Problem 2: solved on attempt 2
        a2_fail = _make_attempt(1, success=False, error_type="SyntaxError")
        a2_ok = _make_attempt(2, success=True, answer_correct=True)
        r2 = _make_solve(correct=True, attempts=[a2_fail, a2_ok])

        results = [r1, r2]
        opt = RetryAnalyzer.optimal_k(results, max_k=3)
        assert opt["optimal_k"] >= 1
        assert opt["accuracy_at_optimal"] > 0

    def test_optimal_k_empty(self):
        opt = RetryAnalyzer.optimal_k([], max_k=5)
        assert opt["optimal_k"] == 1
        assert opt["accuracy_at_optimal"] == 0.0
        assert opt["analysis"] == []

    def test_optimal_k_analysis_structure(self):
        a1 = _make_attempt(1, success=True, answer_correct=True)
        results = [_make_solve(correct=True, attempts=[a1])]

        opt = RetryAnalyzer.optimal_k(results, max_k=3)
        for entry in opt["analysis"]:
            assert "k" in entry
            assert "accuracy" in entry
            assert "marginal_gain" in entry
            assert "utility" in entry

    def test_optimal_k_cost_weight(self):
        """Higher cost_per_attempt should favor fewer retries."""
        a1_fail = _make_attempt(1, success=False, error_type="NameError")
        a2_ok = _make_attempt(2, success=True, answer_correct=True)
        results = [_make_solve(correct=True, attempts=[a1_fail, a2_ok])]

        opt_low_cost = RetryAnalyzer.optimal_k(
            results, max_k=5, cost_per_attempt=0.01
        )
        opt_high_cost = RetryAnalyzer.optimal_k(
            results, max_k=5, cost_per_attempt=10.0
        )
        assert opt_high_cost["optimal_k"] <= opt_low_cost["optimal_k"]


# ── plot_retry_funnel ────────────────────────────────────────────────


class TestPlotRetryFunnel:
    def test_plot_returns_figure(self):
        a1 = _make_attempt(1, success=True, answer_correct=True)
        a2_fail = _make_attempt(1, success=False, error_type="SyntaxError")
        a2_ok = _make_attempt(2, success=True, answer_correct=True)

        results = [
            _make_solve(correct=True, attempts=[a1]),
            _make_solve(correct=True, attempts=[a2_fail, a2_ok]),
            _make_solve(correct=False, attempts=[
                _make_attempt(1, success=False, error_type="NameError"),
                _make_attempt(2, success=False, error_type="NameError"),
            ]),
        ]

        try:
            import matplotlib
            fig = RetryAnalyzer.plot_retry_funnel(results, max_k=3)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            fig = RetryAnalyzer.plot_retry_funnel(results, max_k=3)
            assert fig is None

    def test_plot_empty_results(self):
        fig = RetryAnalyzer.plot_retry_funnel([], max_k=3)
        assert fig is None

    def test_plot_with_output_path(self, tmp_path):
        a1 = _make_attempt(1, success=True, answer_correct=True)
        results = [_make_solve(correct=True, attempts=[a1])]

        try:
            import matplotlib
            output = str(tmp_path / "funnel.png")
            fig = RetryAnalyzer.plot_retry_funnel(
                results, output_path=output, max_k=3
            )
            assert fig is not None
            import os
            assert os.path.exists(output)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pass

    def test_plot_without_matplotlib(self):
        a1 = _make_attempt(1, success=True, answer_correct=True)
        results = [_make_solve(correct=True, attempts=[a1])]

        with patch.dict("sys.modules", {"matplotlib": None}):
            fig = RetryAnalyzer.plot_retry_funnel(results, max_k=3)
            assert fig is None


# ── correction_trajectory ────────────────────────────────────────────


class TestCorrectionTrajectory:
    def test_trajectory_improvement(self):
        """Cumulative accuracy should increase with more attempts."""
        a1_fail = _make_attempt(1, success=False, error_type="SyntaxError")
        a2_ok = _make_attempt(2, success=True, answer_correct=True)

        results = [
            # Solved on attempt 1
            _make_solve(correct=True, attempts=[
                _make_attempt(1, success=True, answer_correct=True)
            ]),
            # Solved on attempt 2
            _make_solve(correct=True, attempts=[a1_fail, a2_ok]),
        ]

        traj = RetryAnalyzer.correction_trajectory(results, max_k=3)
        assert len(traj["cumulative_accuracy"]) == 3
        assert traj["cumulative_accuracy"][0] == pytest.approx(0.5)  # 1/2 at k=1
        assert traj["cumulative_accuracy"][1] == pytest.approx(1.0)  # 2/2 at k=2

    def test_trajectory_empty(self):
        traj = RetryAnalyzer.correction_trajectory([], max_k=3)
        assert traj["cumulative_accuracy"] == []
        assert traj["marginal_gain"] == []
        assert traj["problems_solved_at_k"] == []

    def test_trajectory_marginal_gain(self):
        a1_ok = _make_attempt(1, success=True, answer_correct=True)
        results = [_make_solve(correct=True, attempts=[a1_ok])] * 3

        traj = RetryAnalyzer.correction_trajectory(results, max_k=3)
        # All solved on k=1, so marginal gain at k=2,3 should be 0
        assert traj["marginal_gain"][0] == 0.0  # first is always 0
        assert traj["marginal_gain"][1] == pytest.approx(0.0)
        assert traj["marginal_gain"][2] == pytest.approx(0.0)
