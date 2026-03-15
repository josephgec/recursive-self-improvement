"""Tests for difficulty scaling analysis."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.analysis.difficulty_scaling import DifficultyScaler, ScalingCurve
from src.verification.result_types import (
    AttemptRecord,
    CodeExecutionResult,
    SolveResult,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_result(correct: bool, difficulty: int = 0) -> SolveResult:
    """Build a SolveResult with a _difficulty attribute."""
    r = SolveResult(
        problem="test",
        correct=correct,
        num_attempts=1,
        attempts=[
            AttemptRecord(
                attempt_number=1,
                code="answer = 1",
                execution_result=CodeExecutionResult(success=True, answer="1"),
                answer_correct=correct,
            )
        ],
    )
    r._difficulty = difficulty
    return r


# ── ScalingCurve dataclass ───────────────────────────────────────────


class TestScalingCurve:
    def test_defaults(self):
        curve = ScalingCurve()
        assert curve.difficulties == []
        assert curve.accuracies == []
        assert curve.counts == []
        assert curve.pipeline == ""


# ── compute_scaling_curve ────────────────────────────────────────────


class TestComputeScalingCurve:
    def setup_method(self):
        self.scaler = DifficultyScaler()

    def test_with_explicit_difficulties(self):
        results = [
            _make_result(correct=True),
            _make_result(correct=False),
            _make_result(correct=True),
            _make_result(correct=True),
        ]
        difficulties = [1, 1, 2, 3]
        curve = self.scaler.compute_scaling_curve(
            results, difficulties=difficulties, pipeline="test"
        )
        assert curve.pipeline == "test"
        assert curve.difficulties == [1, 2, 3]
        assert curve.accuracies[0] == pytest.approx(0.5)  # 1 of 2 at diff=1
        assert curve.accuracies[1] == pytest.approx(1.0)  # 1 of 1 at diff=2
        assert curve.accuracies[2] == pytest.approx(1.0)  # 1 of 1 at diff=3
        assert curve.counts == [2, 1, 1]

    def test_with_attribute_difficulties(self):
        results = [
            _make_result(correct=True, difficulty=1),
            _make_result(correct=False, difficulty=1),
            _make_result(correct=True, difficulty=3),
        ]
        curve = self.scaler.compute_scaling_curve(results)
        assert curve.difficulties == [1, 3]
        assert curve.accuracies[0] == pytest.approx(0.5)
        assert curve.accuracies[1] == pytest.approx(1.0)

    def test_empty_results(self):
        curve = self.scaler.compute_scaling_curve([])
        assert curve.difficulties == []
        assert curve.accuracies == []
        assert curve.counts == []

    def test_all_same_difficulty(self):
        results = [
            _make_result(correct=True, difficulty=2),
            _make_result(correct=True, difficulty=2),
            _make_result(correct=False, difficulty=2),
        ]
        curve = self.scaler.compute_scaling_curve(results)
        assert len(curve.difficulties) == 1
        assert curve.difficulties[0] == 2
        assert curve.accuracies[0] == pytest.approx(2 / 3)
        assert curve.counts[0] == 3


# ── plot_scaling_curve ───────────────────────────────────────────────


class TestPlotScalingCurve:
    def setup_method(self):
        self.scaler = DifficultyScaler()

    def test_plot_returns_figure(self):
        curve = ScalingCurve(
            difficulties=[1, 2, 3],
            accuracies=[1.0, 0.75, 0.5],
            counts=[10, 8, 6],
            pipeline="test",
        )
        try:
            import matplotlib
            fig = self.scaler.plot_scaling_curve([curve])
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            # matplotlib not installed, verify graceful fallback
            fig = self.scaler.plot_scaling_curve([curve])
            assert fig is None

    def test_plot_with_output_path(self, tmp_path):
        curve = ScalingCurve(
            difficulties=[1, 2],
            accuracies=[0.9, 0.6],
            counts=[5, 5],
            pipeline="test",
        )
        try:
            import matplotlib
            output = str(tmp_path / "scaling.png")
            fig = self.scaler.plot_scaling_curve(
                [curve], output_path=output, title="Test Plot"
            )
            assert fig is not None
            import os
            assert os.path.exists(output)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pass

    def test_plot_multiple_curves(self):
        c1 = ScalingCurve(
            difficulties=[1, 2, 3],
            accuracies=[1.0, 0.8, 0.5],
            counts=[10, 10, 10],
            pipeline="symcode",
        )
        c2 = ScalingCurve(
            difficulties=[1, 2, 3],
            accuracies=[0.8, 0.6, 0.3],
            counts=[10, 10, 10],
            pipeline="prose",
        )
        try:
            import matplotlib
            fig = self.scaler.plot_scaling_curve([c1, c2])
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pass

    def test_plot_without_matplotlib(self):
        """Verify graceful fallback when matplotlib is not available."""
        curve = ScalingCurve(
            difficulties=[1, 2],
            accuracies=[0.9, 0.6],
            counts=[5, 5],
            pipeline="test",
        )
        with patch.dict("sys.modules", {"matplotlib": None}):
            # Force ImportError
            scaler = DifficultyScaler()
            fig = scaler.plot_scaling_curve([curve])
            assert fig is None


# ── test_scaling_hypothesis ──────────────────────────────────────────


class TestScalingHypothesis:
    def setup_method(self):
        self.scaler = DifficultyScaler()

    def test_decreasing_accuracy(self):
        """Accuracy that decreases with difficulty."""
        results = []
        # Diff 1: 3/3 correct
        for _ in range(3):
            results.append(_make_result(correct=True, difficulty=1))
        # Diff 2: 2/3 correct
        for i in range(3):
            results.append(_make_result(correct=(i < 2), difficulty=2))
        # Diff 3: 1/3 correct
        for i in range(3):
            results.append(_make_result(correct=(i < 1), difficulty=3))

        result = self.scaler.test_scaling_hypothesis(results)
        assert result["monotonically_decreasing"] is True
        assert result["spearman_rho"] is not None
        assert result["spearman_rho"] < 0  # negative correlation
        assert result["p_value"] is not None

    def test_not_enough_levels(self):
        """Less than 3 difficulty levels should return message."""
        results = [
            _make_result(correct=True, difficulty=1),
            _make_result(correct=False, difficulty=2),
        ]
        result = self.scaler.test_scaling_hypothesis(results)
        assert result["spearman_rho"] is None
        assert result["p_value"] is None
        assert result["monotonically_decreasing"] is None
        assert "Not enough" in result["message"]

    def test_non_monotonic(self):
        """Non-monotonically decreasing should be detected."""
        results = []
        # Diff 1: 1/2 correct
        results.append(_make_result(correct=True, difficulty=1))
        results.append(_make_result(correct=False, difficulty=1))
        # Diff 2: 2/2 correct (increase!)
        results.append(_make_result(correct=True, difficulty=2))
        results.append(_make_result(correct=True, difficulty=2))
        # Diff 3: 1/2 correct
        results.append(_make_result(correct=True, difficulty=3))
        results.append(_make_result(correct=False, difficulty=3))

        result = self.scaler.test_scaling_hypothesis(results)
        assert result["monotonically_decreasing"] is False


class TestSpearmanFallback:
    """Test the manual Spearman computation (when scipy is unavailable)."""

    def test_manual_spearman_negative_correlation(self):
        scaler = DifficultyScaler()
        x = [1, 2, 3, 4, 5]
        y = [1.0, 0.8, 0.6, 0.4, 0.2]
        rho, p_value = scaler._spearman(x, y)
        assert rho is not None
        assert rho < 0
        assert p_value is not None

    def test_manual_spearman_too_few_points(self):
        scaler = DifficultyScaler()
        rho, p_value = scaler._spearman([1, 2], [0.5, 0.3])
        assert rho is None
        assert p_value is None

    def test_manual_spearman_perfect_correlation(self):
        scaler = DifficultyScaler()
        x = [1, 2, 3, 4, 5]
        y = [0.2, 0.4, 0.6, 0.8, 1.0]  # perfect positive
        rho, p_value = scaler._spearman(x, y)
        assert rho is not None
        assert rho == pytest.approx(1.0)

    def test_manual_spearman_without_scipy(self):
        """Force fallback to manual implementation."""
        scaler = DifficultyScaler()
        x = [1, 2, 3, 4, 5]
        y = [1.0, 0.9, 0.7, 0.3, 0.1]
        with patch.dict("sys.modules", {"scipy": None, "scipy.stats": None}):
            rho, p_value = scaler._spearman(x, y)
            assert rho is not None
            assert rho < 0
            assert p_value is not None
