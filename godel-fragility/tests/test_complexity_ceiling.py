"""Tests for complexity ceiling analysis."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from src.analysis.complexity_ceiling import (
    CeilingAnalysis,
    ComplexityCeilingAnalyzer,
    SigmoidFit,
)


@pytest.fixture
def analyzer() -> ComplexityCeilingAnalyzer:
    return ComplexityCeilingAnalyzer()


# ------------------------------------------------------------------ #
# SigmoidFit / CeilingAnalysis dataclasses
# ------------------------------------------------------------------ #


class TestDataclasses:
    def test_sigmoid_fit_defaults(self) -> None:
        sf = SigmoidFit()
        assert sf.L == 1.0
        assert sf.k == -0.01
        assert sf.x0 == 100.0
        assert sf.b == 0.0
        assert sf.r_squared == 0.0

    def test_ceiling_analysis_defaults(self) -> None:
        ca = CeilingAnalysis()
        assert ca.ceiling_estimate is None
        assert ca.sigmoid_fit is None
        assert ca.cliff_detected is False
        assert ca.cliff_location is None
        assert ca.decline_characterization == "unknown"
        assert ca.safe_operating_range is None
        assert ca.data_points == 0
        assert ca.complexity_range is None


# ------------------------------------------------------------------ #
# analyze() full pipeline
# ------------------------------------------------------------------ #


class TestAnalyze:
    def test_insufficient_data(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        result = analyzer.analyze([10.0], [0.9])
        assert result.data_points == 1
        assert result.ceiling_estimate is None
        assert result.sigmoid_fit is None

    def test_two_points(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        result = analyzer.analyze([10.0, 20.0], [0.9, 0.8])
        assert result.data_points == 2
        # < 3 points, returns early
        assert result.complexity_range is None

    def test_sigmoid_data(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        """Synthetic sigmoid: accuracy drops from ~1.0 to ~0.0 around x=50."""
        np.random.seed(42)
        x = np.linspace(10, 100, 20)
        # True sigmoid: y = 1 / (1 + exp(0.1*(x - 50)))
        y = 1.0 / (1.0 + np.exp(0.1 * (x - 50)))
        y += np.random.normal(0, 0.02, len(y))
        y = np.clip(y, 0, 1)

        result = analyzer.analyze(x.tolist(), y.tolist())
        assert result.data_points == 20
        assert result.complexity_range is not None
        assert result.complexity_range[0] == pytest.approx(10.0)
        assert result.complexity_range[1] == pytest.approx(100.0)
        # Should find a ceiling around 50
        assert result.ceiling_estimate is not None
        assert 20.0 <= result.ceiling_estimate <= 80.0

    def test_cliff_data(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        """Accuracy stays high then drops suddenly."""
        complexities = list(range(10, 110, 10))
        accuracies = [0.95, 0.93, 0.94, 0.92, 0.91, 0.90, 0.88, 0.40, 0.20, 0.10]
        result = analyzer.analyze(complexities, accuracies)
        assert result.cliff_detected is True
        assert result.cliff_location is not None
        # Cliff should be around complexity 70-80
        assert 60 <= result.cliff_location <= 90

    def test_no_decline_data(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        """Accuracy stays flat -- no decline."""
        complexities = list(range(10, 60, 1))
        accuracies = [0.90 + np.random.uniform(-0.01, 0.01) for _ in complexities]
        result = analyzer.analyze(complexities, accuracies)
        assert result.decline_characterization == "no_decline"

    def test_gradual_decline(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        """Accuracy decreases gradually."""
        complexities = list(range(10, 60, 1))
        accuracies = [0.95 - 0.015 * i for i in range(len(complexities))]
        result = analyzer.analyze(complexities, accuracies)
        assert result.decline_characterization in ("gradual", "cliff")

    def test_fallback_ceiling_from_threshold(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        """When no sigmoid or cliff, ceiling from accuracy < 0.5 threshold."""
        # Flat data that stays above 0.5 -- no ceiling found from sigmoid
        complexities = [10.0, 20.0, 30.0, 40.0, 50.0]
        accuracies = [0.6, 0.55, 0.52, 0.45, 0.30]
        result = analyzer.analyze(complexities, accuracies)
        # Should find ceiling where accuracy first drops below 0.5
        assert result.ceiling_estimate is not None


# ------------------------------------------------------------------ #
# fit_sigmoid
# ------------------------------------------------------------------ #


class TestFitSigmoid:
    def test_fewer_than_5_points(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        result = analyzer.fit_sigmoid([1, 2, 3], [0.9, 0.8, 0.7])
        assert result is None

    def test_constant_x(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        result = analyzer.fit_sigmoid([5, 5, 5, 5, 5], [0.9, 0.8, 0.7, 0.6, 0.5])
        assert result is None

    def test_valid_sigmoid(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        x = np.linspace(0, 100, 20).tolist()
        y = [1.0 / (1.0 + np.exp(0.1 * (xi - 50))) for xi in x]
        result = analyzer.fit_sigmoid(x, y)
        assert result is not None
        assert isinstance(result, SigmoidFit)
        assert result.r_squared > 0.3

    def test_constant_y(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        """When all y values are equal, L_range should fallback to [1.0]."""
        x = list(range(10, 60, 1))
        y = [0.5] * len(x)
        result = analyzer.fit_sigmoid(x, y)
        # Should still return a fit (though with r_squared = 0)
        assert result is not None


# ------------------------------------------------------------------ #
# detect_cliff
# ------------------------------------------------------------------ #


class TestDetectCliff:
    def test_no_cliff(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        complexities = [10, 20, 30, 40, 50]
        accuracies = [0.90, 0.88, 0.87, 0.86, 0.85]
        result = analyzer.detect_cliff(complexities, accuracies)
        assert result is None

    def test_clear_cliff(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        complexities = [10, 20, 30, 40, 50]
        accuracies = [0.90, 0.88, 0.85, 0.30, 0.20]
        result = analyzer.detect_cliff(complexities, accuracies)
        assert result is not None
        # Cliff at complexity 30 (drop from 0.85 to 0.30)
        assert result == 30

    def test_custom_threshold(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        complexities = [10, 20, 30, 40, 50]
        accuracies = [0.90, 0.88, 0.85, 0.70, 0.60]
        # Default threshold 0.2 -- drop of 0.15 not enough
        result_default = analyzer.detect_cliff(complexities, accuracies, threshold=0.2)
        assert result_default is None
        # Lower threshold -- drop of 0.15 is enough
        result_low = analyzer.detect_cliff(complexities, accuracies, threshold=0.1)
        assert result_low is not None

    def test_insufficient_data(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        result = analyzer.detect_cliff([10], [0.9])
        assert result is None


# ------------------------------------------------------------------ #
# characterize_decline
# ------------------------------------------------------------------ #


class TestCharacterizeDecline:
    def test_insufficient_data(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        result = analyzer.characterize_decline([10, 20], [0.9, 0.8])
        assert result == "insufficient_data"

    def test_no_decline(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        complexities = list(range(10, 60))
        accuracies = [0.90] * len(complexities)
        result = analyzer.characterize_decline(complexities, accuracies)
        assert result == "no_decline"

    def test_plateau_then_cliff(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        complexities = list(range(10, 20))
        # First half flat, second half big drop
        accuracies = [0.90, 0.90, 0.91, 0.90, 0.90, 0.89, 0.60, 0.40, 0.30, 0.20]
        result = analyzer.characterize_decline(complexities, accuracies)
        assert result == "plateau_then_cliff"


# ------------------------------------------------------------------ #
# compute_safe_operating_range
# ------------------------------------------------------------------ #


class TestSafeOperatingRange:
    def test_empty_data(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        result = analyzer.compute_safe_operating_range([], [])
        assert result is None

    def test_all_below_threshold(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        result = analyzer.compute_safe_operating_range([10, 20, 30], [0.3, 0.2, 0.1])
        assert result is None

    def test_valid_range(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        complexities = [10, 20, 30, 40, 50]
        accuracies = [0.95, 0.90, 0.80, 0.60, 0.30]
        result = analyzer.compute_safe_operating_range(complexities, accuracies)
        assert result is not None
        assert result == (10, 30)

    def test_custom_min_accuracy(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        complexities = [10, 20, 30, 40, 50]
        accuracies = [0.95, 0.90, 0.80, 0.60, 0.30]
        result = analyzer.compute_safe_operating_range(
            complexities, accuracies, min_accuracy=0.5
        )
        assert result is not None
        assert result == (10, 40)


# ------------------------------------------------------------------ #
# plot_ceiling_analysis
# ------------------------------------------------------------------ #


class TestPlotCeilingAnalysis:
    def test_plot_with_data(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        import matplotlib.pyplot as plt

        x = np.linspace(10, 100, 20).tolist()
        y = [1.0 / (1.0 + np.exp(0.1 * (xi - 50))) for xi in x]
        analysis = analyzer.analyze(x, y)
        fig = analyzer.plot_ceiling_analysis(analysis, x, y)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_no_sigmoid(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        import matplotlib.pyplot as plt

        analysis = CeilingAnalysis(data_points=3)
        fig = analyzer.plot_ceiling_analysis(analysis, [10, 20, 30], [0.9, 0.8, 0.7])
        assert fig is not None
        plt.close(fig)

    def test_plot_with_safe_range(self, analyzer: ComplexityCeilingAnalyzer) -> None:
        import matplotlib.pyplot as plt

        analysis = CeilingAnalysis(
            safe_operating_range=(10, 50),
            ceiling_estimate=60.0,
        )
        fig = analyzer.plot_ceiling_analysis(analysis, [10, 20, 30], [0.9, 0.8, 0.7])
        assert fig is not None
        plt.close(fig)

    def test_plot_saves_to_file(self, analyzer: ComplexityCeilingAnalyzer, tmp_path) -> None:
        import matplotlib.pyplot as plt

        analysis = CeilingAnalysis(data_points=3)
        output_path = str(tmp_path / "ceiling.png")
        fig = analyzer.plot_ceiling_analysis(
            analysis, [10, 20, 30], [0.9, 0.8, 0.7], output_path=output_path
        )
        assert fig is not None
        import os
        assert os.path.exists(output_path)
        plt.close(fig)
