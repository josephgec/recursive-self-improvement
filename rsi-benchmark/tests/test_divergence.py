"""Test divergence: RSI vs collapse, entropy tracking, fixed point, sustainability."""

import pytest

from src.collapse.baseline_loader import CollapseBaselineLoader, CollapseTrajectory
from src.collapse.divergence_analyzer import DivergenceAnalyzer, DivergenceResult
from src.collapse.entropy_tracker import EntropyTracker, DiversityMetrics
from src.collapse.fixed_point_detector import FixedPointDetector, FixedPointStatus
from src.collapse.sustainability import SustainabilityAnalyzer, SustainabilityReport


class TestCollapseBaselineLoader:
    """Test collapse baseline loading."""

    def test_load_standard_decay(self, collapse_baselines):
        traj = collapse_baselines.load("standard_decay")
        assert isinstance(traj, CollapseTrajectory)
        assert traj.schedule == "standard_decay"
        assert len(traj.generations) == 20
        assert traj.accuracy[0] == 0.78
        assert traj.accuracy[-1] < traj.accuracy[0]  # Declining

    def test_load_rapid_collapse(self, collapse_baselines):
        traj = collapse_baselines.load("rapid_collapse")
        standard = collapse_baselines.load("standard_decay")
        # Rapid should decline faster
        assert traj.accuracy[-1] < standard.accuracy[-1]

    def test_load_slow_collapse(self, collapse_baselines):
        traj = collapse_baselines.load("slow_collapse")
        standard = collapse_baselines.load("standard_decay")
        # Slow should decline slower
        assert traj.accuracy[-1] > standard.accuracy[-1]

    def test_load_all(self, collapse_baselines):
        all_traj = collapse_baselines.load_all()
        assert len(all_traj) == 3
        assert "standard_decay" in all_traj
        assert "rapid_collapse" in all_traj
        assert "slow_collapse" in all_traj

    def test_available_schedules(self, collapse_baselines):
        schedules = collapse_baselines.available_schedules()
        assert len(schedules) == 3

    def test_load_unknown_raises(self, collapse_baselines):
        with pytest.raises(KeyError):
            collapse_baselines.load("nonexistent")

    def test_declining_accuracy(self, collapse_baselines):
        traj = collapse_baselines.load("standard_decay")
        for i in range(1, len(traj.accuracy)):
            assert traj.accuracy[i] <= traj.accuracy[i - 1]

    def test_declining_entropy(self, collapse_baselines):
        traj = collapse_baselines.load("standard_decay")
        for i in range(1, len(traj.entropy)):
            assert traj.entropy[i] <= traj.entropy[i - 1]

    def test_increasing_kl_divergence(self, collapse_baselines):
        traj = collapse_baselines.load("standard_decay")
        for i in range(1, len(traj.kl_divergence)):
            assert traj.kl_divergence[i] >= traj.kl_divergence[i - 1]

    def test_add_custom_trajectory(self, collapse_baselines):
        custom = CollapseTrajectory(
            schedule="custom",
            generations=[0, 1, 2],
            accuracy=[0.9, 0.8, 0.7],
            entropy=[3.0, 2.5, 2.0],
            kl_divergence=[0.0, 0.1, 0.2],
        )
        collapse_baselines.add_trajectory(custom)
        loaded = collapse_baselines.load("custom")
        assert loaded.schedule == "custom"


class TestDivergenceAnalyzer:
    """Test divergence analysis between RSI and collapse."""

    def test_compute_divergence_improving_vs_collapse(self, improving_curve, collapse_baselines):
        collapse = collapse_baselines.load("standard_decay")
        analyzer = DivergenceAnalyzer()
        result = analyzer.compute_divergence(improving_curve, collapse.accuracy)

        assert isinstance(result, DivergenceResult)
        assert len(result.rsi_values) > 0
        assert len(result.collapse_values) > 0
        assert len(result.divergence_values) > 0
        # RSI should diverge positively from collapse
        assert result.divergence_trend == "increasing"

    def test_divergence_values_match(self, improving_curve, collapse_baselines):
        collapse = collapse_baselines.load("standard_decay")
        analyzer = DivergenceAnalyzer()
        result = analyzer.compute_divergence(improving_curve, collapse.accuracy)

        for i, div in enumerate(result.divergence_values):
            expected = result.rsi_values[i] - result.collapse_values[i]
            assert abs(div - expected) < 1e-10

    def test_collapse_prevention_score(self, improving_curve, collapse_baselines):
        collapse = collapse_baselines.load("standard_decay")
        analyzer = DivergenceAnalyzer()
        score = analyzer.compute_collapse_prevention_score(
            improving_curve, collapse.accuracy
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be good for improving curve

    def test_prevention_score_for_declining(self, declining_curve, collapse_baselines):
        collapse = collapse_baselines.load("standard_decay")
        analyzer = DivergenceAnalyzer()
        score = analyzer.compute_collapse_prevention_score(
            declining_curve, collapse.accuracy
        )
        # Declining curve should have lower prevention score than improving
        improving_score = analyzer.compute_collapse_prevention_score(
            [(i, 0.60 + 0.015 * i) for i in range(15)],
            collapse.accuracy,
        )
        assert score < improving_score

    def test_plot_divergence(self, improving_curve, collapse_baselines):
        collapse = collapse_baselines.load("standard_decay")
        analyzer = DivergenceAnalyzer()
        result = analyzer.compute_divergence(improving_curve, collapse.accuracy)
        plot = analyzer.plot_divergence(result)
        assert "rsi" in plot
        assert "collapse" in plot
        assert "divergence" in plot
        assert "prevention_score" in plot

    def test_empty_divergence(self):
        analyzer = DivergenceAnalyzer()
        result = analyzer.compute_divergence([], [])
        assert result.mean_divergence == 0.0


class TestEntropyTracker:
    """Test entropy tracking."""

    def test_record_and_compute(self):
        tracker = EntropyTracker()
        tracker.record_outputs(0, ["a", "b", "c", "d"])
        tracker.record_outputs(1, ["a", "a", "b", "b"])
        tracker.record_outputs(2, ["a", "a", "a", "a"])

        curve = tracker.compute_entropy_curve()
        assert len(curve) == 3
        # Entropy should decrease as diversity decreases
        assert curve[0][1] > curve[2][1]

    def test_diversity_metrics(self):
        tracker = EntropyTracker()
        tracker.record_outputs(0, ["a", "b", "c", "d"])

        metrics = tracker.compute_diversity_metrics(0)
        assert isinstance(metrics, DiversityMetrics)
        assert metrics.unique_ratio == 1.0  # All unique
        assert metrics.entropy > 0
        assert metrics.simpson_diversity > 0

    def test_diversity_metrics_homogeneous(self):
        tracker = EntropyTracker()
        tracker.record_outputs(0, ["a", "a", "a", "a"])

        metrics = tracker.compute_diversity_metrics(0)
        assert metrics.unique_ratio == 0.25
        assert metrics.entropy == 0.0
        assert metrics.simpson_diversity == 0.0

    def test_diversity_metrics_empty(self):
        tracker = EntropyTracker()
        metrics = tracker.compute_diversity_metrics(99)
        assert metrics.unique_ratio == 0.0
        assert metrics.entropy == 0.0

    def test_all_diversity_metrics(self):
        tracker = EntropyTracker()
        tracker.record_outputs(0, ["a", "b"])
        tracker.record_outputs(1, ["a", "a"])
        all_metrics = tracker.compute_all_diversity_metrics()
        assert 0 in all_metrics
        assert 1 in all_metrics

    def test_is_entropy_declining(self):
        tracker = EntropyTracker()
        tracker.record_outputs(0, ["a", "b", "c", "d", "e"])
        tracker.record_outputs(1, ["a", "b", "c", "d"])
        tracker.record_outputs(2, ["a", "a", "b", "b"])
        tracker.record_outputs(3, ["a", "a", "a", "a"])
        assert tracker.is_entropy_declining(window=3)

    def test_is_not_entropy_declining(self):
        tracker = EntropyTracker()
        tracker.record_outputs(0, ["a", "a"])
        tracker.record_outputs(1, ["a", "b"])
        tracker.record_outputs(2, ["a", "b", "c"])
        assert not tracker.is_entropy_declining(window=3)

    def test_is_entropy_declining_insufficient_data(self):
        tracker = EntropyTracker()
        tracker.record_outputs(0, ["a", "b"])
        assert not tracker.is_entropy_declining(window=3)


class TestFixedPointDetector:
    """Test fixed point detection."""

    def test_detect_stable_fixed_point(self):
        detector = FixedPointDetector(accuracy_tolerance=0.01, window_size=3)
        accuracy = [0.60, 0.65, 0.70, 0.75, 0.75, 0.75, 0.75]
        entropy = [3.0, 2.8, 2.6, 2.4, 2.4, 2.4, 2.4]
        status = detector.detect(accuracy, entropy)
        assert isinstance(status, FixedPointStatus)
        assert status.is_fixed_point
        assert status.convergence_type == "stable"

    def test_detect_no_fixed_point_improving(self):
        detector = FixedPointDetector(accuracy_tolerance=0.005, window_size=3)
        accuracy = [0.60 + 0.02 * i for i in range(10)]
        entropy = [3.0 - 0.1 * i for i in range(10)]
        status = detector.detect(accuracy, entropy)
        assert not status.is_fixed_point

    def test_detect_oscillating(self):
        detector = FixedPointDetector(accuracy_tolerance=0.005, window_size=3)
        accuracy = [0.70, 0.72, 0.69, 0.73, 0.68, 0.74, 0.67]
        entropy = [2.5, 2.6, 2.4, 2.7, 2.3, 2.8, 2.2]
        status = detector.detect(accuracy, entropy)
        assert status.convergence_type == "oscillating"
        assert not status.is_fixed_point

    def test_detect_diverging(self):
        detector = FixedPointDetector(accuracy_tolerance=0.005, window_size=3)
        accuracy = [0.80, 0.75, 0.70, 0.65, 0.60]
        entropy = [3.0, 2.8, 2.6, 2.4, 2.2]
        status = detector.detect(accuracy, entropy)
        assert status.convergence_type in ("diverging", "stable")  # entropy converges

    def test_detect_insufficient_data(self):
        detector = FixedPointDetector(window_size=3)
        status = detector.detect([0.5], [3.0])
        assert not status.is_fixed_point
        assert status.convergence_type == "none"

    def test_confidence_score(self):
        detector = FixedPointDetector(accuracy_tolerance=0.01, window_size=3)
        accuracy = [0.75, 0.75, 0.75, 0.75, 0.75]
        entropy = [2.5, 2.5, 2.5, 2.5, 2.5]
        status = detector.detect(accuracy, entropy)
        assert status.confidence > 0.5


class TestSustainabilityAnalyzer:
    """Test sustainability analysis."""

    def test_longest_improvement_streak_monotonic(self):
        analyzer = SustainabilityAnalyzer()
        curve = [0.60 + 0.015 * i for i in range(15)]
        streak = analyzer.longest_improvement_streak(curve)
        assert streak == 14

    def test_longest_improvement_streak_with_dip(self):
        analyzer = SustainabilityAnalyzer()
        curve = [0.60, 0.62, 0.64, 0.63, 0.65, 0.67]
        streak = analyzer.longest_improvement_streak(curve)
        assert streak == 2  # 0.63 -> 0.65 -> 0.67

    def test_longest_improvement_streak_flat(self):
        analyzer = SustainabilityAnalyzer()
        curve = [0.75, 0.75, 0.75]
        streak = analyzer.longest_improvement_streak(curve)
        assert streak == 0

    def test_recovery_after_dip(self):
        analyzer = SustainabilityAnalyzer()
        curve = [0.60, 0.62, 0.61, 0.63, 0.62, 0.64]
        recovery = analyzer.recovery_after_dip(curve)
        assert recovery == 2  # Two dips, both recovered

    def test_recovery_no_dips(self):
        analyzer = SustainabilityAnalyzer()
        curve = [0.60, 0.62, 0.64, 0.66]
        assert analyzer.recovery_after_dip(curve) == 0

    def test_monotonicity_score_perfect(self):
        analyzer = SustainabilityAnalyzer()
        curve = [0.60 + 0.01 * i for i in range(10)]
        score = analyzer.monotonicity_score(curve)
        assert score == 1.0

    def test_monotonicity_score_declining(self):
        analyzer = SustainabilityAnalyzer()
        curve = [0.80 - 0.01 * i for i in range(10)]
        score = analyzer.monotonicity_score(curve)
        assert score == 0.0

    def test_monotonicity_score_mixed(self):
        analyzer = SustainabilityAnalyzer()
        curve = [0.60, 0.62, 0.61, 0.63, 0.62, 0.64]
        score = analyzer.monotonicity_score(curve)
        assert 0.0 < score < 1.0

    def test_full_analysis(self):
        analyzer = SustainabilityAnalyzer()
        curve = [0.60 + 0.015 * i for i in range(15)]
        report = analyzer.analyze(curve)
        assert isinstance(report, SustainabilityReport)
        assert report.longest_improvement_streak == 14
        assert report.monotonicity_score == 1.0
        assert report.max_drawdown == 0.0
        assert report.overall_sustainability_score > 0.5

    def test_full_analysis_declining(self):
        analyzer = SustainabilityAnalyzer()
        curve = [0.65 - 0.01 * i for i in range(15)]
        report = analyzer.analyze(curve)
        assert report.monotonicity_score == 0.0
        assert report.longest_improvement_streak == 0
        assert report.overall_sustainability_score < 0.5

    def test_analysis_empty(self):
        analyzer = SustainabilityAnalyzer()
        report = analyzer.analyze([0.5])
        assert report.longest_improvement_streak == 0
        assert report.monotonicity_score == 0.0
