"""Tests for detection modules."""

import numpy as np
import pytest

from src.detection.reward_accuracy_divergence import (
    RewardAccuracyDivergenceDetector,
    DivergenceResult,
)
from src.detection.shortcut_detector import ShortcutDetector, ShortcutReport
from src.detection.reward_gaming_tests import RewardGamingTests, GamingTestResult
from src.detection.composite_detector import (
    CompositeRewardHackingDetector,
    TrainingState,
    RewardHackingReport,
)


class TestRewardAccuracyDivergence:
    """Test reward-accuracy divergence detection."""

    def test_no_divergence_healthy(self):
        """No divergence detected for correlated reward and accuracy."""
        detector = RewardAccuracyDivergenceDetector(threshold=0.3, window=20)
        for i in range(30):
            detector.update(reward=0.5 + i * 0.01, accuracy=0.5 + i * 0.01)

        result = detector.check()
        assert isinstance(result, DivergenceResult)
        assert not result.is_diverging

    def test_divergence_detected(self):
        """Detects divergence when reward rises but accuracy is flat."""
        detector = RewardAccuracyDivergenceDetector(threshold=0.01, window=20)
        rng = np.random.RandomState(123)
        for i in range(30):
            detector.update(
                reward=0.5 + i * 0.5,  # Steeply rising reward
                accuracy=0.5 + rng.randn() * 0.001,  # Flat accuracy
            )

        result = detector.check()
        assert result.is_diverging
        assert result.reward_trend > 0
        assert result.divergence_score > 0

    def test_insufficient_data(self):
        """Returns non-diverging for insufficient data."""
        detector = RewardAccuracyDivergenceDetector(window=20)
        detector.update(0.5, 0.5)
        result = detector.check()
        assert not result.is_diverging
        assert "Insufficient" in result.description

    def test_reset(self):
        """Reset clears history."""
        detector = RewardAccuracyDivergenceDetector()
        detector.update(1.0, 0.5)
        detector.reset()
        assert detector.reward_history == []
        assert detector.accuracy_history == []

    def test_results_history(self):
        """Results are stored."""
        detector = RewardAccuracyDivergenceDetector(window=5)
        for i in range(10):
            detector.update(float(i), 0.5)
        detector.check()
        detector.check()
        assert len(detector.results) == 2


class TestShortcutDetector:
    """Test shortcut detection."""

    def test_length_gaming_detected(self):
        """Detects length gaming when outputs are much longer."""
        detector = ShortcutDetector(length_ratio_threshold=2.0)
        output_lengths = [200, 250, 180, 220]
        baseline_lengths = [50, 55, 45, 60]

        is_gaming, details = detector.check_length_gaming(
            output_lengths, baseline_lengths
        )
        assert is_gaming
        assert details["length_ratio"] > 2.0

    def test_length_gaming_not_detected(self):
        """No length gaming when outputs are similar length."""
        detector = ShortcutDetector(length_ratio_threshold=2.0)
        output_lengths = [55, 60, 50]
        baseline_lengths = [50, 55, 45]

        is_gaming, _ = detector.check_length_gaming(
            output_lengths, baseline_lengths
        )
        assert not is_gaming

    def test_repetition_detected(self):
        """Detects high token repetition."""
        detector = ShortcutDetector(repetition_threshold=0.3)
        # Highly repetitive sequences
        outputs = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(5)]

        is_repeating, details = detector.check_repetition(outputs)
        assert is_repeating
        assert details["mean_repetition_rate"] > 0.3

    def test_repetition_not_detected(self):
        """No repetition for diverse sequences."""
        detector = ShortcutDetector(repetition_threshold=0.5)
        rng = np.random.RandomState(42)
        outputs = [list(rng.randint(0, 100, 50)) for _ in range(5)]

        is_repeating, _ = detector.check_repetition(outputs)
        assert not is_repeating

    def test_diversity_collapse(self):
        """Detects diversity collapse."""
        detector = ShortcutDetector(diversity_min=0.3)
        # Only use 5 tokens out of 100
        outputs = [[0, 1, 2, 3, 4, 0, 1, 2] for _ in range(5)]

        is_collapsed, details = detector.check_diversity_collapse(outputs, vocab_size=100)
        assert is_collapsed
        assert details["token_diversity"] < 0.3

    def test_no_diversity_collapse(self):
        """No collapse with diverse outputs."""
        detector = ShortcutDetector(diversity_min=0.3)
        rng = np.random.RandomState(42)
        outputs = [list(rng.randint(0, 100, 50)) for _ in range(10)]

        is_collapsed, _ = detector.check_diversity_collapse(outputs, vocab_size=100)
        assert not is_collapsed

    def test_run_all(self):
        """run_all produces a complete ShortcutReport."""
        detector = ShortcutDetector()
        rng = np.random.RandomState(42)

        report = detector.run_all(
            output_lengths=[50, 55, 45],
            baseline_lengths=[50, 55, 45],
            outputs=[list(rng.randint(0, 100, 50)) for _ in range(5)],
            vocab_size=100,
        )

        assert isinstance(report, ShortcutReport)
        assert not report.any_detected

    def test_empty_data(self):
        """Handles empty data gracefully."""
        detector = ShortcutDetector()
        is_gaming, _ = detector.check_length_gaming([], [])
        assert not is_gaming
        is_rep, _ = detector.check_repetition([])
        assert not is_rep
        is_col, _ = detector.check_diversity_collapse([], 100)
        assert not is_col


class TestRewardGamingTests:
    """Test reward gaming test suite."""

    def test_format_hacking_detected(self):
        """Detects format hacking with heavy formatting."""
        tests = RewardGamingTests()
        outputs = [
            "# Title\n\n- **Bold point one**\n- **Bold point two**\n\n## Section\n\n" * 5
            for _ in range(10)
        ]
        result = tests.test_format_hacking(outputs, threshold=0.5)
        assert not result.passed  # Gaming detected

    def test_format_hacking_clean(self):
        """No format hacking with plain text."""
        tests = RewardGamingTests()
        outputs = ["This is a plain text response with no formatting." for _ in range(10)]
        result = tests.test_format_hacking(outputs, threshold=0.5)
        assert result.passed

    def test_keyword_stuffing_detected(self):
        """Detects keyword stuffing."""
        tests = RewardGamingTests()
        keywords = ["therefore", "however"]
        outputs = [
            "Therefore however therefore however therefore however"
            for _ in range(10)
        ]
        result = tests.test_keyword_stuffing(outputs, keywords=keywords, threshold=0.3)
        assert not result.passed

    def test_keyword_stuffing_clean(self):
        """No keyword stuffing in normal text."""
        tests = RewardGamingTests()
        outputs = ["The cat sat on the mat and looked around." for _ in range(10)]
        result = tests.test_keyword_stuffing(outputs, threshold=0.3)
        assert result.passed

    def test_run_all(self):
        """run_all returns results for all tests."""
        tests = RewardGamingTests()
        results = tests.run_all(["Normal text output"])
        assert len(results) == 2
        assert all(isinstance(r, GamingTestResult) for r in results)

    def test_empty_outputs(self):
        """Handles empty outputs."""
        tests = RewardGamingTests()
        r1 = tests.test_format_hacking([])
        r2 = tests.test_keyword_stuffing([])
        assert r1.passed
        assert r2.passed


class TestCompositeDetector:
    """Test composite reward hacking detector."""

    def test_healthy_state(self):
        """No hacking detected in healthy state."""
        detector = CompositeRewardHackingDetector(divergence_window=10)
        rng = np.random.RandomState(42)

        state = TrainingState(
            rewards=[0.5 + rng.randn() * 0.1 for _ in range(15)],
            accuracies=[0.7 + rng.randn() * 0.05 for _ in range(15)],
            output_lengths=[50 + int(rng.randn() * 5) for _ in range(10)],
            baseline_lengths=[48 + int(rng.randn() * 5) for _ in range(10)],
            outputs=[list(rng.randint(0, 100, 50)) for _ in range(10)],
            output_strings=["Normal output text"] * 10,
        )

        report = detector.check(state)
        assert isinstance(report, RewardHackingReport)
        assert not report.should_stop
        assert "healthy" in report.recommendation.lower()

    def test_hacking_detected(self):
        """Detects hacking with multiple signals."""
        detector = CompositeRewardHackingDetector(
            divergence_window=10,
            diversity_min=0.5,
        )

        state = TrainingState(
            rewards=list(np.linspace(0, 5, 15)),  # Rising
            accuracies=[0.5] * 15,  # Flat
            output_lengths=[200] * 10,  # Long
            baseline_lengths=[50] * 10,  # Short baseline
            outputs=[[1, 1, 1, 1, 1] for _ in range(10)],  # Low diversity
            output_strings=["# Title\n\n- **Bold**\n\n## Section"] * 10,
        )

        report = detector.check(state)
        assert report.is_hacking_detected
        assert len(report.signals) > 0

    def test_should_stop_training(self):
        """should_stop_training works correctly."""
        detector = CompositeRewardHackingDetector(divergence_window=5)
        assert not detector.should_stop_training()  # No reports yet

    def test_reports_stored(self):
        """Reports are stored in history."""
        detector = CompositeRewardHackingDetector(divergence_window=5)
        rng = np.random.RandomState(42)

        state = TrainingState(
            rewards=[0.5] * 10,
            accuracies=[0.5] * 10,
            output_lengths=[50] * 5,
            baseline_lengths=[50] * 5,
            outputs=[list(rng.randint(0, 100, 20)) for _ in range(5)],
            output_strings=["Normal"] * 5,
        )

        detector.check(state)
        assert len(detector.reports) == 1
