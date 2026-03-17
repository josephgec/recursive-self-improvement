"""Tests for behavioral/internal divergence detection."""

import numpy as np
import pytest

from src.probing.extractor import MockModel, MockModifiedModel, ActivationExtractor
from src.probing.diff import ActivationDiff, ActivationDiffResult
from src.probing.probe_set import ProbeInput
from src.anomaly.divergence_detector import (
    BehavioralInternalDivergenceDetector, DivergenceCheckResult,
)
from src.anomaly.behavioral_similarity import (
    measure_behavioral_change, measure_behavioral_change_numeric,
    measure_behavioral_change_from_magnitude,
)
from src.anomaly.internal_distance import (
    measure_internal_change, measure_internal_change_per_layer,
    measure_safety_internal_change,
)
from src.anomaly.ratio_monitor import RatioMonitor


class TestDivergenceDetector:
    """Test behavioral/internal divergence detection."""

    def test_small_internal_small_behavioral_ok(self, sample_snapshot):
        """Small internal + small behavioral change should be OK."""
        differ = ActivationDiff()
        # Compare snapshot with itself = zero change
        diff_result = differ.compute(sample_snapshot, sample_snapshot)

        detector = BehavioralInternalDivergenceDetector(ratio_threshold=3.0)
        result = detector.check(diff_result, behavioral_change=0.0, iteration=1)

        assert isinstance(result, DivergenceCheckResult)
        assert result.divergence_ratio == pytest.approx(1.0)  # Both zero -> ratio = 1
        assert not result.is_anomalous

    def test_large_internal_small_behavioral_anomalous(self, sample_snapshot, modified_snapshot):
        """Large internal + small behavioral change should be anomalous."""
        differ = ActivationDiff()
        diff_result = differ.compute(sample_snapshot, modified_snapshot)

        # Build up history with small ratios so the spike is detectable
        detector = BehavioralInternalDivergenceDetector(
            ratio_threshold=3.0, z_score_threshold=2.0
        )
        # Seed history with normal values
        normal_diff = differ.compute(sample_snapshot, sample_snapshot)
        for i in range(10):
            detector.check(normal_diff, behavioral_change=0.0, iteration=i)

        # Now check with large internal change, small behavioral change
        result = detector.check(diff_result, behavioral_change=0.001, iteration=11)

        assert result.internal_change > 0
        # With very small behavioral change, ratio should be high
        assert result.divergence_ratio > 1.0

    def test_safety_flags(self, sample_snapshot, modified_snapshot):
        """Should propagate safety flags from diff result."""
        differ = ActivationDiff()
        diff_result = differ.compute(sample_snapshot, modified_snapshot)

        detector = BehavioralInternalDivergenceDetector()
        result = detector.check(diff_result, behavioral_change=0.1, iteration=1)

        # Safety flag should match diff result
        assert result.safety_flag == diff_result.safety_disproportionate

    def test_result_to_dict(self, sample_snapshot, modified_snapshot):
        """DivergenceCheckResult should serialize."""
        differ = ActivationDiff()
        diff_result = differ.compute(sample_snapshot, modified_snapshot)

        detector = BehavioralInternalDivergenceDetector()
        result = detector.check(diff_result, behavioral_change=0.1, iteration=1)

        d = result.to_dict()
        assert "divergence_ratio" in d
        assert "is_anomalous" in d
        assert "z_score" in d
        assert "details" in d

    def test_history_tracking(self, sample_snapshot, modified_snapshot):
        """Should track ratio history."""
        differ = ActivationDiff()
        diff_result = differ.compute(sample_snapshot, modified_snapshot)

        detector = BehavioralInternalDivergenceDetector()
        for i in range(5):
            detector.check(diff_result, behavioral_change=0.1, iteration=i)

        history = detector.get_history()
        assert len(history) == 5

    def test_reset(self, sample_snapshot, modified_snapshot):
        """Should reset state."""
        differ = ActivationDiff()
        diff_result = differ.compute(sample_snapshot, modified_snapshot)

        detector = BehavioralInternalDivergenceDetector()
        detector.check(diff_result, behavioral_change=0.1, iteration=1)
        detector.reset()
        assert len(detector.get_history()) == 0
        assert detector.iteration == 0

    def test_large_behavioral_change_not_anomalous(self, sample_snapshot, modified_snapshot):
        """Large internal + large behavioral should not be anomalous (ratio near 1)."""
        differ = ActivationDiff()
        diff_result = differ.compute(sample_snapshot, modified_snapshot)
        internal_mag = diff_result.overall_change_magnitude

        detector = BehavioralInternalDivergenceDetector(ratio_threshold=3.0)
        # behavioral_change close to internal_change -> ratio near 1
        result = detector.check(
            diff_result,
            behavioral_change=internal_mag,
            iteration=1,
        )
        assert result.divergence_ratio == pytest.approx(1.0, abs=0.1)


class TestRatioMonitor:
    """Test ratio monitoring."""

    def test_record_and_get(self):
        """Should record and retrieve history."""
        monitor = RatioMonitor()
        monitor.record(1.0)
        monitor.record(2.0)
        monitor.record(3.0)
        assert monitor.get_history() == [1.0, 2.0, 3.0]

    def test_window(self):
        """Should return window of recent values."""
        monitor = RatioMonitor(window_size=3)
        for i in range(10):
            monitor.record(float(i))
        window = monitor.get_window()
        assert window == [7.0, 8.0, 9.0]

    def test_z_score_computation(self):
        """Should compute z-scores correctly."""
        monitor = RatioMonitor()
        for i in range(20):
            monitor.record(1.0)
        monitor.record(10.0)  # Spike
        z = monitor.compute_z_score()
        assert z > 2.0  # Should be very high

    def test_anomaly_detection(self):
        """Should detect anomalous values."""
        monitor = RatioMonitor(z_score_threshold=2.0)
        for i in range(20):
            monitor.record(1.0)
        monitor.record(10.0)
        assert monitor.is_anomalous()

    def test_no_anomaly(self):
        """Should not flag normal values."""
        monitor = RatioMonitor(z_score_threshold=2.0)
        rng = np.random.RandomState(42)
        for i in range(20):
            monitor.record(1.0 + rng.randn() * 0.1)
        # Value well within normal range
        monitor.record(1.05)
        assert not monitor.is_anomalous()

    def test_trend_detection(self):
        """Should detect trends."""
        monitor = RatioMonitor()
        for i in range(10):
            monitor.record(float(i))
        assert monitor.get_trend() == "increasing"

    def test_stable_trend(self):
        """Should detect stable trend."""
        monitor = RatioMonitor()
        for i in range(10):
            monitor.record(1.0)
        assert monitor.get_trend() == "stable"

    def test_mean_and_std(self):
        """Should compute mean and std."""
        monitor = RatioMonitor()
        monitor.record(1.0)
        monitor.record(3.0)
        assert monitor.get_mean() == pytest.approx(2.0)
        assert monitor.get_std() > 0

    def test_spike_detection(self):
        """Should detect spikes."""
        monitor = RatioMonitor(z_score_threshold=2.0)
        for i in range(20):
            monitor.record(1.0)
        monitor.record(10.0)
        assert monitor.detect_spike()

    def test_z_score_external_value(self):
        """Should compute z-score for external value."""
        monitor = RatioMonitor()
        for i in range(20):
            monitor.record(1.0)
        z = monitor.compute_z_score(value=5.0)
        assert z > 2.0

    def test_empty_history(self):
        """Should handle empty history."""
        monitor = RatioMonitor()
        assert monitor.get_mean() == 0.0
        assert monitor.get_std() == 0.0
        assert monitor.compute_z_score() == 0.0


class TestBehavioralSimilarity:
    """Test behavioral change measurement."""

    def test_no_change(self):
        """Identical outputs should give 0 change."""
        before = {"p1": "hello", "p2": "world"}
        after = {"p1": "hello", "p2": "world"}
        assert measure_behavioral_change(before, after) == 0.0

    def test_full_change(self):
        """All different outputs should give 1.0 change."""
        before = {"p1": "hello", "p2": "world"}
        after = {"p1": "bye", "p2": "earth"}
        assert measure_behavioral_change(before, after) == 1.0

    def test_partial_change(self):
        """Half different should give 0.5."""
        before = {"p1": "hello", "p2": "world"}
        after = {"p1": "hello", "p2": "earth"}
        assert measure_behavioral_change(before, after) == 0.5

    def test_numeric_change(self):
        """Should compute mean absolute change for numeric scores."""
        before = {"p1": 1.0, "p2": 2.0}
        after = {"p1": 1.5, "p2": 2.5}
        change = measure_behavioral_change_numeric(before, after)
        assert change == pytest.approx(0.5)

    def test_magnitude_clamping(self):
        """Should clamp magnitude to [0,1]."""
        assert measure_behavioral_change_from_magnitude(0.5) == 0.5
        assert measure_behavioral_change_from_magnitude(2.0) == 1.0
        assert measure_behavioral_change_from_magnitude(-1.0) == 0.0


class TestInternalDistance:
    """Test internal distance measurement."""

    def test_zero_diff(self, sample_snapshot):
        """Identical snapshots should give 0 internal change."""
        differ = ActivationDiff()
        diff = differ.compute(sample_snapshot, sample_snapshot)
        change = measure_internal_change(diff)
        assert change == pytest.approx(0.0)

    def test_nonzero_diff(self, sample_snapshot, modified_snapshot):
        """Modified snapshots should give nonzero change."""
        differ = ActivationDiff()
        diff = differ.compute(sample_snapshot, modified_snapshot)
        change = measure_internal_change(diff)
        assert change > 0

    def test_per_layer(self, sample_snapshot, modified_snapshot):
        """Should give per-layer changes."""
        differ = ActivationDiff()
        diff = differ.compute(sample_snapshot, modified_snapshot)
        per_layer = measure_internal_change_per_layer(diff)
        assert len(per_layer) > 0
        for layer_name, info in per_layer.items():
            assert "mean_change" in info
            assert "direction_similarity" in info

    def test_safety_internal_change(self, sample_snapshot, modified_snapshot):
        """Should measure safety-specific changes."""
        differ = ActivationDiff()
        diff = differ.compute(sample_snapshot, modified_snapshot)
        safety_change = measure_safety_internal_change(diff)
        assert isinstance(safety_change, float)

    def test_empty_diff(self):
        """Should handle empty diff."""
        diff = ActivationDiffResult()
        assert measure_internal_change(diff) == 0.0
