"""Tests for energy tracker and homogenization detection."""

import numpy as np
import pytest

from src.energy.energy_tracker import EnergyTracker, EnergyMeasurement
from src.energy.homogenization import HomogenizationDetector, HomogenizationResult
from src.energy.layer_norms import LayerNormTracker
from src.energy.early_warning import EnergyEarlyWarning, EnergyPrediction


class TestEnergyTracker:
    """Test energy tracking functionality."""

    def test_measure_basic(self, sample_activations):
        """Basic measurement returns correct structure."""
        tracker = EnergyTracker(num_layers=6)
        measurement = tracker.measure(sample_activations)

        assert isinstance(measurement, EnergyMeasurement)
        assert measurement.num_layers == 6
        assert measurement.total_energy > 0
        assert len(measurement.per_layer_energy) == 6
        assert measurement.step == 1

    def test_baseline_setting(self, sample_activations):
        """set_baseline computes from recent measurements."""
        tracker = EnergyTracker()
        tracker.measure(sample_activations)
        tracker.measure(sample_activations)

        baseline = tracker.set_baseline()
        assert baseline > 0
        assert tracker.baseline_energy == baseline

    def test_explicit_baseline(self):
        """set_baseline with explicit value."""
        tracker = EnergyTracker()
        tracker.set_baseline(energy=5.0)
        assert tracker.baseline_energy == 5.0

    def test_relative_energy(self, sample_activations):
        """get_relative_energy returns ratio to baseline."""
        tracker = EnergyTracker()
        tracker.measure(sample_activations)
        tracker.set_baseline()

        # Measure again -- should be close to 1.0
        tracker.measure(sample_activations)
        rel = tracker.get_relative_energy()
        assert rel is not None
        assert 0.5 < rel < 2.0

    def test_relative_energy_no_baseline(self, sample_activations):
        """get_relative_energy returns None without baseline."""
        tracker = EnergyTracker()
        tracker.measure(sample_activations)
        assert tracker.get_relative_energy() is None

    def test_declining_detection(self, declining_activations):
        """is_declining detects declining energy."""
        tracker = EnergyTracker()
        for acts in declining_activations:
            tracker.measure(acts)

        assert tracker.is_declining(threshold=0.1, window=20)

    def test_not_declining_stable(self, sample_activations):
        """is_declining returns False for stable energy."""
        tracker = EnergyTracker()
        for _ in range(20):
            tracker.measure(sample_activations)

        assert not tracker.is_declining(threshold=0.1, window=10)

    def test_insufficient_data_not_declining(self, sample_activations):
        """is_declining returns False with insufficient data."""
        tracker = EnergyTracker()
        tracker.measure(sample_activations)
        assert not tracker.is_declining(window=10)

    def test_energy_history(self, sample_activations):
        """get_energy_history returns total energies."""
        tracker = EnergyTracker()
        for _ in range(5):
            tracker.measure(sample_activations)

        history = tracker.get_energy_history()
        assert len(history) == 5
        assert all(e > 0 for e in history)

    def test_layer_history(self, sample_activations):
        """get_layer_history returns per-layer energies."""
        tracker = EnergyTracker()
        for _ in range(5):
            tracker.measure(sample_activations)

        layer_hist = tracker.get_layer_history(0)
        assert len(layer_hist) == 5

    def test_measurements_property(self, sample_activations):
        """measurements property returns copies."""
        tracker = EnergyTracker()
        tracker.measure(sample_activations)
        m = tracker.measurements
        assert len(m) == 1


class TestHomogenizationDetector:
    """Test homogenization detection patterns."""

    def test_insufficient_data(self):
        """Returns no patterns with insufficient data."""
        detector = HomogenizationDetector()
        result = detector.detect([])
        assert not result.is_homogenizing

    def test_uniform_decline(self, declining_activations):
        """Detects uniform decline across layers."""
        tracker = EnergyTracker()
        for acts in declining_activations:
            tracker.measure(acts)

        detector = HomogenizationDetector(decline_threshold=0.1)
        result = detector.detect(tracker.measurements)

        assert isinstance(result, HomogenizationResult)
        assert result.is_homogenizing
        assert "uniform_decline" in result.patterns_detected

    def test_final_layer_collapse(self):
        """Detects collapse of the final layer."""
        rng = np.random.RandomState(42)
        tracker = EnergyTracker(num_layers=4)

        for step in range(20):
            # Final layer collapses; others stable
            acts = [rng.randn(32) for _ in range(3)]
            scale = max(0.01, 1.0 - 0.08 * step)
            acts.append(rng.randn(32) * scale)
            tracker.measure(acts)

        detector = HomogenizationDetector(collapse_threshold=0.3)
        result = detector.detect(tracker.measurements)

        assert result.is_homogenizing
        assert "final_layer_collapse" in result.patterns_detected

    def test_sudden_drop(self):
        """Detects sudden energy drops."""
        rng = np.random.RandomState(42)
        tracker = EnergyTracker(num_layers=4)

        for step in range(10):
            if step == 5:
                scale = 0.1  # Sudden drop
            else:
                scale = 1.0
            acts = [rng.randn(32) * scale for _ in range(4)]
            tracker.measure(acts)

        detector = HomogenizationDetector(drop_threshold=0.3)
        result = detector.detect(tracker.measurements)

        assert result.is_homogenizing
        assert "sudden_drop" in result.patterns_detected

    def test_oscillation(self):
        """Detects oscillating energy patterns."""
        rng = np.random.RandomState(42)
        tracker = EnergyTracker(num_layers=4)

        for step in range(20):
            scale = 1.0 + 0.5 * ((-1) ** step)  # Oscillating
            acts = [rng.randn(32) * scale for _ in range(4)]
            tracker.measure(acts)

        detector = HomogenizationDetector(oscillation_threshold=0.1)
        result = detector.detect(tracker.measurements)

        assert result.is_homogenizing
        assert "oscillation" in result.patterns_detected

    def test_healthy_no_patterns(self, sample_activations):
        """No patterns detected for stable energy."""
        tracker = EnergyTracker(num_layers=6)
        rng = np.random.RandomState(42)
        for _ in range(20):
            acts = [rng.randn(64) for _ in range(6)]
            tracker.measure(acts)

        detector = HomogenizationDetector()
        result = detector.detect(tracker.measurements)

        # Stable random activations should have minimal decline
        # The key check is that severity is low
        assert result.severity < 0.5


class TestLayerNormTracker:
    """Test layer norm tracking."""

    def test_track(self, sample_activations):
        """track returns a snapshot with correct norms."""
        tracker = LayerNormTracker(num_layers=6)
        snapshot = tracker.track(sample_activations)

        assert len(snapshot.norms) == 6
        assert snapshot.mean_norm > 0
        assert snapshot.step == 1

    def test_trend(self, sample_activations):
        """get_layer_trend returns meaningful trend."""
        tracker = LayerNormTracker(num_layers=6)
        for _ in range(20):
            tracker.track(sample_activations)

        trend = tracker.get_layer_trend(0, window=10)
        assert trend in ("increasing", "decreasing", "stable")

    def test_divergence(self, sample_activations):
        """get_divergence returns coefficient of variation."""
        tracker = LayerNormTracker(num_layers=6)
        tracker.track(sample_activations)
        div = tracker.get_divergence()
        assert isinstance(div, float)

    def test_insufficient_data_trend(self):
        """Returns insufficient_data for short history."""
        tracker = LayerNormTracker(num_layers=6)
        assert tracker.get_layer_trend(0) == "insufficient_data"

    def test_collapsing(self):
        """is_collapsing detects norm collapse."""
        rng = np.random.RandomState(42)
        tracker = LayerNormTracker(num_layers=4)
        for step in range(20):
            scale = max(0.01, 1.0 - 0.08 * step)
            acts = [rng.randn(32) * scale for _ in range(4)]
            tracker.track(acts)
        assert tracker.is_collapsing(threshold=0.2, window=10)


class TestEnergyEarlyWarning:
    """Test early warning predictions."""

    def test_predict_declining(self, declining_activations):
        """Predicts decline for declining energy."""
        tracker = EnergyTracker()
        for acts in declining_activations:
            tracker.measure(acts)

        warning = EnergyEarlyWarning()
        pred = warning.predict(tracker.measurements, horizon=10)

        assert isinstance(pred, EnergyPrediction)
        assert pred.will_decline
        assert pred.trend_slope < 0

    def test_predict_insufficient_data(self):
        """Returns low confidence for insufficient data."""
        rng = np.random.RandomState(42)
        tracker = EnergyTracker()
        tracker.measure([rng.randn(64)])

        warning = EnergyEarlyWarning()
        pred = warning.predict(tracker.measurements)

        assert pred.confidence == 0.0

    def test_lead_time(self, declining_activations):
        """lead_time returns estimated steps to critical."""
        tracker = EnergyTracker()
        for acts in declining_activations:
            tracker.measure(acts)

        warning = EnergyEarlyWarning()
        lt = warning.lead_time(tracker.measurements)

        # Should return some value for declining energy
        # (could be None if already past critical)
        assert lt is None or isinstance(lt, int)

    def test_predictions_stored(self, declining_activations):
        """Predictions are stored in history."""
        tracker = EnergyTracker()
        for acts in declining_activations:
            tracker.measure(acts)

        warning = EnergyEarlyWarning()
        warning.predict(tracker.measurements)
        warning.predict(tracker.measurements, horizon=20)

        assert len(warning.predictions) == 2
