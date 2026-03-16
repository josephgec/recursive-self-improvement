"""Tests for GDI calibration."""

import pytest

from src.composite.gdi import GoalDriftIndex
from src.calibration.collapse_calibrator import CollapseCalibrator, CalibratedThresholds
from src.calibration.threshold_optimizer import ThresholdOptimizer
from src.calibration.roc_analysis import compute_roc_curve, compute_auc, find_best_threshold


class TestCollapseCalibrator:
    """Tests for CollapseCalibrator."""

    def test_calibrate_from_collapse_data(
        self, reference_outputs, collapsed_outputs
    ):
        """Calibration should produce valid thresholds from collapse data."""
        gdi = GoalDriftIndex()
        calibrator = CollapseCalibrator()

        collapse_data = [
            {"outputs": reference_outputs, "reference": reference_outputs, "health": "healthy"},
            {"outputs": collapsed_outputs, "reference": reference_outputs, "health": "collapsed"},
        ]

        thresholds = calibrator.calibrate(gdi, collapse_data)

        assert isinstance(thresholds, CalibratedThresholds)
        assert 0 < thresholds.green_max < thresholds.yellow_max
        assert thresholds.yellow_max <= thresholds.orange_max
        assert thresholds.orange_max <= thresholds.red_min
        assert 0 <= thresholds.auc <= 1.0

    def test_calibrate_high_auc(
        self, reference_outputs, collapsed_outputs
    ):
        """Calibration on clear healthy/collapsed split should yield high AUC."""
        gdi = GoalDriftIndex()
        calibrator = CollapseCalibrator()

        collapse_data = [
            {"outputs": reference_outputs, "reference": reference_outputs, "health": "healthy"},
            {"outputs": reference_outputs, "reference": reference_outputs, "health": "healthy"},
            {"outputs": collapsed_outputs, "reference": reference_outputs, "health": "collapsed"},
            {"outputs": collapsed_outputs, "reference": reference_outputs, "health": "collapsed"},
        ]

        thresholds = calibrator.calibrate(gdi, collapse_data)
        assert thresholds.auc >= 0.8

    def test_calibrate_empty_data(self):
        """Empty data should return defaults."""
        gdi = GoalDriftIndex()
        calibrator = CollapseCalibrator()
        thresholds = calibrator.calibrate(gdi, [])

        assert thresholds.green_max == 0.15
        assert thresholds.yellow_max == 0.40

    def test_calibrate_metadata(
        self, reference_outputs, collapsed_outputs
    ):
        """Calibration should include metadata."""
        gdi = GoalDriftIndex()
        calibrator = CollapseCalibrator()

        collapse_data = [
            {"outputs": reference_outputs, "reference": reference_outputs, "health": "healthy"},
            {"outputs": collapsed_outputs, "reference": reference_outputs, "health": "collapsed"},
        ]

        thresholds = calibrator.calibrate(gdi, collapse_data)
        assert "optimal_binary_threshold" in thresholds.metadata
        assert "num_samples" in thresholds.metadata
        assert thresholds.metadata["num_samples"] == 2


class TestThresholdOptimizer:
    """Tests for ThresholdOptimizer."""

    def test_find_optimal_threshold(self):
        """Should find a reasonable threshold."""
        opt = ThresholdOptimizer()
        values = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]
        labels = [0, 0, 0, 1, 1, 1]
        threshold = opt.find_optimal_threshold(values, labels)
        assert 0.2 <= threshold <= 0.7

    def test_plot_roc_data(self):
        """Should return complete ROC data."""
        opt = ThresholdOptimizer()
        values = [0.1, 0.2, 0.7, 0.8]
        labels = [0, 0, 1, 1]
        data = opt.plot_roc(values, labels)

        assert "fpr" in data
        assert "tpr" in data
        assert "auc" in data
        assert "optimal_threshold" in data
        assert 0 <= data["auc"] <= 1.0


class TestROCAnalysis:
    """Tests for ROC analysis functions."""

    def test_compute_roc_curve(self):
        """ROC curve should have correct shape."""
        scores = [0.1, 0.4, 0.6, 0.9]
        labels = [0, 0, 1, 1]
        fpr, tpr, thresholds = compute_roc_curve(scores, labels)

        assert len(fpr) == len(tpr)
        assert fpr[0] == 0.0
        assert tpr[0] == 0.0

    def test_compute_auc_perfect(self):
        """Perfect classifier should have AUC = 1.0."""
        fpr = [0.0, 0.0, 1.0]
        tpr = [0.0, 1.0, 1.0]
        auc = compute_auc(fpr, tpr)
        assert auc == 1.0

    def test_compute_auc_random(self):
        """Random classifier should have AUC ~ 0.5."""
        fpr = [0.0, 0.5, 1.0]
        tpr = [0.0, 0.5, 1.0]
        auc = compute_auc(fpr, tpr)
        assert abs(auc - 0.5) < 0.01

    def test_find_best_threshold(self):
        """Should find threshold maximizing Youden's J."""
        fpr = [0.0, 0.0, 0.5, 1.0]
        tpr = [0.0, 1.0, 1.0, 1.0]
        thresholds = [1.0, 0.5, 0.3, 0.1]
        best = find_best_threshold(fpr, tpr, thresholds)
        assert best == 0.5  # J = 1.0 - 0.0 = 1.0 at this point

    def test_roc_empty(self):
        """Empty inputs should return default curve."""
        fpr, tpr, thresholds = compute_roc_curve([], [])
        assert len(fpr) == 2

    def test_roc_all_positive(self):
        """All positive labels should return default curve."""
        fpr, tpr, thresholds = compute_roc_curve([0.5, 0.6], [1, 1])
        assert len(fpr) == 2
