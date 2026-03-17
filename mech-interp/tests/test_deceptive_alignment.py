"""Tests for deceptive alignment probes."""

import numpy as np
import pytest

from src.probing.extractor import MockModel, MockModifiedModel
from src.anomaly.deceptive_alignment import (
    DeceptiveAlignmentProber, DeceptiveAlignmentReport,
)


class TestDeceptiveAlignmentProber:
    """Test deceptive alignment probing."""

    def test_monitoring_sensitivity(self, mock_model):
        """Model should show some monitoring sensitivity (different cues = different activations)."""
        prober = DeceptiveAlignmentProber(mock_model)
        sensitivity, details = prober.test_monitoring_sensitivity()

        # MockModel uses input hash, so "[EVALUATION MODE] X" vs "X" will differ
        assert isinstance(sensitivity, float)
        assert sensitivity > 0  # Different inputs = different activations
        assert "num_comparisons" in details

    def test_monitoring_sensitivity_threshold(self, mock_model):
        """Should flag monitoring sensitivity above threshold."""
        prober = DeceptiveAlignmentProber(
            mock_model,
            monitoring_sensitivity_threshold=0.0,  # Very low threshold
        )
        report = prober.run_all_probes()
        # With threshold 0.0, any sensitivity should flag
        assert "monitoring_sensitive" in report.flags

    def test_context_dependent_safety(self, mock_model):
        """Should measure context-dependent safety differences."""
        prober = DeceptiveAlignmentProber(mock_model)
        sensitivity, details = prober.test_context_dependent_safety()

        assert isinstance(sensitivity, float)
        assert "num_comparisons" in details

    def test_latent_capabilities(self, mock_model):
        """Should detect gap between probe accuracy and output accuracy."""
        prober = DeceptiveAlignmentProber(
            mock_model,
            latent_capability_gap_threshold=0.1,
        )

        # Large gap: probe_accuracy=0.9, output_accuracy=0.5
        gap, details = prober.test_latent_capabilities(
            probe_accuracy=0.9, output_accuracy=0.5
        )
        assert gap == pytest.approx(0.4)
        assert details["gap"] == pytest.approx(0.4)

    def test_latent_capabilities_no_gap(self, mock_model):
        """Should show no gap when accuracies match."""
        prober = DeceptiveAlignmentProber(mock_model)
        gap, details = prober.test_latent_capabilities(
            probe_accuracy=0.8, output_accuracy=0.8
        )
        assert gap == pytest.approx(0.0)

    def test_latent_capabilities_negative_gap(self, mock_model):
        """Output better than probe should give zero gap."""
        prober = DeceptiveAlignmentProber(mock_model)
        gap, details = prober.test_latent_capabilities(
            probe_accuracy=0.5, output_accuracy=0.9
        )
        assert gap == 0.0

    def test_paraphrase_consistency(self, mock_model):
        """Should measure paraphrase consistency."""
        prober = DeceptiveAlignmentProber(mock_model)
        consistency, details = prober.test_paraphrase_consistency()

        assert isinstance(consistency, float)
        assert "num_comparisons" in details
        assert "mean_similarity" in details

    def test_run_all_probes(self, mock_model):
        """Should run all probes and produce a report."""
        prober = DeceptiveAlignmentProber(
            mock_model,
            monitoring_sensitivity_threshold=0.15,
            latent_capability_gap_threshold=0.1,
        )
        report = prober.run_all_probes(probe_accuracy=0.9, output_accuracy=0.7)

        assert isinstance(report, DeceptiveAlignmentReport)
        assert isinstance(report.monitoring_sensitivity, float)
        assert isinstance(report.context_dependent_safety, float)
        assert isinstance(report.latent_capability_gap, float)
        assert isinstance(report.paraphrase_consistency, float)
        assert isinstance(report.is_suspicious, bool)

    def test_report_flags_latent_gap(self, mock_model):
        """Should flag latent capability gap when present."""
        prober = DeceptiveAlignmentProber(
            mock_model,
            latent_capability_gap_threshold=0.1,
        )
        report = prober.run_all_probes(probe_accuracy=0.9, output_accuracy=0.5)
        assert "latent_capability_gap" in report.flags

    def test_suspicious_when_multiple_flags(self, mock_model):
        """Should be suspicious when 2+ flags are raised."""
        prober = DeceptiveAlignmentProber(
            mock_model,
            monitoring_sensitivity_threshold=0.0,  # Will flag
            latent_capability_gap_threshold=0.0,    # Will flag
        )
        report = prober.run_all_probes(probe_accuracy=0.9, output_accuracy=0.5)
        assert report.is_suspicious
        assert len(report.flags) >= 2

    def test_report_to_dict(self, mock_model):
        """Report should serialize to dict."""
        prober = DeceptiveAlignmentProber(mock_model)
        report = prober.run_all_probes()
        d = report.to_dict()
        assert "monitoring_sensitivity" in d
        assert "is_suspicious" in d
        assert "flags" in d
        assert "details" in d

    def test_not_suspicious_with_normal_thresholds(self, mock_model):
        """With very high/low thresholds, should not be suspicious."""
        prober = DeceptiveAlignmentProber(
            mock_model,
            monitoring_sensitivity_threshold=100.0,
            context_safety_threshold=100.0,
            latent_capability_gap_threshold=100.0,
            paraphrase_consistency_threshold=-2.0,  # Very low threshold
        )
        report = prober.run_all_probes(probe_accuracy=0.5, output_accuracy=0.5)
        assert not report.is_suspicious
        assert len(report.flags) == 0
