"""Tests for distributional drift signal."""

import pytest

from src.signals.distributional import DistributionalDriftSignal


class TestDistributionalDriftSignal:
    """Tests for DistributionalDriftSignal."""

    def setup_method(self):
        self.signal = DistributionalDriftSignal()

    def test_same_outputs_near_zero(self, reference_outputs):
        """Same outputs should produce near-zero drift."""
        result = self.signal.compute(reference_outputs, reference_outputs)
        assert result.normalized_score < 0.05
        assert result.signal_name == "distributional"

    def test_slight_shift_lower_than_collapsed(self, reference_outputs, drifted_outputs, collapsed_outputs):
        """Drifted outputs should produce less drift than collapsed."""
        result_drift = self.signal.compute(drifted_outputs, reference_outputs)
        result_collapse = self.signal.compute(collapsed_outputs, reference_outputs)
        assert result_drift.normalized_score <= result_collapse.normalized_score

    def test_different_high_drift(self, reference_outputs, collapsed_outputs):
        """Very different outputs should produce high drift."""
        result = self.signal.compute(collapsed_outputs, reference_outputs)
        assert result.normalized_score > 0.3

    def test_smoothing_prevents_inf(self):
        """Laplace smoothing should prevent infinite KL divergence."""
        a = ["alpha beta gamma"]
        b = ["delta epsilon zeta"]
        kl = self.signal.kl_divergence(a, b)
        assert kl < float('inf')
        assert kl >= 0

    def test_kl_divergence_identical(self):
        """KL divergence of identical distributions should be ~0."""
        texts = ["hello world foo bar baz"]
        kl = self.signal.kl_divergence(texts, texts)
        assert kl < 0.01

    def test_kl_divergence_different(self):
        """KL divergence of different distributions should be > 0."""
        a = ["cat dog bird fish"]
        b = ["quantum electron proton atom"]
        kl = self.signal.kl_divergence(a, b)
        assert kl > 0

    def test_kl_reverse(self):
        """Reverse KL should differ from forward KL."""
        a = ["cat dog bird fish mammal"]
        b = ["quantum electron proton"]
        kl_fwd = self.signal.kl_divergence(a, b, reverse=False)
        kl_rev = self.signal.kl_divergence(a, b, reverse=True)
        # Both should be non-negative
        assert kl_fwd >= 0
        assert kl_rev >= 0

    def test_kl_divergence_empty(self):
        """Empty inputs should be handled."""
        assert self.signal.kl_divergence([], []) == 0.0
        assert self.signal.kl_divergence(["text"], []) == 1.0

    def test_total_variation_identical(self):
        """TV of identical distributions should be ~0."""
        texts = ["hello world foo bar"]
        tv = self.signal.total_variation(texts, texts)
        assert tv < 0.01

    def test_total_variation_different(self):
        """TV of different distributions should be > 0."""
        a = ["cat dog bird"]
        b = ["quantum electron proton"]
        tv = self.signal.total_variation(a, b)
        assert tv > 0.3

    def test_total_variation_empty(self):
        """Empty inputs should be handled."""
        assert self.signal.total_variation([], []) == 0.0
        assert self.signal.total_variation(["text"], []) == 1.0

    def test_js_divergence_identical(self):
        """JS divergence of identical should be ~0."""
        texts = ["hello world foo bar"]
        js = self.signal.js_divergence(texts, texts)
        assert js < 0.01

    def test_js_divergence_different(self):
        """JS divergence of different should be > 0."""
        a = ["cat dog bird"]
        b = ["quantum electron proton"]
        js = self.signal.js_divergence(a, b)
        assert js > 0.5

    def test_js_divergence_empty(self):
        """Empty inputs should be handled."""
        assert self.signal.js_divergence([], []) == 0.0
        assert self.signal.js_divergence(["text"], []) == 1.0

    def test_normalize(self):
        """Normalization should cap at 1.0."""
        assert self.signal.normalize(0.0) == 0.0
        assert self.signal.normalize(0.25) == 0.5
        assert self.signal.normalize(0.5) == 1.0
        assert self.signal.normalize(0.8) == 1.0

    def test_custom_smoothing(self):
        """Custom smoothing parameter should work."""
        signal = DistributionalDriftSignal(smoothing=0.01)
        a = ["hello world"]
        b = ["goodbye world"]
        kl = signal.kl_divergence(a, b)
        assert kl >= 0 and kl < float('inf')

    def test_result_components(self, reference_outputs):
        """Result should contain all components."""
        result = self.signal.compute(reference_outputs, reference_outputs)
        assert "kl_divergence_forward" in result.components
        assert "kl_divergence_reverse" in result.components
        assert "total_variation" in result.components
        assert "js_divergence" in result.components
