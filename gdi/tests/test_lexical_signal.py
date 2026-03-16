"""Tests for lexical drift signal."""

import pytest

from src.signals.lexical import LexicalDriftSignal


class TestLexicalDriftSignal:
    """Tests for LexicalDriftSignal."""

    def setup_method(self):
        self.signal = LexicalDriftSignal()

    def test_same_outputs_near_zero(self, reference_outputs):
        """Same outputs should produce near-zero drift."""
        result = self.signal.compute(reference_outputs, reference_outputs)
        assert result.normalized_score < 0.05
        assert result.signal_name == "lexical"

    def test_synonym_substitution_lower_than_collapsed(self, reference_outputs, drifted_outputs, collapsed_outputs):
        """Drifted outputs should produce less drift than collapsed."""
        result_drift = self.signal.compute(drifted_outputs, reference_outputs)
        result_collapse = self.signal.compute(collapsed_outputs, reference_outputs)
        assert result_drift.normalized_score <= result_collapse.normalized_score

    def test_different_high_drift(self, reference_outputs, collapsed_outputs):
        """Very different outputs should produce high drift."""
        result = self.signal.compute(collapsed_outputs, reference_outputs)
        assert result.normalized_score > 0.3

    def test_js_divergence_identical(self):
        """JS divergence of identical texts should be ~0."""
        texts = ["hello world foo bar baz"]
        js = self.signal.js_divergence(texts, texts)
        assert js < 0.01

    def test_js_divergence_different(self):
        """JS divergence of very different texts should be high."""
        a = ["alpha beta gamma delta"]
        b = ["quantum electron proton neutron"]
        js = self.signal.js_divergence(a, b)
        assert js > 0.5

    def test_js_divergence_empty(self):
        """JS divergence with empty inputs."""
        assert self.signal.js_divergence([], []) == 0.0
        assert self.signal.js_divergence(["text"], []) == 1.0
        assert self.signal.js_divergence([], ["text"]) == 1.0

    def test_vocabulary_shift_identical(self):
        """Same vocabulary should have zero shift."""
        texts = ["hello world test"]
        shift = self.signal.vocabulary_shift(texts, texts)
        assert shift < 0.01

    def test_vocabulary_shift_different(self):
        """Different vocabulary should have high shift."""
        a = ["alpha beta gamma"]
        b = ["quantum electron proton"]
        shift = self.signal.vocabulary_shift(a, b)
        assert shift > 0.8

    def test_vocabulary_shift_empty(self):
        """Empty inputs should be handled."""
        assert self.signal.vocabulary_shift([], []) == 0.0
        assert self.signal.vocabulary_shift(["text"], []) == 1.0

    def test_ngram_novelty_identical(self):
        """Same text should have zero novelty."""
        texts = ["hello world this is a test"]
        novelty = self.signal.ngram_novelty(texts, texts)
        assert novelty < 0.01

    def test_ngram_novelty_different(self):
        """Different text should have high novelty."""
        a = ["alpha beta gamma delta epsilon"]
        b = ["one two three four five"]
        novelty = self.signal.ngram_novelty(a, b)
        assert novelty > 0.8

    def test_ngram_novelty_empty_ref(self):
        """Empty reference should return 1.0."""
        assert self.signal.ngram_novelty(["text"], []) == 1.0

    def test_ngram_novelty_empty_current(self):
        """Empty current should return 0.0."""
        assert self.signal.ngram_novelty([], ["text text"]) == 0.0

    def test_normalize(self):
        """Normalization should work correctly."""
        assert self.signal.normalize(0.0) == 0.0
        assert self.signal.normalize(0.25) == 0.5
        assert self.signal.normalize(0.5) == 1.0
        assert self.signal.normalize(0.8) == 1.0

    def test_signal_result_components(self, reference_outputs):
        """Result should contain all components."""
        result = self.signal.compute(reference_outputs, reference_outputs)
        assert "js_divergence" in result.components
        assert "vocabulary_shift" in result.components
        assert "ngram_novelty" in result.components

    def test_composite_weighting(self):
        """Composite should be 0.5*JS + 0.25*vocab + 0.25*novelty."""
        a = ["the quick brown fox"]
        b = ["the lazy brown dog"]
        result = self.signal.compute(a, b)

        js = result.components["js_divergence"]
        vocab = result.components["vocabulary_shift"]
        novelty = result.components["ngram_novelty"]

        expected_raw = 0.5 * js + 0.25 * vocab + 0.25 * novelty
        assert abs(result.raw_score - expected_raw) < 1e-6
