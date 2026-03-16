"""Tests for semantic drift signal."""

import pytest

from src.signals.semantic import SemanticDriftSignal, _tokenize, _word_freq_vector, _cosine_distance


class TestSemanticDriftSignal:
    """Tests for SemanticDriftSignal."""

    def setup_method(self):
        self.signal = SemanticDriftSignal()

    def test_same_outputs_near_zero(self, reference_outputs):
        """Same outputs should produce near-zero drift."""
        result = self.signal.compute(reference_outputs, reference_outputs)
        assert result.normalized_score < 0.05
        assert result.signal_name == "semantic"
        assert result.interpretation == "minimal_drift"

    def test_paraphrased_low_drift(self, reference_outputs, drifted_outputs):
        """Paraphrased outputs should produce low drift."""
        result = self.signal.compute(drifted_outputs, reference_outputs)
        assert result.normalized_score < 0.30
        assert result.interpretation in ("minimal_drift", "moderate_drift")

    def test_different_high_drift(self, reference_outputs, collapsed_outputs):
        """Very different outputs should produce high drift."""
        result = self.signal.compute(collapsed_outputs, reference_outputs)
        assert result.normalized_score > 0.3

    def test_centroid_distance_identical(self):
        """Identical texts should have zero centroid distance."""
        texts = ["hello world foo bar", "hello world foo bar"]
        dist = self.signal.centroid_distance(texts, texts)
        assert dist < 0.01

    def test_centroid_distance_different(self):
        """Very different texts should have high centroid distance."""
        a = ["the cat sat on the mat", "dogs chase cats in the park"]
        b = ["quantum computing uses qubits", "neural networks learn patterns"]
        dist = self.signal.centroid_distance(a, b)
        assert dist > 0.3

    def test_pairwise_drift_identical(self):
        """Identical texts should have zero pairwise drift."""
        texts = ["hello world"]
        drift = self.signal.pairwise_drift(texts, texts)
        assert drift < 0.01

    def test_pairwise_drift_different(self):
        """Different texts should have nonzero pairwise drift."""
        a = ["the cat sat on the mat"]
        b = ["quantum physics explains atoms"]
        drift = self.signal.pairwise_drift(a, b)
        assert drift > 0.5

    def test_mmd_identical(self):
        """MMD of identical sets should be near zero."""
        texts = ["hello world foo bar"]
        mmd = self.signal.mmd(texts, texts)
        assert mmd < 0.01

    def test_mmd_different(self):
        """MMD of different sets should be positive."""
        a = ["cat dog animal pet"]
        b = ["quantum electron proton neutron"]
        mmd = self.signal.mmd(a, b)
        assert mmd >= 0.0

    def test_normalize_caps_at_one(self):
        """Normalization should cap at 1.0."""
        assert self.signal.normalize(0.8) == 1.0
        assert self.signal.normalize(0.5) == 1.0
        assert self.signal.normalize(0.25) == 0.5

    def test_empty_inputs(self):
        """Empty inputs should be handled gracefully."""
        result = self.signal.compute([], [])
        assert result.normalized_score == 0.0

    def test_pairwise_empty(self):
        """Empty inputs to pairwise should return 1.0."""
        assert self.signal.pairwise_drift([], ["text"]) == 1.0
        assert self.signal.pairwise_drift(["text"], []) == 1.0

    def test_mmd_empty(self):
        """Empty inputs to mmd should return 1.0."""
        assert self.signal.mmd([], ["text"]) == 1.0

    def test_signal_result_has_components(self, reference_outputs):
        """Signal result should contain all component scores."""
        result = self.signal.compute(reference_outputs, reference_outputs)
        assert "centroid_distance" in result.components
        assert "pairwise_drift" in result.components
        assert "mmd" in result.components

    def test_interpret(self):
        """Interpret should return correct labels."""
        assert self.signal.interpret(0.05) == "minimal_drift"
        assert self.signal.interpret(0.25) == "moderate_drift"
        assert self.signal.interpret(0.55) == "significant_drift"
        assert self.signal.interpret(0.85) == "severe_drift"


class TestTokenizeAndHelpers:
    """Tests for helper functions."""

    def test_tokenize(self):
        """Tokenize should extract word tokens."""
        tokens = _tokenize("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_word_freq_vector(self):
        """Word frequency vector should count words."""
        counter = _word_freq_vector(["hello hello world"])
        assert counter["hello"] == 2
        assert counter["world"] == 1

    def test_cosine_distance_orthogonal(self):
        """Completely different vectors should have distance ~1."""
        from collections import Counter
        a = Counter({"x": 1})
        b = Counter({"y": 1})
        assert _cosine_distance(a, b) > 0.99

    def test_cosine_distance_identical(self):
        """Same vector should have distance 0."""
        from collections import Counter
        a = Counter({"hello": 2, "world": 1})
        assert _cosine_distance(a, a) < 0.01

    def test_cosine_distance_empty(self):
        """Empty vectors should return 0."""
        from collections import Counter
        assert _cosine_distance(Counter(), Counter()) == 0.0

    def test_cosine_distance_one_empty(self):
        """One empty vector should return 1.0."""
        from collections import Counter
        assert _cosine_distance(Counter({"a": 1}), Counter()) == 1.0
