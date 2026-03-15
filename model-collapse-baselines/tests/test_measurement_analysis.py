"""Tests for measurement and analysis modules with coverage gaps.

Covers:
- src/measurement/tail_analysis.py (TailAnalyzer)
- src/measurement/variance.py (VarianceTracker)
- src/measurement/kl_divergence.py (estimate_kl with mocked model)
- src/measurement/fixed_point.py (FixedPointDetector convergence edge cases)
- src/analysis/report.py (generate_report)
- src/analysis/phase_diagrams.py (plot_phase_diagram, plot_collapse_boundary)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# ======================================================================
# TAIL ANALYSIS
# ======================================================================

from src.measurement.tail_analysis import TailAnalyzer, TailResult


class _FakeTokenizer:
    """Minimal tokenizer stub for TailAnalyzer."""
    pass


class TestTailAnalyzerUniform:
    """Uniform distribution: Gini ~ 0, tail mass matches expectation."""

    def test_uniform_gini_near_zero(self):
        n = 1000
        p = np.ones(n, dtype=np.float64) / n
        tokenizer = _FakeTokenizer()
        analyzer = TailAnalyzer(p, tokenizer)
        result = analyzer.measure(p)
        assert abs(result.gini_coefficient) < 0.01, (
            f"Gini should be ~0 for uniform, got {result.gini_coefficient}"
        )

    def test_uniform_tail_mass_ordering(self):
        n = 1000
        p = np.ones(n, dtype=np.float64) / n
        tokenizer = _FakeTokenizer()
        analyzer = TailAnalyzer(p, tokenizer)
        result = analyzer.measure(p)
        assert result.tail_mass_p01 <= result.tail_mass_p05 <= result.tail_mass_p10, (
            f"Tail masses should be ordered: {result.tail_mass_p01} <= "
            f"{result.tail_mass_p05} <= {result.tail_mass_p10}"
        )

    def test_uniform_tail_mass_approximate_values(self):
        """For uniform dist, tail mass at fraction f should be ~f."""
        n = 1000
        p = np.ones(n, dtype=np.float64) / n
        tokenizer = _FakeTokenizer()
        analyzer = TailAnalyzer(p, tokenizer)
        result = analyzer.measure(p)
        assert abs(result.tail_mass_p01 - 0.01) < 0.005
        assert abs(result.tail_mass_p05 - 0.05) < 0.005
        assert abs(result.tail_mass_p10 - 0.10) < 0.005

    def test_uniform_top_10_mass(self):
        """For uniform dist over 1000 tokens, top-10 mass = 10/1000 = 0.01."""
        n = 1000
        p = np.ones(n, dtype=np.float64) / n
        tokenizer = _FakeTokenizer()
        analyzer = TailAnalyzer(p, tokenizer)
        result = analyzer.measure(p)
        expected = 10.0 / n
        assert abs(result.top_10_mass - expected) < 1e-9

    def test_uniform_rank_correlation_with_self(self):
        n = 1000
        p = np.ones(n, dtype=np.float64) / n
        tokenizer = _FakeTokenizer()
        analyzer = TailAnalyzer(p, tokenizer)
        # Uniform vs uniform: all ranks are tied, but spearmanr handles it
        result = analyzer.measure(p)
        # With uniform dist, rank correlation is undefined/nan; just check it runs
        assert np.isfinite(result.rank_correlation) or np.isnan(result.rank_correlation)


class TestTailAnalyzerOneHot:
    """One-hot distribution: Gini ~ 1."""

    def test_one_hot_gini_near_one(self):
        n = 1000
        p = np.zeros(n, dtype=np.float64)
        p[0] = 1.0
        tokenizer = _FakeTokenizer()
        # Use uniform reference to avoid issues
        ref = np.ones(n, dtype=np.float64) / n
        analyzer = TailAnalyzer(ref, tokenizer)
        result = analyzer.measure(p)
        assert result.gini_coefficient > 0.95, (
            f"Gini should be ~1 for one-hot, got {result.gini_coefficient}"
        )

    def test_one_hot_top_10_mass_is_one(self):
        n = 1000
        p = np.zeros(n, dtype=np.float64)
        p[0] = 1.0
        ref = np.ones(n, dtype=np.float64) / n
        tokenizer = _FakeTokenizer()
        analyzer = TailAnalyzer(ref, tokenizer)
        result = analyzer.measure(p)
        assert abs(result.top_10_mass - 1.0) < 1e-9


class TestTailAnalyzerRankCorrelation:
    """Reference = model distribution: rank_correlation = 1.0."""

    def test_same_distribution_rank_correlation_one(self):
        n = 500
        rng = np.random.RandomState(42)
        p = rng.dirichlet(np.ones(n))
        tokenizer = _FakeTokenizer()
        analyzer = TailAnalyzer(p, tokenizer)
        result = analyzer.measure(p)
        assert abs(result.rank_correlation - 1.0) < 1e-6, (
            f"Rank correlation should be 1.0 for same dist, got {result.rank_correlation}"
        )

    def test_different_distribution_lower_correlation(self):
        """A shuffled distribution should have lower rank correlation than 1.0."""
        n = 500
        rng = np.random.RandomState(42)
        p = rng.dirichlet(np.ones(n))
        # Shuffle to destroy rank ordering
        q = p.copy()
        rng.shuffle(q)
        tokenizer = _FakeTokenizer()
        analyzer = TailAnalyzer(p, tokenizer)
        result = analyzer.measure(q)
        assert result.rank_correlation < 0.95, (
            f"Shuffled dist should have lower rank corr, got {result.rank_correlation}"
        )


class TestTailAnalyzerEdgeCases:
    """Edge cases: padding, truncation, zero distribution."""

    def test_shorter_model_distribution_padded(self):
        n = 1000
        ref = np.ones(n, dtype=np.float64) / n
        tokenizer = _FakeTokenizer()
        analyzer = TailAnalyzer(ref, tokenizer)
        # Model dist shorter than reference
        short_q = np.ones(500, dtype=np.float64) / 500
        result = analyzer.measure(short_q)
        assert isinstance(result, TailResult)
        assert np.isfinite(result.gini_coefficient)

    def test_longer_model_distribution_truncated(self):
        n = 500
        ref = np.ones(n, dtype=np.float64) / n
        tokenizer = _FakeTokenizer()
        analyzer = TailAnalyzer(ref, tokenizer)
        long_q = np.ones(1000, dtype=np.float64) / 1000
        result = analyzer.measure(long_q)
        assert isinstance(result, TailResult)
        assert np.isfinite(result.gini_coefficient)

    def test_zero_model_distribution_fallback(self):
        n = 100
        ref = np.ones(n, dtype=np.float64) / n
        tokenizer = _FakeTokenizer()
        analyzer = TailAnalyzer(ref, tokenizer)
        zero_q = np.zeros(n, dtype=np.float64)
        result = analyzer.measure(zero_q)
        # Should fall back to uniform
        assert isinstance(result, TailResult)
        assert abs(result.gini_coefficient) < 0.02

    def test_small_vocab(self):
        """Vocab smaller than 10 should still work for top_10_mass."""
        n = 5
        ref = np.ones(n, dtype=np.float64) / n
        tokenizer = _FakeTokenizer()
        analyzer = TailAnalyzer(ref, tokenizer)
        result = analyzer.measure(ref)
        # top_10_mass should be 1.0 since vocab < 10
        assert abs(result.top_10_mass - 1.0) < 1e-9


# ======================================================================
# VARIANCE TRACKER
# ======================================================================

from src.measurement.variance import VarianceTracker, VarianceResult


class MockEncoder:
    """Returns deterministic embeddings based on text hash.

    Each text is mapped to a fixed 32-dimensional embedding vector
    derived from a hash of the text.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, texts, show_progress_bar=False):
        embeddings = []
        for text in texts:
            h = hashlib.md5(text.encode()).hexdigest()
            seed = int(h[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim)
            # Normalize to unit length
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float64)


class MockEncoderIdentical:
    """Always returns the exact same embedding regardless of input."""

    def __init__(self, dim: int = 32):
        self.dim = dim
        self._vec = np.ones(dim, dtype=np.float64)
        self._vec = self._vec / np.linalg.norm(self._vec)

    def encode(self, texts, show_progress_bar=False):
        return np.tile(self._vec, (len(texts), 1))


class TestVarianceTrackerDiverseTexts:
    """Diverse texts should have high variance."""

    def _make_diverse_texts(self, n: int = 50) -> list[str]:
        """Generate n unique diverse texts."""
        return [f"sentence number {i} with unique words like {chr(65 + i % 26)}" for i in range(n)]

    def _make_reference_embeddings(self, encoder, n: int = 50) -> np.ndarray:
        texts = [f"reference doc {i} about topic {chr(65 + i % 26)}" for i in range(n)]
        return encoder.encode(texts)

    def test_diverse_texts_positive_variance(self):
        encoder = MockEncoder(dim=32)
        ref_emb = self._make_reference_embeddings(encoder)
        tracker = VarianceTracker(encoder, ref_emb)
        texts = self._make_diverse_texts(50)
        result = tracker.measure(texts)
        assert result.embedding_variance > 0.0, (
            f"Diverse texts should have positive variance, got {result.embedding_variance}"
        )

    def test_diverse_texts_mean_cosine_less_than_one(self):
        encoder = MockEncoder(dim=32)
        ref_emb = self._make_reference_embeddings(encoder)
        tracker = VarianceTracker(encoder, ref_emb)
        texts = self._make_diverse_texts(50)
        result = tracker.measure(texts)
        assert result.mean_pairwise_cosine < 0.99, (
            f"Diverse texts should have cosine < 1, got {result.mean_pairwise_cosine}"
        )

    def test_diverse_texts_effective_dimension_greater_than_one(self):
        encoder = MockEncoder(dim=32)
        ref_emb = self._make_reference_embeddings(encoder)
        tracker = VarianceTracker(encoder, ref_emb)
        texts = self._make_diverse_texts(50)
        result = tracker.measure(texts)
        assert result.effective_dimension > 1, (
            f"Diverse texts should need >1 PCA dim, got {result.effective_dimension}"
        )


class TestVarianceTrackerIdenticalTexts:
    """Identical texts: low variance, cosine ~ 1."""

    def test_identical_texts_zero_variance(self):
        encoder = MockEncoderIdentical(dim=32)
        ref_emb = encoder.encode(["ref text"] * 20)
        tracker = VarianceTracker(encoder, ref_emb)
        texts = ["identical text"] * 20
        result = tracker.measure(texts)
        assert abs(result.embedding_variance) < 1e-6, (
            f"Identical texts should have ~0 variance, got {result.embedding_variance}"
        )

    def test_identical_texts_cosine_one(self):
        encoder = MockEncoderIdentical(dim=32)
        ref_emb = encoder.encode(["ref text"] * 20)
        tracker = VarianceTracker(encoder, ref_emb)
        texts = ["identical text"] * 20
        result = tracker.measure(texts)
        assert abs(result.mean_pairwise_cosine - 1.0) < 1e-6, (
            f"Identical texts should have cosine=1, got {result.mean_pairwise_cosine}"
        )

    def test_identical_texts_effective_dimension_low(self):
        encoder = MockEncoderIdentical(dim=32)
        ref_emb = encoder.encode(["ref text"] * 20)
        tracker = VarianceTracker(encoder, ref_emb)
        texts = ["identical text"] * 20
        result = tracker.measure(texts)
        # With zero variance PCA will need 0 or 1 dims
        assert result.effective_dimension <= 1, (
            f"Identical texts should need <=1 PCA dim, got {result.effective_dimension}"
        )


class TestVarianceTrackerDrift:
    """Drift from reference."""

    def test_drift_zero_when_same_reference(self):
        """When model texts equal reference texts, drift should be ~0."""
        encoder = MockEncoder(dim=32)
        ref_texts = [f"ref text {i}" for i in range(30)]
        ref_emb = encoder.encode(ref_texts)
        tracker = VarianceTracker(encoder, ref_emb)
        # Measure with the exact same texts
        result = tracker.measure(ref_texts)
        assert abs(result.drift_from_reference) < 1e-6, (
            f"Drift from reference should be ~0 for same texts, got {result.drift_from_reference}"
        )

    def test_drift_positive_for_different_texts(self):
        encoder = MockEncoder(dim=32)
        ref_texts = [f"reference document {i}" for i in range(30)]
        ref_emb = encoder.encode(ref_texts)
        tracker = VarianceTracker(encoder, ref_emb)
        different_texts = [f"completely different text {i * 1000}" for i in range(30)]
        result = tracker.measure(different_texts)
        assert result.drift_from_reference > 0.0, (
            f"Drift should be positive for different texts, got {result.drift_from_reference}"
        )


class TestVarianceTrackerClusterCount:
    """Cluster count should return a reasonable value."""

    def test_cluster_count_is_at_least_one(self):
        encoder = MockEncoder(dim=32)
        ref_emb = encoder.encode([f"ref {i}" for i in range(30)])
        tracker = VarianceTracker(encoder, ref_emb)
        texts = [f"text {i}" for i in range(30)]
        result = tracker.measure(texts)
        assert result.cluster_count >= 1

    def test_too_few_samples_returns_one_cluster(self):
        encoder = MockEncoder(dim=32)
        ref_emb = encoder.encode([f"ref {i}" for i in range(5)])
        tracker = VarianceTracker(encoder, ref_emb)
        # Less than min_samples=10, so cluster_count should be 1
        texts = [f"text {i}" for i in range(5)]
        result = tracker.measure(texts)
        assert result.cluster_count == 1


class TestVarianceTrackerEmptyTexts:
    """Empty texts should return safe defaults."""

    def test_empty_texts(self):
        encoder = MockEncoder(dim=32)
        ref_emb = encoder.encode([f"ref {i}" for i in range(10)])
        tracker = VarianceTracker(encoder, ref_emb)
        result = tracker.measure([])
        assert result.embedding_variance == 0.0
        assert result.mean_pairwise_cosine == 1.0
        assert result.drift_from_reference == 0.0
        assert result.effective_dimension == 0
        assert result.cluster_count == 0


class TestVarianceTrackerSingleText:
    """Single text edge case."""

    def test_single_text(self):
        encoder = MockEncoder(dim=32)
        ref_emb = encoder.encode([f"ref {i}" for i in range(10)])
        tracker = VarianceTracker(encoder, ref_emb)
        result = tracker.measure(["single text"])
        assert result.embedding_variance == 0.0
        assert result.mean_pairwise_cosine == 1.0
        assert result.effective_dimension == 0


# ======================================================================
# KL DIVERGENCE — estimate_kl with mocked model (lines 80-130)
# ======================================================================

from src.measurement.kl_divergence import KLDivergenceEstimator


class _FakeTokenizerOutput:
    """Mimics a HuggingFace BatchEncoding."""

    def __init__(self, input_ids, attention_mask):
        self._data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def to(self, device):
        return self

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)


class _FakeModelOutput:
    """Mimics a HuggingFace model output with .logits."""

    def __init__(self, logits):
        self.logits = logits


def _make_kl_mock_tokenizer(vocab_size: int = 100):
    """Mock tokenizer that supports both encode() and __call__()."""
    import torch

    class FakeKLTokenizer:
        def encode(self, text, add_special_tokens=False, **kwargs):
            words = text.split()
            return [hash(w) % vocab_size for w in words]

        def __call__(self, texts, **kwargs):
            max_len = kwargs.get("max_length", 32)
            all_ids = []
            all_masks = []
            for text in texts:
                words = text.split()
                ids = [hash(w) % vocab_size for w in words[:max_len]]
                if not ids:
                    ids = [0]
                all_ids.append(ids)
                all_masks.append([1] * len(ids))

            max_l = max(len(ids) for ids in all_ids)
            for i in range(len(all_ids)):
                pad_len = max_l - len(all_ids[i])
                all_ids[i] += [0] * pad_len
                all_masks[i] += [0] * pad_len

            return _FakeTokenizerOutput(
                input_ids=torch.tensor(all_ids, dtype=torch.long),
                attention_mask=torch.tensor(all_masks, dtype=torch.long),
            )

        def __len__(self):
            return vocab_size

    return FakeKLTokenizer()


def _make_kl_mock_model(vocab_size: int = 100, uniform: bool = True):
    """Mock causal LM returning predictable logits."""
    import torch

    fake_param = torch.zeros(1)

    class FakeModel:
        def eval(self):
            pass

        def parameters(self):
            return iter([fake_param])

        def __call__(self, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None:
                for v in kwargs.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        input_ids = v
                        break

            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]

            if uniform:
                logits = torch.zeros(batch_size, seq_len, vocab_size)
            else:
                logits = torch.full((batch_size, seq_len, vocab_size), -10.0)
                logits[:, :, 0] = 10.0

            return _FakeModelOutput(logits)

    return FakeModel()


class TestEstimateKLWithModel:
    """Test KLDivergenceEstimator.estimate_kl with mocked model."""

    def test_estimate_kl_returns_valid_dict(self):
        import torch

        vocab_size = 100
        p = np.ones(vocab_size, dtype=np.float64) / vocab_size
        tokenizer = _make_kl_mock_tokenizer(vocab_size)
        model = _make_kl_mock_model(vocab_size, uniform=True)
        estimator = KLDivergenceEstimator(p, tokenizer)

        texts = ["the cat sat on the mat", "a dog ran fast today"]
        result = estimator.estimate_kl(model, texts, batch_size=2)

        assert hasattr(result, "kl_p_q")
        assert hasattr(result, "kl_q_p")
        assert hasattr(result, "js_divergence")
        assert hasattr(result, "total_variation")

    def test_estimate_kl_values_finite(self):
        import torch

        vocab_size = 100
        p = np.ones(vocab_size, dtype=np.float64) / vocab_size
        tokenizer = _make_kl_mock_tokenizer(vocab_size)
        model = _make_kl_mock_model(vocab_size, uniform=True)
        estimator = KLDivergenceEstimator(p, tokenizer)

        texts = ["hello world foo bar", "baz qux quux corge"]
        result = estimator.estimate_kl(model, texts, batch_size=2)

        assert np.isfinite(result.kl_p_q), f"kl_p_q not finite: {result.kl_p_q}"
        assert np.isfinite(result.kl_q_p), f"kl_q_p not finite: {result.kl_q_p}"
        assert np.isfinite(result.js_divergence), f"js not finite: {result.js_divergence}"
        assert np.isfinite(result.total_variation), f"tv not finite: {result.total_variation}"

    def test_estimate_kl_values_non_negative(self):
        import torch

        vocab_size = 100
        p = np.ones(vocab_size, dtype=np.float64) / vocab_size
        tokenizer = _make_kl_mock_tokenizer(vocab_size)
        model = _make_kl_mock_model(vocab_size, uniform=True)
        estimator = KLDivergenceEstimator(p, tokenizer)

        texts = ["one two three four five"]
        result = estimator.estimate_kl(model, texts, batch_size=1)

        assert result.kl_p_q >= -1e-10, f"kl_p_q should be non-negative: {result.kl_p_q}"
        assert result.kl_q_p >= -1e-10, f"kl_q_p should be non-negative: {result.kl_q_p}"
        assert result.js_divergence >= -1e-10, f"js should be non-negative: {result.js_divergence}"
        assert result.total_variation >= -1e-10, f"tv should be non-negative: {result.total_variation}"

    def test_estimate_kl_uniform_model_near_zero_kl(self):
        """Uniform model with uniform reference -> KL should be small."""
        import torch

        vocab_size = 100
        p = np.ones(vocab_size, dtype=np.float64) / vocab_size
        tokenizer = _make_kl_mock_tokenizer(vocab_size)
        model = _make_kl_mock_model(vocab_size, uniform=True)
        estimator = KLDivergenceEstimator(p, tokenizer)

        texts = ["the cat sat on the mat"] * 5
        result = estimator.estimate_kl(model, texts, batch_size=2)

        # Uniform model should produce near-uniform Q -> low KL
        assert result.kl_p_q < 1.0, f"KL(P||Q) should be small for uniform model, got {result.kl_p_q}"

    def test_estimate_kl_peaked_model_larger_kl(self):
        """Peaked model with uniform reference -> higher KL."""
        import torch

        vocab_size = 100
        p = np.ones(vocab_size, dtype=np.float64) / vocab_size
        tokenizer = _make_kl_mock_tokenizer(vocab_size)
        model = _make_kl_mock_model(vocab_size, uniform=False)
        estimator = KLDivergenceEstimator(p, tokenizer)

        texts = ["a b c d e f g h i j"]
        result = estimator.estimate_kl(model, texts, batch_size=1)

        # Peaked model should have higher KL than uniform
        assert result.kl_p_q > 0.1, f"Peaked model KL should be substantial, got {result.kl_p_q}"

    def test_estimate_kl_multiple_batches(self):
        """Test with more texts than batch_size to exercise batching."""
        import torch

        vocab_size = 50
        p = np.ones(vocab_size, dtype=np.float64) / vocab_size
        tokenizer = _make_kl_mock_tokenizer(vocab_size)
        model = _make_kl_mock_model(vocab_size, uniform=True)
        estimator = KLDivergenceEstimator(p, tokenizer)

        texts = [f"sentence number {i} with content" for i in range(10)]
        result = estimator.estimate_kl(model, texts, batch_size=3)

        assert np.isfinite(result.kl_p_q)
        assert np.isfinite(result.js_divergence)

    def test_estimate_kl_no_attention_mask(self):
        """Test when tokenizer output has no attention_mask (returns None)."""
        import torch

        vocab_size = 50
        p = np.ones(vocab_size, dtype=np.float64) / vocab_size

        class NoMaskTokenizerOutput:
            def __init__(self, input_ids):
                self._data = {"input_ids": input_ids}

            def to(self, device):
                return self

            def get(self, key, default=None):
                return self._data.get(key, default)

            def __getitem__(self, key):
                return self._data[key]

            def keys(self):
                return self._data.keys()

            def items(self):
                return self._data.items()

            def __contains__(self, key):
                return key in self._data

            def __iter__(self):
                return iter(self._data)

        class NoMaskTokenizer:
            def encode(self, text, **kwargs):
                return [hash(w) % vocab_size for w in text.split()]

            def __call__(self, texts, **kwargs):
                all_ids = []
                for text in texts:
                    ids = [hash(w) % vocab_size for w in text.split()]
                    if not ids:
                        ids = [0]
                    all_ids.append(ids)
                max_l = max(len(ids) for ids in all_ids)
                for i in range(len(all_ids)):
                    all_ids[i] += [0] * (max_l - len(all_ids[i]))
                return NoMaskTokenizerOutput(
                    input_ids=torch.tensor(all_ids, dtype=torch.long),
                )

        tokenizer = NoMaskTokenizer()
        model = _make_kl_mock_model(vocab_size, uniform=True)
        estimator = KLDivergenceEstimator(p, tokenizer)

        texts = ["hello world foo"]
        result = estimator.estimate_kl(model, texts, batch_size=1)
        assert np.isfinite(result.kl_p_q)


# ======================================================================
# FIXED POINT DETECTOR
# ======================================================================

from src.measurement.fixed_point import FixedPointDetector, ConvergenceReport


class TestFixedPointConvergence:
    """Test convergence detection with metrics that plateau."""

    def test_plateau_triggers_convergence(self):
        """Metrics that stabilize should trigger convergence."""
        detector = FixedPointDetector(patience=3, kl_tolerance=0.01, diversity_tolerance=0.01)

        # First 5 generations change; then 4 generations stable
        changing_metrics = [
            {"kl_divergence": 0.1, "diversity": 0.9, "entropy": 5.0},
            {"kl_divergence": 0.3, "diversity": 0.7, "entropy": 4.5},
            {"kl_divergence": 0.5, "diversity": 0.5, "entropy": 4.0},
            {"kl_divergence": 0.6, "diversity": 0.4, "entropy": 3.5},
            {"kl_divergence": 0.7, "diversity": 0.3, "entropy": 3.0},
        ]
        stable_metrics = {"kl_divergence": 0.7, "diversity": 0.3, "entropy": 3.0}

        converged = False
        for gen, m in enumerate(changing_metrics):
            converged = detector.update(gen, m)
        assert not converged, "Should not converge while metrics are changing"

        # Now add stable generations (patience=3 needed)
        for gen in range(5, 9):
            converged = detector.update(gen, stable_metrics)

        assert converged, "Should converge after patience=3 stable generations"

    def test_convergence_generation_recorded(self):
        detector = FixedPointDetector(patience=2, kl_tolerance=0.05, diversity_tolerance=0.05)

        # Immediate plateau
        metrics = {"kl_divergence": 1.0, "diversity": 0.5, "entropy": 3.0}
        for gen in range(5):
            detector.update(gen, metrics)

        report = detector.get_convergence_report()
        assert report.converged is True
        assert report.generation_converged is not None
        assert report.generation_converged >= 2  # need at least 2 data points + patience


class TestFixedPointNoConvergence:
    """Constantly changing metrics should not converge."""

    def test_no_convergence_with_changing_metrics(self):
        detector = FixedPointDetector(patience=3, kl_tolerance=0.01, diversity_tolerance=0.01)

        for gen in range(20):
            metrics = {
                "kl_divergence": gen * 0.1,
                "diversity": 1.0 - gen * 0.05,
                "entropy": 5.0 - gen * 0.2,
            }
            result = detector.update(gen, metrics)

        assert not result, "Should not converge with constantly changing metrics"
        report = detector.get_convergence_report()
        assert report.converged is False
        assert report.generation_converged is None

    def test_alternating_metrics_no_convergence(self):
        detector = FixedPointDetector(patience=2, kl_tolerance=0.01, diversity_tolerance=0.01)

        for gen in range(20):
            if gen % 2 == 0:
                metrics = {"kl_divergence": 0.5, "diversity": 0.5, "entropy": 3.0}
            else:
                metrics = {"kl_divergence": 0.8, "diversity": 0.2, "entropy": 2.5}
            detector.update(gen, metrics)

        report = detector.get_convergence_report()
        assert report.converged is False


class TestFixedPointConvergenceReport:
    """Test get_convergence_report with actual convergence."""

    def test_report_fields_on_convergence(self):
        detector = FixedPointDetector(patience=2, kl_tolerance=0.05, diversity_tolerance=0.05)

        # Gen 0: initial
        detector.update(0, {"kl_divergence": 0.1, "diversity": 0.9, "entropy": 5.0})
        # Gen 1: change
        detector.update(1, {"kl_divergence": 0.5, "diversity": 0.5, "entropy": 3.0})
        # Gen 2: stable (consecutive_stable=1)
        detector.update(2, {"kl_divergence": 0.5, "diversity": 0.5, "entropy": 3.0})
        # Gen 3: stable (consecutive_stable=2 >= patience=2 -> converge)
        detector.update(3, {"kl_divergence": 0.5, "diversity": 0.5, "entropy": 3.0})

        report = detector.get_convergence_report()

        assert report.converged is True
        assert report.generation_converged is not None
        assert report.kl_at_convergence is not None
        assert report.diversity_at_convergence is not None
        assert report.total_entropy_lost == pytest.approx(5.0 - 3.0, abs=1e-6)
        assert len(report.kl_trajectory) == 4
        assert len(report.diversity_trajectory) == 4
        assert len(report.entropy_trajectory) == 4

    def test_report_trajectories_correct(self):
        detector = FixedPointDetector(patience=2, kl_tolerance=0.1, diversity_tolerance=0.1)

        kl_values = [0.1, 0.3, 0.5, 0.5, 0.5]
        div_values = [0.9, 0.7, 0.5, 0.5, 0.5]
        ent_values = [5.0, 4.0, 3.0, 3.0, 3.0]

        for gen in range(5):
            detector.update(gen, {
                "kl_divergence": kl_values[gen],
                "diversity": div_values[gen],
                "entropy": ent_values[gen],
            })

        report = detector.get_convergence_report()
        assert report.kl_trajectory == pytest.approx(kl_values)
        assert report.diversity_trajectory == pytest.approx(div_values)
        assert report.entropy_trajectory == pytest.approx(ent_values)


class TestFixedPointDiversityThreshold:
    """Test that diversity tolerance is checked independently."""

    def test_kl_stable_diversity_unstable(self):
        """If KL is stable but diversity isn't, should not converge."""
        detector = FixedPointDetector(patience=2, kl_tolerance=0.01, diversity_tolerance=0.01)

        for gen in range(10):
            metrics = {
                "kl_divergence": 0.5,  # stable
                "diversity": 0.5 + gen * 0.05,  # changing
                "entropy": 3.0,
            }
            detector.update(gen, metrics)

        report = detector.get_convergence_report()
        assert report.converged is False

    def test_diversity_stable_kl_unstable(self):
        """If diversity is stable but KL isn't, should not converge."""
        detector = FixedPointDetector(patience=2, kl_tolerance=0.01, diversity_tolerance=0.01)

        for gen in range(10):
            metrics = {
                "kl_divergence": 0.1 * gen,  # changing
                "diversity": 0.5,  # stable
                "entropy": 3.0,
            }
            detector.update(gen, metrics)

        report = detector.get_convergence_report()
        assert report.converged is False


class TestFixedPointEdgeCases:
    """Edge cases: single generation, empty metrics, already converged."""

    def test_single_generation(self):
        detector = FixedPointDetector(patience=2)
        result = detector.update(0, {"kl_divergence": 0.5, "diversity": 0.5})
        assert not result
        report = detector.get_convergence_report()
        assert report.converged is False
        assert report.total_entropy_lost == 0.0

    def test_missing_keys_default_to_zero(self):
        detector = FixedPointDetector(patience=2, kl_tolerance=0.01, diversity_tolerance=0.01)
        # Empty metrics dict -> defaults to 0 for all
        for gen in range(5):
            detector.update(gen, {})

        report = detector.get_convergence_report()
        # All zeros stable -> should converge
        assert report.converged is True

    def test_already_converged_returns_true(self):
        detector = FixedPointDetector(patience=1, kl_tolerance=0.1, diversity_tolerance=0.1)

        metrics = {"kl_divergence": 0.5, "diversity": 0.5, "entropy": 3.0}
        detector.update(0, metrics)
        detector.update(1, metrics)
        assert detector.update(1, metrics) is True  # already converged

        # Subsequent calls should still return True
        assert detector.update(2, metrics) is True
        assert detector.update(3, {"kl_divergence": 999.0, "diversity": 0.0}) is True

    def test_reset_clears_state(self):
        detector = FixedPointDetector(patience=1)
        metrics = {"kl_divergence": 0.5, "diversity": 0.5}
        detector.update(0, metrics)
        detector.update(1, metrics)

        report = detector.get_convergence_report()
        assert report.converged is True

        detector.reset()
        report = detector.get_convergence_report()
        assert report.converged is False
        assert len(report.kl_trajectory) == 0

    def test_convergence_report_single_entry(self):
        """Single-entry history should report 0 entropy lost."""
        detector = FixedPointDetector()
        detector.update(0, {"kl_divergence": 0.1, "diversity": 0.9, "entropy": 5.0})
        report = detector.get_convergence_report()
        assert report.total_entropy_lost == 0.0

    def test_convergence_report_index_out_of_range_fallback(self):
        """Test the fallback path when convergence_generation > len(history).

        This exercises lines 151-161 where the code falls back to the
        last recorded value if the convergence generation index exceeds
        the history length.
        """
        detector = FixedPointDetector(patience=1, kl_tolerance=0.05, diversity_tolerance=0.05)
        metrics = {"kl_divergence": 0.5, "diversity": 0.5, "entropy": 3.0}
        detector.update(0, metrics)
        detector.update(1, metrics)

        # Force convergence_generation to be beyond history length
        detector._convergence_generation = 999
        detector._converged = True

        report = detector.get_convergence_report()
        # Should fall back to last recorded values
        assert report.kl_at_convergence == 0.5
        assert report.diversity_at_convergence == 0.5


# ======================================================================
# REPORT (src/analysis/report.py)
# ======================================================================

from src.analysis.report import (
    generate_report,
    _load_config,
    _load_all_schedule_metrics,
    _find_scale_data,
    _build_fixed_point_table,
    _build_collapse_rate_table,
    _generate_recommendations,
)


def _make_metrics_records(n_gens: int = 5, start_kl: float = 0.1) -> list[dict]:
    """Create synthetic metrics records for n generations."""
    records = []
    for gen in range(n_gens):
        records.append({
            "generation": gen,
            "kl_divergence": start_kl + gen * 0.15,
            "entropy": 5.0 - gen * 0.3,
            "self_bleu": 0.1 + gen * 0.05,
            "distinct_2": 0.9 - gen * 0.05,
            "embedding_variance": 1.0 - gen * 0.1,
            "vocabulary_usage": 0.95 - gen * 0.05,
        })
    return records


def _create_experiment_dir(
    base_dir: Path,
    schedules: list[str] | None = None,
    with_config: bool = True,
    with_scale_data: bool = False,
) -> Path:
    """Create a mock experiment directory structure."""
    base_dir.mkdir(parents=True, exist_ok=True)

    if schedules is None:
        schedules = ["zero_alpha", "constant_05"]

    if with_config:
        config = {
            "experiment": {"name": "test", "num_generations": 5},
            "base_model": "test-model",
        }
        config_path = base_dir / "config.json"
        config_path.write_text(json.dumps(config))

    for sched in schedules:
        metrics_dir = base_dir / sched / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        records = _make_metrics_records(n_gens=5, start_kl=0.1 if sched == "zero_alpha" else 0.05)
        metrics_file = metrics_dir / "metrics.json"
        metrics_file.write_text(json.dumps(records))

    if with_scale_data:
        for scale_name, start_kl in [("1b", 0.2), ("7b", 0.1)]:
            metrics_dir = base_dir / scale_name / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            records = _make_metrics_records(n_gens=5, start_kl=start_kl)
            metrics_file = metrics_dir / "metrics.json"
            metrics_file.write_text(json.dumps(records))

    return base_dir


class TestReportGeneration:
    """Test generate_report produces valid markdown."""

    def test_generate_report_basic(self, tmp_path):
        exp_dir = _create_experiment_dir(tmp_path / "exp")
        output = tmp_path / "report" / "report.md"

        # Mock plot_all_curves to avoid needing full curve infrastructure
        with patch("src.analysis.collapse_curves.plot_all_curves") as mock_curves:
            # Make plot_all_curves create a dummy PNG and return paths
            def fake_plot_all_curves(df, output_dir):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                dummy = output_dir / "entropy_curve.png"
                fig, ax = plt.subplots()
                ax.plot([0, 1], [0, 1])
                fig.savefig(dummy)
                plt.close(fig)
                return [dummy]

            mock_curves.side_effect = fake_plot_all_curves

            result = generate_report(exp_dir, output)

        assert result.exists()
        content = result.read_text()
        assert "# Model Collapse Experiment Report" in content

    def test_report_has_all_sections(self, tmp_path):
        exp_dir = _create_experiment_dir(tmp_path / "exp")
        output = tmp_path / "report" / "report.md"

        with patch("src.analysis.collapse_curves.plot_all_curves") as mock_curves:
            def fake_plot_all_curves(df, output_dir):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                dummy = output_dir / "dummy.png"
                fig, ax = plt.subplots()
                fig.savefig(dummy)
                plt.close(fig)
                return [dummy]

            mock_curves.side_effect = fake_plot_all_curves

            result = generate_report(exp_dir, output)

        content = result.read_text()
        assert "## Configuration" in content
        assert "## Per-Schedule Collapse Curves" in content
        assert "## Fixed-Point Analysis" in content
        assert "## Collapse Rate Table" in content
        assert "## Recommendations" in content

    def test_report_with_phase_diagrams(self, tmp_path):
        """Multiple schedules should trigger phase diagram section."""
        exp_dir = _create_experiment_dir(tmp_path / "exp", schedules=["sched_a", "sched_b"])
        output = tmp_path / "report" / "report.md"

        with patch("src.analysis.collapse_curves.plot_all_curves") as mock_curves:
            def fake_plot_all_curves(df, output_dir):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                dummy = output_dir / "dummy.png"
                fig, ax = plt.subplots()
                fig.savefig(dummy)
                plt.close(fig)
                return [dummy]

            mock_curves.side_effect = fake_plot_all_curves

            result = generate_report(exp_dir, output)

        content = result.read_text()
        assert "## Phase Diagrams" in content

    def test_report_without_config(self, tmp_path):
        """Missing config should produce fallback text."""
        exp_dir = _create_experiment_dir(tmp_path / "exp", with_config=False)
        output = tmp_path / "report" / "report.md"

        with patch("src.analysis.collapse_curves.plot_all_curves") as mock_curves:
            def fake_plot_all_curves(df, output_dir):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                dummy = output_dir / "dummy.png"
                fig, ax = plt.subplots()
                fig.savefig(dummy)
                plt.close(fig)
                return [dummy]

            mock_curves.side_effect = fake_plot_all_curves

            result = generate_report(exp_dir, output)

        content = result.read_text()
        assert "_No configuration file found._" in content

    def test_report_with_scale_comparison(self, tmp_path):
        """Scale comparison data should produce scale section."""
        exp_dir = _create_experiment_dir(
            tmp_path / "exp",
            schedules=["sched_a"],
            with_scale_data=True,
        )
        output = tmp_path / "report" / "report.md"

        with patch("src.analysis.collapse_curves.plot_all_curves") as mock_curves, \
             patch("src.analysis.scale_comparison.plot_scale_comparison_panel") as mock_scale_plot, \
             patch("src.analysis.scale_comparison.compute_scale_interaction_stats") as mock_scale_stats:

            def fake_plot_all_curves(df, output_dir):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                dummy = output_dir / "dummy.png"
                fig, ax = plt.subplots()
                fig.savefig(dummy)
                plt.close(fig)
                return [dummy]

            mock_curves.side_effect = fake_plot_all_curves

            scale_png = tmp_path / "report" / "plots" / "scale_comparison_panel.png"
            scale_png.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots()
            fig.savefig(scale_png)
            plt.close(fig)
            mock_scale_plot.return_value = scale_png

            mock_scale_stats.return_value = {
                "collapse_rate_ratio": 2.0,
                "entropy_floors": {"1b": 2.5, "7b": 3.0},
            }

            result = generate_report(exp_dir, output)

        content = result.read_text()
        assert "## Scale Comparison (1B vs 7B)" in content
        assert "### Scale Interaction Statistics" in content


class TestReportHelpers:
    """Test report helper functions directly."""

    def test_load_config_json(self, tmp_path):
        config = {"model": "test", "lr": 0.001}
        (tmp_path / "config.json").write_text(json.dumps(config))
        result = _load_config(tmp_path)
        assert result == config

    def test_load_config_missing(self, tmp_path):
        result = _load_config(tmp_path)
        assert result is None

    def test_load_all_schedule_metrics(self, tmp_path):
        _create_experiment_dir(tmp_path, schedules=["sched_a", "sched_b"], with_config=False)
        result = _load_all_schedule_metrics(tmp_path)
        assert "sched_a" in result
        assert "sched_b" in result
        assert len(result["sched_a"]) == 5

    def test_load_single_schedule_metrics(self, tmp_path):
        """Fallback to single-schedule pattern (metrics subdir found by Pattern 1).

        The _load_all_schedule_metrics function first iterates subdirectories
        looking for <subdir>/metrics/metrics.json or <subdir>/metrics.json.
        A 'metrics' subdirectory containing metrics.json will be found
        via Pattern 1 with key 'metrics'.
        """
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir(parents=True)
        records = _make_metrics_records(3)
        (metrics_dir / "metrics.json").write_text(json.dumps(records))
        result = _load_all_schedule_metrics(tmp_path)
        # The 'metrics' directory is found via Pattern 1 as a schedule named "metrics"
        assert "metrics" in result
        assert len(result["metrics"]) == 3

    def test_find_scale_data_present(self, tmp_path):
        _create_experiment_dir(tmp_path, schedules=[], with_scale_data=True)
        m1b, m7b = _find_scale_data(tmp_path)
        assert m1b is not None
        assert m7b is not None

    def test_find_scale_data_absent(self, tmp_path):
        m1b, m7b = _find_scale_data(tmp_path)
        assert m1b is None
        assert m7b is None

    def test_build_fixed_point_table(self):
        df = pd.DataFrame(_make_metrics_records(5))
        table = _build_fixed_point_table({"test_sched": df})
        assert any("test_sched" in line for line in table)
        assert any("Schedule" in line for line in table)

    def test_build_fixed_point_table_no_kl(self):
        """Schedule with no kl_divergence column."""
        df = pd.DataFrame({"generation": [0, 1, 2], "entropy": [5.0, 4.0, 3.0]})
        table = _build_fixed_point_table({"no_kl": df})
        assert any("no_kl" in line for line in table)

    def test_build_collapse_rate_table(self):
        df = pd.DataFrame(_make_metrics_records(5))
        table = _build_collapse_rate_table({"sched_x": df})
        assert any("sched_x" in line for line in table)

    def test_build_collapse_rate_table_single_gen(self):
        df = pd.DataFrame(_make_metrics_records(1))
        table = _build_collapse_rate_table({"tiny": df})
        assert any("N/A" in line for line in table)

    def test_generate_recommendations_no_collapse(self):
        df = pd.DataFrame({
            "generation": [0, 1, 2],
            "kl_divergence": [0.1, 0.15, 0.2],
        })
        recs = _generate_recommendations({"stable": df}, None)
        assert any("stable" in r.lower() or "no immediate" in r.lower() for r in recs)

    def test_generate_recommendations_with_collapse(self):
        df = pd.DataFrame({
            "generation": [0, 1, 2],
            "kl_divergence": [0.1, 0.8, 1.5],
        })
        recs = _generate_recommendations({"collapsing": df}, None)
        assert any("collapse" in r.lower() or "alpha" in r.lower() for r in recs)

    def test_generate_recommendations_with_scale_stats(self):
        df = pd.DataFrame({
            "generation": [0, 1, 2],
            "kl_divergence": [0.1, 0.2, 0.3],
        })
        scale_stats = {"collapse_rate_ratio": 2.5}
        recs = _generate_recommendations({"sched": df}, scale_stats)
        assert any("1B" in r for r in recs)

    def test_generate_recommendations_empty_metrics(self):
        recs = _generate_recommendations({}, None)
        assert any("no metrics" in r.lower() for r in recs)

    def test_generate_recommendations_scale_ratio_low(self):
        """When 7B collapses faster (ratio < 0.67)."""
        df = pd.DataFrame({
            "generation": [0, 1, 2],
            "kl_divergence": [0.1, 0.2, 0.3],
        })
        scale_stats = {"collapse_rate_ratio": 0.5}
        recs = _generate_recommendations({"sched": df}, scale_stats)
        assert any("7B" in r or "LoRA" in r for r in recs)


# ======================================================================
# PHASE DIAGRAMS (src/analysis/phase_diagrams.py)
# ======================================================================

from src.analysis.phase_diagrams import plot_phase_diagram, plot_collapse_boundary


class TestPhaseDiagramPlot:
    """Test plot_phase_diagram with synthetic data."""

    def _make_runs(self, n_schedules=3, n_gens=5) -> dict[str, pd.DataFrame]:
        runs = {}
        for s in range(n_schedules):
            name = f"schedule_{s}"
            records = []
            for g in range(n_gens):
                records.append({
                    "generation": g,
                    "kl_divergence": 0.1 * (s + 1) * (g + 1),
                    "entropy": 5.0 - 0.2 * (s + 1) * g,
                })
            runs[name] = pd.DataFrame(records)
        return runs

    def test_phase_diagram_saves_file(self, tmp_path):
        runs = self._make_runs()
        output = tmp_path / "phase_kl.png"
        result = plot_phase_diagram(runs, "kl_divergence", output)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_phase_diagram_returns_path(self, tmp_path):
        runs = self._make_runs()
        output = tmp_path / "phase.png"
        result = plot_phase_diagram(runs, "entropy", output)
        assert result == output

    def test_phase_diagram_empty_runs(self, tmp_path):
        output = tmp_path / "phase_empty.png"
        result = plot_phase_diagram({}, "kl_divergence", output)
        assert result.exists()  # Should produce a "No data" plot

    def test_phase_diagram_single_schedule(self, tmp_path):
        runs = self._make_runs(n_schedules=1, n_gens=5)
        output = tmp_path / "phase_single.png"
        result = plot_phase_diagram(runs, "kl_divergence", output)
        assert result.exists()

    def test_phase_diagram_single_generation(self, tmp_path):
        runs = self._make_runs(n_schedules=3, n_gens=1)
        output = tmp_path / "phase_one_gen.png"
        result = plot_phase_diagram(runs, "kl_divergence", output)
        assert result.exists()

    def test_phase_diagram_missing_metric(self, tmp_path):
        """If metric_name column doesn't exist, cells should be NaN."""
        runs = self._make_runs(n_schedules=2, n_gens=3)
        output = tmp_path / "phase_missing.png"
        # "nonexistent_metric" is not in the DataFrames
        result = plot_phase_diagram(runs, "nonexistent_metric", output)
        assert result.exists()


class TestCollapseBoundaryPlot:
    """Test plot_collapse_boundary with synthetic data."""

    def _make_runs(self, n_schedules=3, n_gens=5) -> dict[str, pd.DataFrame]:
        runs = {}
        for s in range(n_schedules):
            name = f"schedule_{s}"
            records = []
            for g in range(n_gens):
                records.append({
                    "generation": g,
                    "kl_divergence": 0.1 * (s + 1) * (g + 1),
                })
            runs[name] = pd.DataFrame(records)
        return runs

    def test_collapse_boundary_saves_file(self, tmp_path):
        runs = self._make_runs()
        output = tmp_path / "boundary.png"
        result = plot_collapse_boundary(runs, collapse_threshold=0.5, output_path=output)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_collapse_boundary_empty_runs(self, tmp_path):
        output = tmp_path / "boundary_empty.png"
        result = plot_collapse_boundary({}, collapse_threshold=0.5, output_path=output)
        assert result.exists()

    def test_collapse_boundary_single_schedule(self, tmp_path):
        runs = self._make_runs(n_schedules=1, n_gens=5)
        output = tmp_path / "boundary_single.png"
        result = plot_collapse_boundary(runs, collapse_threshold=0.3, output_path=output)
        assert result.exists()

    def test_collapse_boundary_single_generation(self, tmp_path):
        runs = self._make_runs(n_schedules=3, n_gens=1)
        output = tmp_path / "boundary_one_gen.png"
        result = plot_collapse_boundary(runs, collapse_threshold=0.5, output_path=output)
        assert result.exists()

    def test_collapse_boundary_high_threshold(self, tmp_path):
        """Threshold above all values: no contour crossing, still no crash."""
        runs = self._make_runs(n_schedules=2, n_gens=3)
        output = tmp_path / "boundary_high.png"
        result = plot_collapse_boundary(runs, collapse_threshold=999.0, output_path=output)
        assert result.exists()

    def test_collapse_boundary_custom_metric(self, tmp_path):
        """Use a custom metric_name instead of default kl_divergence."""
        runs = {}
        for s in range(2):
            records = []
            for g in range(4):
                records.append({
                    "generation": g,
                    "entropy": 5.0 - 0.3 * s * g,
                })
            runs[f"sched_{s}"] = pd.DataFrame(records)
        output = tmp_path / "boundary_entropy.png"
        result = plot_collapse_boundary(
            runs, collapse_threshold=4.0, output_path=output, metric_name="entropy"
        )
        assert result.exists()
