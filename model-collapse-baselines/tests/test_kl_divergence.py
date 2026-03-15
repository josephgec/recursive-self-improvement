"""Tests for src.measurement.kl_divergence.KLDivergenceEstimator."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.measurement.kl_divergence import KLDivergenceEstimator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_mock_tokenizer(vocab_size: int = 100):
    """Create a mock tokenizer for KL tests."""
    tokenizer = MagicMock()

    def encode_fn(text, **kwargs):
        # Deterministic encoding: hash each word to a token id.
        words = text.split()
        return [hash(w) % vocab_size for w in words]

    tokenizer.encode = encode_fn
    tokenizer.__len__ = MagicMock(return_value=vocab_size)
    return tokenizer


def _uniform_distribution(vocab_size: int) -> np.ndarray:
    """Create a uniform distribution over vocab_size tokens."""
    return np.ones(vocab_size, dtype=np.float64) / vocab_size


def _peaked_distribution(vocab_size: int, peak_token: int = 0) -> np.ndarray:
    """Create a distribution with most mass on one token."""
    dist = np.ones(vocab_size, dtype=np.float64) * 0.001
    dist[peak_token] = 1.0
    dist /= dist.sum()
    return dist


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestKLSelfDivergence:
    """KL(P || P) should be approximately zero."""

    def test_kl_self_is_near_zero(self):
        vocab_size = 100
        p = _uniform_distribution(vocab_size)
        tokenizer = _make_mock_tokenizer(vocab_size)
        estimator = KLDivergenceEstimator(p, tokenizer)

        result = estimator._compute_divergences(p, p)
        assert abs(result.kl_p_q) < 1e-6, (
            f"KL(P || P) should be ~0, got {result.kl_p_q}"
        )
        assert abs(result.kl_q_p) < 1e-6
        assert abs(result.js_divergence) < 1e-6

    def test_kl_self_peaked(self):
        """KL(P || P) = 0 even for non-uniform P."""
        vocab_size = 50
        p = _peaked_distribution(vocab_size)
        tokenizer = _make_mock_tokenizer(vocab_size)
        estimator = KLDivergenceEstimator(p, tokenizer)

        result = estimator._compute_divergences(p, p)
        assert abs(result.kl_p_q) < 1e-5


class TestKLPositive:
    """KL(P || Q) should be positive when P != Q."""

    def test_kl_uniform_vs_peaked(self):
        vocab_size = 100
        p = _uniform_distribution(vocab_size)
        q = _peaked_distribution(vocab_size)
        tokenizer = _make_mock_tokenizer(vocab_size)
        estimator = KLDivergenceEstimator(p, tokenizer)

        result = estimator._compute_divergences(p, q)
        assert result.kl_p_q > 0, "KL(uniform || peaked) should be > 0"
        assert result.kl_q_p > 0, "KL(peaked || uniform) should be > 0"

    def test_kl_peaked_vs_uniform(self):
        vocab_size = 100
        p = _peaked_distribution(vocab_size)
        q = _uniform_distribution(vocab_size)
        tokenizer = _make_mock_tokenizer(vocab_size)
        estimator = KLDivergenceEstimator(p, tokenizer)

        result = estimator._compute_divergences(p, q)
        assert result.kl_p_q > 0


class TestJSSymmetry:
    """Jensen-Shannon divergence should be symmetric."""

    def test_js_symmetric(self):
        vocab_size = 100
        p = _uniform_distribution(vocab_size)
        q = _peaked_distribution(vocab_size)
        tokenizer = _make_mock_tokenizer(vocab_size)
        estimator = KLDivergenceEstimator(p, tokenizer)

        result_pq = estimator._compute_divergences(p, q)
        result_qp = estimator._compute_divergences(q, p)

        assert abs(result_pq.js_divergence - result_qp.js_divergence) < 1e-10, (
            f"JS should be symmetric: {result_pq.js_divergence} vs {result_qp.js_divergence}"
        )

    def test_js_bounded(self):
        """JS divergence should be in [0, log(2)]."""
        vocab_size = 100
        p = _uniform_distribution(vocab_size)
        q = _peaked_distribution(vocab_size)
        tokenizer = _make_mock_tokenizer(vocab_size)
        estimator = KLDivergenceEstimator(p, tokenizer)

        result = estimator._compute_divergences(p, q)
        assert 0 <= result.js_divergence <= np.log(2) + 1e-6


class TestSmoothingPreventsNaN:
    """Smoothing should prevent NaN and Inf in all outputs."""

    def test_zero_entries_no_nan(self):
        """Distribution with zeros should not produce NaN."""
        vocab_size = 100
        p = np.zeros(vocab_size, dtype=np.float64)
        p[0] = 1.0  # all mass on one token
        q = np.zeros(vocab_size, dtype=np.float64)
        q[1] = 1.0  # all mass on a *different* token

        tokenizer = _make_mock_tokenizer(vocab_size)
        estimator = KLDivergenceEstimator(p, tokenizer)

        result = estimator._compute_divergences(p, q)
        assert np.isfinite(result.kl_p_q), "KL should be finite"
        assert np.isfinite(result.kl_q_p), "KL should be finite"
        assert np.isfinite(result.js_divergence), "JS should be finite"
        assert np.isfinite(result.total_variation), "TV should be finite"

    def test_all_zero_distribution(self):
        """All-zero distribution should not crash."""
        vocab_size = 50
        p = np.zeros(vocab_size, dtype=np.float64)
        q = _uniform_distribution(vocab_size)
        tokenizer = _make_mock_tokenizer(vocab_size)
        estimator = KLDivergenceEstimator(p, tokenizer)

        result = estimator._compute_divergences(p, q)
        assert np.isfinite(result.kl_p_q)
        assert np.isfinite(result.js_divergence)


class TestTotalVariation:
    """Total variation distance properties."""

    def test_tv_self_is_zero(self):
        """TV(P, P) = 0."""
        vocab_size = 100
        p = _uniform_distribution(vocab_size)
        tokenizer = _make_mock_tokenizer(vocab_size)
        estimator = KLDivergenceEstimator(p, tokenizer)

        result = estimator._compute_divergences(p, p)
        assert abs(result.total_variation) < 1e-6

    def test_tv_bounded(self):
        """TV should be in [0, 1]."""
        vocab_size = 100
        p = _uniform_distribution(vocab_size)
        q = _peaked_distribution(vocab_size)
        tokenizer = _make_mock_tokenizer(vocab_size)
        estimator = KLDivergenceEstimator(p, tokenizer)

        result = estimator._compute_divergences(p, q)
        assert 0.0 <= result.total_variation <= 1.0 + 1e-6


class TestFromGeneratedText:
    """Test KL estimation from pre-generated text."""

    def test_estimate_from_text(self):
        """Should produce finite divergence values from text."""
        vocab_size = 100
        p = _uniform_distribution(vocab_size)
        tokenizer = _make_mock_tokenizer(vocab_size)
        estimator = KLDivergenceEstimator(p, tokenizer)

        texts = [
            "the cat sat on the mat",
            "a quick brown fox jumps over",
            "she sells sea shells by the shore",
        ]
        result = estimator.estimate_from_generated_text(texts)

        assert np.isfinite(result.kl_p_q)
        assert np.isfinite(result.kl_q_p)
        assert np.isfinite(result.js_divergence)
        assert np.isfinite(result.total_variation)

    def test_token_distribution_sums_to_one(self):
        """Token distribution should be a valid probability distribution."""
        vocab_size = 100
        p = _uniform_distribution(vocab_size)
        tokenizer = _make_mock_tokenizer(vocab_size)
        estimator = KLDivergenceEstimator(p, tokenizer)

        texts = ["hello world foo bar baz"]
        dist = estimator.compute_token_distribution(texts)

        assert abs(dist.sum() - 1.0) < 1e-10
        assert len(dist) == vocab_size
        assert (dist > 0).all()  # Laplace smoothing ensures no zeros
