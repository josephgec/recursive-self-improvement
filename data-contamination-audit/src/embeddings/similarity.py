"""Cosine-similarity utilities for pre-normalised embedding matrices.

All functions assume that input embeddings are already L2-normalised to unit
vectors, so cosine similarity reduces to a simple dot product.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Return the upper-triangle pairwise cosine similarities as a flat array.

    For *n* embeddings the result has ``n * (n - 1) / 2`` elements,
    corresponding to every unique ``(i, j)`` pair with ``i < j``.

    Parameters
    ----------
    embeddings:
        Unit-normalised float array of shape ``(n, d)``.

    Returns
    -------
    np.ndarray
        1-D float32 array of upper-triangle similarities.
    """
    # Dot product of unit vectors == cosine similarity.
    sim_matrix = embeddings @ embeddings.T
    # Extract upper triangle (k=1 excludes the diagonal).
    return sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)].astype(np.float32)


def corpus_mean_similarity(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine similarity within a corpus.

    For corpora larger than 50 000 documents the exact computation is
    prohibitively expensive (O(n^2)), so 10 000 random pairs are sampled
    instead.

    Parameters
    ----------
    embeddings:
        Unit-normalised float array of shape ``(n, d)``.

    Returns
    -------
    float
        Scalar mean similarity.
    """
    n = embeddings.shape[0]
    if n < 2:
        return 1.0

    if n <= 50_000:
        sims = pairwise_cosine_similarity(embeddings)
        return float(np.mean(sims))

    # Random sampling for large corpora.
    logger.info(
        "Corpus has %d docs (> 50K) — sampling 10K pairs for mean similarity", n
    )
    rng = np.random.default_rng(42)
    n_pairs = 10_000
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    # Ensure i != j.
    mask = idx_a == idx_b
    idx_b[mask] = (idx_b[mask] + 1) % n

    sims = np.sum(embeddings[idx_a] * embeddings[idx_b], axis=1)
    return float(np.mean(sims))


def cross_corpus_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Mean cosine similarity between every pair from corpus A and corpus B.

    For large corpora (product > 50K pairs) a random sample of 10 000
    cross-corpus pairs is used.

    Parameters
    ----------
    emb_a:
        Unit-normalised float array of shape ``(n_a, d)``.
    emb_b:
        Unit-normalised float array of shape ``(n_b, d)``.

    Returns
    -------
    float
        Scalar mean cross-corpus similarity.
    """
    n_a, n_b = emb_a.shape[0], emb_b.shape[0]
    if n_a == 0 or n_b == 0:
        return 0.0

    total_pairs = n_a * n_b
    if total_pairs <= 50_000:
        # Full cross-similarity matrix.
        sim_matrix = emb_a @ emb_b.T
        return float(np.mean(sim_matrix))

    # Random sampling.
    logger.info(
        "Cross-corpus has %d x %d = %d pairs (> 50K) — sampling 10K pairs",
        n_a,
        n_b,
        total_pairs,
    )
    rng = np.random.default_rng(42)
    n_pairs = 10_000
    idx_a = rng.integers(0, n_a, size=n_pairs)
    idx_b = rng.integers(0, n_b, size=n_pairs)
    sims = np.sum(emb_a[idx_a] * emb_b[idx_b], axis=1)
    return float(np.mean(sims))


def similarity_percentiles(
    embeddings: np.ndarray,
    percentiles: list[int] | None = None,
) -> dict[int, float]:
    """Compute distributional percentiles of pairwise cosine similarities.

    Parameters
    ----------
    embeddings:
        Unit-normalised float array of shape ``(n, d)``.
    percentiles:
        Which percentiles to report (default ``[5, 25, 50, 75, 95]``).

    Returns
    -------
    dict[int, float]
        Mapping from percentile value to similarity score.
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    sims = pairwise_cosine_similarity(embeddings)
    if sims.size == 0:
        return {p: 0.0 for p in percentiles}

    values = np.percentile(sims, percentiles)
    return {int(p): float(v) for p, v in zip(percentiles, values)}
