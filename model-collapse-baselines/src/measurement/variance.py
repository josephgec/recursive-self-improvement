"""Variance and embedding-space diversity tracking.

Measures how the distribution of generated text evolves in embedding
space across generations, detecting mode collapse via reduced variance,
increased pairwise similarity, and drift from the reference distribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class VarianceResult:
    """Results from variance and embedding analysis."""

    embedding_variance: float  # trace of covariance matrix
    mean_pairwise_cosine: float  # avg cosine sim between all pairs
    drift_from_reference: float  # cosine distance of centroids
    effective_dimension: int  # PCA dims for 95% variance
    cluster_count: int  # number of clusters found


class VarianceTracker:
    """Track embedding-space variance across generations.

    Uses a sentence-transformer model to embed generated texts and then
    analyses the geometry of the embedding cloud to detect model collapse.

    Args:
        embedding_model: A ``sentence_transformers.SentenceTransformer`` or
            any object with an ``encode(list[str])`` method returning an
            ndarray of shape ``(n, d)``.
        reference_embeddings: 2-D array of shape ``(n_ref, d)`` containing
            embeddings of the reference (real data) corpus.
    """

    def __init__(
        self,
        embedding_model,
        reference_embeddings: np.ndarray,
    ) -> None:
        self._encoder = embedding_model
        self._reference = np.asarray(reference_embeddings, dtype=np.float64)
        self._reference_centroid = self._reference.mean(axis=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure(self, texts: list[str]) -> VarianceResult:
        """Analyse the embedding-space properties of *texts*.

        Args:
            texts: List of generated text strings.

        Returns:
            A ``VarianceResult`` with variance, similarity, drift,
            effective dimension, and cluster count.
        """
        if not texts:
            return VarianceResult(
                embedding_variance=0.0,
                mean_pairwise_cosine=1.0,
                drift_from_reference=0.0,
                effective_dimension=0,
                cluster_count=0,
            )

        embeddings = self._encoder.encode(texts, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float64)

        emb_var = self._embedding_variance(embeddings)
        mean_cos = self._mean_pairwise_cosine(embeddings)
        drift = self._drift_from_reference(embeddings)
        eff_dim = self._effective_dimension(embeddings)
        n_clusters = self._cluster_count(embeddings)

        return VarianceResult(
            embedding_variance=emb_var,
            mean_pairwise_cosine=mean_cos,
            drift_from_reference=drift,
            effective_dimension=eff_dim,
            cluster_count=n_clusters,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _embedding_variance(embeddings: np.ndarray) -> float:
        """Trace of the covariance matrix (total variance)."""
        if embeddings.shape[0] < 2:
            return 0.0
        cov = np.cov(embeddings, rowvar=False)
        return float(np.trace(cov))

    @staticmethod
    def _mean_pairwise_cosine(
        embeddings: np.ndarray, max_pairs: int = 5000
    ) -> float:
        """Average pairwise cosine similarity (sampled if too many pairs)."""
        n = embeddings.shape[0]
        if n < 2:
            return 1.0

        # Normalize for cosine similarity.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = embeddings / norms

        # If small enough, compute full matrix.
        if n * (n - 1) // 2 <= max_pairs:
            sim_matrix = normed @ normed.T
            # Extract upper triangle (excluding diagonal).
            idx = np.triu_indices(n, k=1)
            return float(np.mean(sim_matrix[idx]))

        # Sample random pairs.
        rng = np.random.RandomState(42)
        idx_a = rng.randint(0, n, size=max_pairs)
        idx_b = rng.randint(0, n, size=max_pairs)
        # Avoid self-pairs.
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]
        sims = np.sum(normed[idx_a] * normed[idx_b], axis=1)
        return float(np.mean(sims))

    def _drift_from_reference(self, embeddings: np.ndarray) -> float:
        """Cosine distance between the centroid of *embeddings* and reference."""
        centroid = embeddings.mean(axis=0)
        return float(self._cosine_distance(centroid, self._reference_centroid))

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """1 - cosine_similarity(a, b)."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 1.0
        return 1.0 - dot / (norm_a * norm_b)

    @staticmethod
    def _effective_dimension(
        embeddings: np.ndarray, explained_threshold: float = 0.95
    ) -> int:
        """Number of PCA components needed to explain 95% of variance."""
        from sklearn.decomposition import PCA

        if embeddings.shape[0] < 2:
            return 0

        n_components = min(embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(embeddings)

        cumulative = np.cumsum(pca.explained_variance_ratio_)
        dims_needed = int(np.searchsorted(cumulative, explained_threshold) + 1)
        return min(dims_needed, n_components)

    @staticmethod
    def _cluster_count(
        embeddings: np.ndarray, max_k: int = 20, min_samples: int = 10
    ) -> int:
        """Estimate number of clusters using silhouette score.

        Tries k=2..max_k and returns the k with the best silhouette
        score.  Falls back to 1 if clustering fails or the data is too
        small.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n = embeddings.shape[0]
        if n < min_samples:
            return 1

        max_k = min(max_k, n - 1)
        if max_k < 2:
            return 1

        best_k = 1
        best_score = -1.0

        for k in range(2, max_k + 1):
            try:
                km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
                labels = km.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels, sample_size=min(n, 1000))
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        return best_k
