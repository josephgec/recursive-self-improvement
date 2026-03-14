"""Embeddings package for the data-contamination-audit pipeline.

Re-exports the public API of the encoder, similarity, and temporal-curves
sub-modules.
"""

from src.embeddings.encoder import DocumentEncoder
from src.embeddings.similarity import (
    corpus_mean_similarity,
    cross_corpus_similarity,
    pairwise_cosine_similarity,
    similarity_percentiles,
)
from src.embeddings.temporal_curves import (
    compute_temporal_curve,
    detect_inflection_point,
)

__all__ = [
    "DocumentEncoder",
    "compute_temporal_curve",
    "corpus_mean_similarity",
    "cross_corpus_similarity",
    "detect_inflection_point",
    "pairwise_cosine_similarity",
    "similarity_percentiles",
]
