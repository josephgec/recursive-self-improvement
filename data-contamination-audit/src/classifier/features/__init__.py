"""Classifier feature extraction modules.

Public API
----------
- :class:`PerplexityScorer` — perplexity-based features using a reference LM.
- :class:`WatermarkDetector` — Kirchenbauer et al. (2023) green-list watermark
  detection.
- :func:`compute_stylometric_features` — 12 regex-based stylometric features.
- :func:`extract_all_features` — combine all feature families for one document.
- :func:`build_feature_matrix` — build a feature DataFrame for a corpus.
"""

from .ensemble import build_feature_matrix, extract_all_features
from .perplexity import PerplexityScorer
from .stylometry import compute_stylometric_features
from .watermark import WatermarkDetector

__all__ = [
    "PerplexityScorer",
    "WatermarkDetector",
    "compute_stylometric_features",
    "extract_all_features",
    "build_feature_matrix",
]
