"""Semantic drift signal using word-frequency cosine distance."""

import math
import re
from collections import Counter
from typing import Dict, List, Optional

from .base import DriftSignal, SignalResult


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def _word_freq_vector(texts: List[str]) -> Counter:
    """Build a word frequency counter from a list of texts."""
    counter: Counter = Counter()
    for text in texts:
        counter.update(_tokenize(text))
    return counter


def _cosine_distance(vec_a: Counter, vec_b: Counter) -> float:
    """Compute cosine distance between two frequency vectors.

    Returns value in [0, 1] where 0 = identical, 1 = orthogonal.
    """
    all_keys = set(vec_a.keys()) | set(vec_b.keys())
    if not all_keys:
        return 0.0

    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in all_keys)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 1.0

    cosine_sim = dot / (mag_a * mag_b)
    # Clamp to [0, 1] to handle floating point errors
    cosine_sim = max(0.0, min(1.0, cosine_sim))
    return 1.0 - cosine_sim


class SemanticDriftSignal(DriftSignal):
    """Semantic drift signal using word-frequency cosine distance.

    No sentence-transformers needed — uses word frequency vectors as a
    lightweight proxy for semantic similarity.
    """

    @property
    def name(self) -> str:
        return "semantic"

    def centroid_distance(
        self, current: List[str], reference: List[str]
    ) -> float:
        """Cosine distance of mean word-frequency vectors."""
        vec_cur = _word_freq_vector(current)
        vec_ref = _word_freq_vector(reference)
        return _cosine_distance(vec_cur, vec_ref)

    def pairwise_drift(
        self, current: List[str], reference: List[str]
    ) -> float:
        """Mean pairwise cosine distance between outputs."""
        if not current or not reference:
            return 1.0

        distances = []
        for c in current:
            for r in reference:
                vec_c = _word_freq_vector([c])
                vec_r = _word_freq_vector([r])
                distances.append(_cosine_distance(vec_c, vec_r))

        return sum(distances) / len(distances) if distances else 0.0

    def mmd(self, current: List[str], reference: List[str]) -> float:
        """Simplified Maximum Mean Discrepancy.

        Computes MMD as: mean(d(c,c')) + mean(d(r,r')) - 2*mean(d(c,r))
        using cosine distance as the kernel proxy.
        """
        if not current or not reference:
            return 1.0

        def _mean_dist(texts_a: List[str], texts_b: List[str]) -> float:
            dists = []
            for a in texts_a:
                for b in texts_b:
                    va = _word_freq_vector([a])
                    vb = _word_freq_vector([b])
                    dists.append(_cosine_distance(va, vb))
            return sum(dists) / len(dists) if dists else 0.0

        cc = _mean_dist(current, current)
        rr = _mean_dist(reference, reference)
        cr = _mean_dist(current, reference)

        # MMD^2 estimate; clamp to >= 0
        mmd_val = cc + rr - 2 * cr
        return max(0.0, mmd_val)

    def compute(
        self, current: List[str], reference: List[str]
    ) -> SignalResult:
        """Compute semantic drift using centroid distance as primary metric."""
        centroid = self.centroid_distance(current, reference)
        pairwise = self.pairwise_drift(current, reference)
        mmd_val = self.mmd(current, reference)

        # Primary score is centroid distance
        raw = centroid
        normalized = self.normalize(raw)

        return SignalResult(
            signal_name=self.name,
            raw_score=raw,
            normalized_score=normalized,
            interpretation=self.interpret(normalized),
            components={
                "centroid_distance": centroid,
                "pairwise_drift": pairwise,
                "mmd": mmd_val,
            },
        )

    def normalize(self, raw: float) -> float:
        """Normalize: raw / 0.5, capped at 1.0."""
        return min(1.0, raw / 0.5)
