"""Distributional drift signal using token-level distribution metrics."""

import math
import re
from collections import Counter
from typing import Dict, List

from .base import DriftSignal, SignalResult

# Default Laplace smoothing parameter
SMOOTHING_ALPHA = 1e-6


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def _build_distribution(
    texts: List[str], smoothing: float = SMOOTHING_ALPHA
) -> Dict[str, float]:
    """Build a smoothed word probability distribution.

    Uses Laplace smoothing to prevent zero probabilities.
    """
    counter: Counter = Counter()
    for text in texts:
        counter.update(_tokenize(text))

    if not counter:
        return {}

    # Get vocabulary for smoothing
    vocab_size = len(counter)
    total = sum(counter.values())

    # Apply Laplace smoothing
    smoothed_total = total + smoothing * vocab_size
    return {
        k: (v + smoothing) / smoothed_total
        for k, v in counter.items()
    }


def _merged_vocab(
    dist_a: Dict[str, float], dist_b: Dict[str, float]
) -> set:
    """Get merged vocabulary from two distributions."""
    return set(dist_a.keys()) | set(dist_b.keys())


class DistributionalDriftSignal(DriftSignal):
    """Distributional drift signal using token-level distribution metrics.

    Measures drift in word-level token distributions using KL divergence,
    total variation distance, and Jensen-Shannon divergence.
    """

    def __init__(self, smoothing: float = SMOOTHING_ALPHA):
        self.smoothing = smoothing

    @property
    def name(self) -> str:
        return "distributional"

    def kl_divergence(
        self,
        current: List[str],
        reference: List[str],
        reverse: bool = False,
    ) -> float:
        """KL divergence from reference to current (or reverse).

        With Laplace smoothing to prevent infinite values.

        Args:
            current: Current output texts.
            reference: Reference output texts.
            reverse: If True, compute KL(reference || current).

        Returns:
            KL divergence value (non-negative).
        """
        p_dist = _build_distribution(current, self.smoothing)
        q_dist = _build_distribution(reference, self.smoothing)

        if not p_dist or not q_dist:
            return 0.0 if (not p_dist and not q_dist) else 1.0

        if reverse:
            p_dist, q_dist = q_dist, p_dist

        vocab = _merged_vocab(p_dist, q_dist)

        # Apply smoothing to missing keys
        total_p = sum(p_dist.values())
        total_q = sum(q_dist.values())

        kl = 0.0
        for word in vocab:
            p = p_dist.get(word, self.smoothing / (total_p + self.smoothing * len(vocab)))
            q = q_dist.get(word, self.smoothing / (total_q + self.smoothing * len(vocab)))
            if p > 0 and q > 0:
                kl += p * math.log(p / q)

        return max(0.0, kl)

    def total_variation(
        self, current: List[str], reference: List[str]
    ) -> float:
        """Total variation distance between distributions.

        Returns value in [0, 1].
        """
        p_dist = _build_distribution(current, self.smoothing)
        q_dist = _build_distribution(reference, self.smoothing)

        if not p_dist and not q_dist:
            return 0.0
        if not p_dist or not q_dist:
            return 1.0

        vocab = _merged_vocab(p_dist, q_dist)

        tv = 0.0
        for word in vocab:
            p = p_dist.get(word, 0.0)
            q = q_dist.get(word, 0.0)
            tv += abs(p - q)

        return min(1.0, tv / 2.0)

    def js_divergence(
        self, current: List[str], reference: List[str]
    ) -> float:
        """Jensen-Shannon divergence between distributions.

        Returns value in [0, 1] (using natural log, normalized).
        """
        p_dist = _build_distribution(current, self.smoothing)
        q_dist = _build_distribution(reference, self.smoothing)

        if not p_dist and not q_dist:
            return 0.0
        if not p_dist or not q_dist:
            return 1.0

        vocab = _merged_vocab(p_dist, q_dist)

        # Compute midpoint distribution
        m = {}
        for word in vocab:
            m[word] = 0.5 * (p_dist.get(word, 0.0) + q_dist.get(word, 0.0))

        def _kl_from(dist: Dict[str, float]) -> float:
            result = 0.0
            for word in vocab:
                p = dist.get(word, 0.0)
                mw = m.get(word, 0.0)
                if p > 0 and mw > 0:
                    result += p * math.log2(p / mw)
            return result

        js = 0.5 * _kl_from(p_dist) + 0.5 * _kl_from(q_dist)
        return min(1.0, max(0.0, js))

    def compute(
        self, current: List[str], reference: List[str]
    ) -> SignalResult:
        """Compute distributional drift.

        Uses average of forward KL, reverse KL, total variation, and JS divergence.
        """
        kl_fwd = self.kl_divergence(current, reference, reverse=False)
        kl_rev = self.kl_divergence(current, reference, reverse=True)
        tv = self.total_variation(current, reference)
        js = self.js_divergence(current, reference)

        # Combine: use JS as primary (bounded), with TV as secondary
        raw = 0.4 * js + 0.3 * tv + 0.15 * min(1.0, kl_fwd) + 0.15 * min(1.0, kl_rev)
        normalized = self.normalize(raw)

        return SignalResult(
            signal_name=self.name,
            raw_score=raw,
            normalized_score=normalized,
            interpretation=self.interpret(normalized),
            components={
                "kl_divergence_forward": kl_fwd,
                "kl_divergence_reverse": kl_rev,
                "total_variation": tv,
                "js_divergence": js,
            },
        )

    def normalize(self, raw: float) -> float:
        """Normalize: raw / 0.5, capped at 1.0."""
        return min(1.0, raw / 0.5)
