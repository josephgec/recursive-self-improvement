"""Deduplicator for removing near-duplicate training pairs."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from src.synthesis.synthesizer import TrainingPair


class Deduplicator:
    """Remove near-duplicate training pairs based on text similarity.

    Uses character n-gram Jaccard similarity for efficiency.
    """

    def __init__(self, similarity_threshold: float = 0.85, ngram_size: int = 3):
        self.similarity_threshold = similarity_threshold
        self.ngram_size = ngram_size
        self._removed_count: int = 0

    @property
    def removed_count(self) -> int:
        return self._removed_count

    def deduplicate(self, pairs: List[TrainingPair]) -> List[TrainingPair]:
        """Remove near-duplicate pairs, keeping the highest quality version."""
        if not pairs:
            return []

        # Sort by quality score descending so we keep the best
        sorted_pairs = sorted(pairs, key=lambda p: -p.quality_score)

        kept: List[TrainingPair] = []
        kept_signatures: List[Set[str]] = []

        for pair in sorted_pairs:
            sig = self._compute_ngrams(pair.completion)
            is_dup = False
            for existing_sig in kept_signatures:
                sim = self._jaccard_similarity(sig, existing_sig)
                if sim >= self.similarity_threshold:
                    is_dup = True
                    break

            if not is_dup:
                kept.append(pair)
                kept_signatures.append(sig)

        self._removed_count = len(pairs) - len(kept)
        return kept

    def _compute_ngrams(self, text: str) -> Set[str]:
        """Compute character n-grams from text."""
        text = text.lower().strip()
        if len(text) < self.ngram_size:
            return {text}
        return {text[i : i + self.ngram_size] for i in range(len(text) - self.ngram_size + 1)}

    @staticmethod
    def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def summary(self) -> Dict[str, Any]:
        return {"removed_count": self._removed_count}
