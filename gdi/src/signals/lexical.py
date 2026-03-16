"""Lexical drift signal using distribution divergence measures."""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from .base import DriftSignal, SignalResult


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _word_distribution(texts: List[str]) -> Dict[str, float]:
    """Build a normalized word probability distribution."""
    counter: Counter = Counter()
    for text in texts:
        counter.update(_tokenize(text))
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


class LexicalDriftSignal(DriftSignal):
    """Lexical drift signal using word distribution divergence.

    Measures drift in vocabulary usage, word distributions, and n-gram patterns.
    """

    @property
    def name(self) -> str:
        return "lexical"

    def js_divergence(
        self, current: List[str], reference: List[str]
    ) -> float:
        """Jensen-Shannon divergence between word distributions.

        Returns value in [0, 1] (using log base 2).
        """
        p = _word_distribution(current)
        q = _word_distribution(reference)

        if not p and not q:
            return 0.0
        if not p or not q:
            return 1.0

        all_keys = set(p.keys()) | set(q.keys())

        # Compute midpoint distribution
        m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in all_keys}

        def _kl(dist_a: Dict[str, float], dist_b: Dict[str, float]) -> float:
            result = 0.0
            for k in dist_a:
                if dist_a[k] > 0 and dist_b.get(k, 0) > 0:
                    result += dist_a[k] * math.log2(dist_a[k] / dist_b[k])
            return result

        js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
        return min(1.0, max(0.0, js))

    def vocabulary_shift(
        self, current: List[str], reference: List[str]
    ) -> float:
        """Jaccard distance between vocabularies.

        Returns 1 - |intersection| / |union|.
        """
        vocab_cur = set()
        for text in current:
            vocab_cur.update(_tokenize(text))

        vocab_ref = set()
        for text in reference:
            vocab_ref.update(_tokenize(text))

        if not vocab_cur and not vocab_ref:
            return 0.0
        if not vocab_cur or not vocab_ref:
            return 1.0

        intersection = vocab_cur & vocab_ref
        union = vocab_cur | vocab_ref

        return 1.0 - len(intersection) / len(union)

    def ngram_novelty(
        self, current: List[str], reference: List[str], n: int = 2
    ) -> float:
        """Fraction of current n-grams not seen in reference.

        Returns value in [0, 1].
        """
        ref_ngrams = set()
        for text in reference:
            tokens = _tokenize(text)
            ref_ngrams.update(_get_ngrams(tokens, n))

        if not ref_ngrams:
            return 1.0

        cur_ngrams = []
        for text in current:
            tokens = _tokenize(text)
            cur_ngrams.extend(_get_ngrams(tokens, n))

        if not cur_ngrams:
            return 0.0

        novel = sum(1 for ng in cur_ngrams if ng not in ref_ngrams)
        return novel / len(cur_ngrams)

    def compute(
        self, current: List[str], reference: List[str]
    ) -> SignalResult:
        """Compute lexical drift as weighted combination.

        Composite: 0.5*JS + 0.25*vocab_shift + 0.25*novelty
        """
        js = self.js_divergence(current, reference)
        vocab = self.vocabulary_shift(current, reference)
        novelty = self.ngram_novelty(current, reference)

        raw = 0.5 * js + 0.25 * vocab + 0.25 * novelty
        normalized = self.normalize(raw)

        return SignalResult(
            signal_name=self.name,
            raw_score=raw,
            normalized_score=normalized,
            interpretation=self.interpret(normalized),
            components={
                "js_divergence": js,
                "vocabulary_shift": vocab,
                "ngram_novelty": novelty,
            },
        )

    def normalize(self, raw: float) -> float:
        """Normalize: raw / 0.5, capped at 1.0."""
        return min(1.0, raw / 0.5)
