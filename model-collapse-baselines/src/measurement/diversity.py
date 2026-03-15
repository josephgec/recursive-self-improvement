"""Surface-level diversity metrics for generated text.

Measures lexical diversity (distinct n-grams, type-token ratio),
self-similarity (self-BLEU), and vocabulary coverage to detect the
homogenisation that characterises model collapse.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class DiversityResult:
    """Results from diversity measurement."""

    distinct_1: float
    distinct_2: float
    distinct_3: float
    distinct_4: float
    type_token_ratio: float
    self_bleu: float
    vocabulary_usage: int  # number of unique tokens used
    hapax_legomena_ratio: float  # fraction of words appearing exactly once


class DiversityMeasurer:
    """Measure lexical diversity of a collection of generated texts.

    All metrics operate on simple whitespace-tokenized, lower-cased text
    so they are model-agnostic and fast to compute.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure(self, texts: list[str]) -> DiversityResult:
        """Compute all diversity metrics for *texts*.

        Args:
            texts: List of generated text strings.

        Returns:
            A ``DiversityResult`` with distinct-n, TTR, self-BLEU,
            vocabulary usage, and hapax legomena ratio.
        """
        # Tokenize all texts.
        tokenized = [text.lower().split() for text in texts]
        all_tokens: list[str] = []
        for tokens in tokenized:
            all_tokens.extend(tokens)

        if not all_tokens:
            return DiversityResult(
                distinct_1=0.0,
                distinct_2=0.0,
                distinct_3=0.0,
                distinct_4=0.0,
                type_token_ratio=0.0,
                self_bleu=0.0,
                vocabulary_usage=0,
                hapax_legomena_ratio=0.0,
            )

        distinct_1 = self._distinct_n(all_tokens, n=1)
        distinct_2 = self._distinct_n(all_tokens, n=2)
        distinct_3 = self._distinct_n(all_tokens, n=3)
        distinct_4 = self._distinct_n(all_tokens, n=4)

        ttr = self._type_token_ratio(all_tokens)
        self_bleu = self._self_bleu(tokenized, max_pairs=500)
        vocab_usage = len(set(all_tokens))
        hapax = self._hapax_legomena_ratio(all_tokens)

        return DiversityResult(
            distinct_1=distinct_1,
            distinct_2=distinct_2,
            distinct_3=distinct_3,
            distinct_4=distinct_4,
            type_token_ratio=ttr,
            self_bleu=self_bleu,
            vocabulary_usage=vocab_usage,
            hapax_legomena_ratio=hapax,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _distinct_n(tokens: list[str], n: int) -> float:
        """Fraction of unique n-grams out of all n-grams.

        distinct-n = |unique n-grams| / |total n-grams|

        A higher value indicates more diverse text.
        """
        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0

        return len(set(ngrams)) / len(ngrams)

    @staticmethod
    def _type_token_ratio(tokens: list[str]) -> float:
        """Ratio of unique words to total words."""
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    @staticmethod
    def _self_bleu(
        tokenized_texts: list[list[str]], max_pairs: int = 500
    ) -> float:
        """Average pairwise BLEU score (self-BLEU).

        Lower self-BLEU means more diverse outputs.  We sample up to
        ``max_pairs`` random pairs to keep computation tractable.

        Uses ``nltk.translate.bleu_score.sentence_bleu`` with smoothing
        to avoid zero scores on short texts.
        """
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        n = len(tokenized_texts)
        if n < 2:
            return 0.0

        # Filter out empty texts.
        non_empty = [t for t in tokenized_texts if len(t) > 0]
        n = len(non_empty)
        if n < 2:
            return 0.0

        rng = np.random.RandomState(42)
        num_pairs = min(max_pairs, n * (n - 1) // 2)

        scores: list[float] = []
        smoother = SmoothingFunction().method1

        # Generate unique random pairs.
        pairs_seen: set[tuple[int, int]] = set()
        attempts = 0
        max_attempts = num_pairs * 10

        while len(scores) < num_pairs and attempts < max_attempts:
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i == j:
                attempts += 1
                continue
            pair = (min(i, j), max(i, j))
            if pair in pairs_seen:
                attempts += 1
                continue
            pairs_seen.add(pair)

            # BLEU: hypothesis = text[i], reference = [text[j]]
            score = sentence_bleu(
                [non_empty[j]],
                non_empty[i],
                smoothing_function=smoother,
            )
            scores.append(score)
            attempts += 1

        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _hapax_legomena_ratio(tokens: list[str]) -> float:
        """Fraction of words that appear exactly once."""
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        hapax_count = sum(1 for c in counts.values() if c == 1)
        return hapax_count / len(counts)
