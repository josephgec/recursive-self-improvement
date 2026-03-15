"""Tail-distribution analysis for model-collapse detection.

Tracks how the probability mass in the "tail" of the token distribution
changes across generations.  Model collapse compresses the tail --
rare tokens become rarer -- which this module quantifies via tail mass
fractions, the Gini coefficient, and rank correlation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class TailResult:
    """Results from tail-distribution analysis."""

    tail_mass_p01: float  # mass in the bottom 1% of tokens
    tail_mass_p05: float  # mass in the bottom 5% of tokens
    tail_mass_p10: float  # mass in the bottom 10% of tokens
    gini_coefficient: float  # Gini coefficient of the distribution
    top_10_mass: float  # mass in the top 10 tokens
    rank_correlation: float  # Spearman rho between P and Q ranks


class TailAnalyzer:
    """Analyse the tail behaviour of a token distribution.

    Compares a model's (possibly collapsed) distribution Q against the
    reference distribution P to detect tail compression -- one of the
    earliest and most reliable signals of model collapse.

    Args:
        reference_distribution: 1-D array of shape ``(vocab_size,)``
            summing to 1.  This is P.
        tokenizer: Tokenizer (used only for vocab-size validation).
    """

    def __init__(self, reference_distribution: np.ndarray, tokenizer) -> None:
        self._reference = np.array(reference_distribution, dtype=np.float64)
        self._tokenizer = tokenizer
        self._vocab_size = len(self._reference)

        # Pre-compute reference rank order (ascending probability).
        self._ref_rank_order = np.argsort(self._reference)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure(self, model_distribution: np.ndarray) -> TailResult:
        """Analyse the tail of *model_distribution* relative to reference.

        Args:
            model_distribution: 1-D array of shape ``(vocab_size,)``
                summing to 1.  This is Q.

        Returns:
            A ``TailResult`` with tail masses, Gini, top-10 mass, and
            rank correlation.
        """
        q = np.array(model_distribution, dtype=np.float64)

        # Ensure proper length -- truncate or pad if needed.
        if len(q) < self._vocab_size:
            padded = np.zeros(self._vocab_size, dtype=np.float64)
            padded[: len(q)] = q
            q = padded
        elif len(q) > self._vocab_size:
            q = q[: self._vocab_size]

        # Normalize.
        total = q.sum()
        if total > 0:
            q = q / total
        else:
            q = np.ones(self._vocab_size, dtype=np.float64) / self._vocab_size

        # Sort Q in ascending order of *reference* rank to measure
        # how much mass Q puts on the rarest tokens according to P.
        q_sorted_by_ref_rank = q[self._ref_rank_order]

        tail_01 = self._tail_mass(q_sorted_by_ref_rank, fraction=0.01)
        tail_05 = self._tail_mass(q_sorted_by_ref_rank, fraction=0.05)
        tail_10 = self._tail_mass(q_sorted_by_ref_rank, fraction=0.10)

        gini = self._gini_coefficient(q)

        # Top-10 mass: sum of 10 highest-probability tokens.
        top_10_mass = float(np.sort(q)[-10:].sum()) if len(q) >= 10 else float(q.sum())

        # Spearman rank correlation between P and Q orderings.
        rank_corr = self._rank_correlation(self._reference, q)

        return TailResult(
            tail_mass_p01=tail_01,
            tail_mass_p05=tail_05,
            tail_mass_p10=tail_10,
            gini_coefficient=gini,
            top_10_mass=top_10_mass,
            rank_correlation=rank_corr,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _tail_mass(sorted_by_ref_rank: np.ndarray, fraction: float) -> float:
        """Sum of Q's mass on the bottom *fraction* of tokens (by P rank).

        The array must be sorted by ascending reference probability, so
        ``sorted_by_ref_rank[:k]`` corresponds to the rarest k tokens
        according to P.
        """
        k = max(int(len(sorted_by_ref_rank) * fraction), 1)
        return float(sorted_by_ref_rank[:k].sum())

    @staticmethod
    def _gini_coefficient(distribution: np.ndarray) -> float:
        """Compute the Gini coefficient of a probability distribution.

        Gini = 0 means perfectly uniform; Gini -> 1 means all mass on
        one token.  Calculated via the "relative mean absolute
        difference" formula.
        """
        sorted_d = np.sort(distribution)
        n = len(sorted_d)
        if n == 0 or sorted_d.sum() == 0:
            return 0.0

        # Gini = (2 * sum_i (i+1) * y_i) / (n * sum(y)) - (n+1)/n
        index = np.arange(1, n + 1, dtype=np.float64)
        return float(
            (2.0 * np.sum(index * sorted_d)) / (n * np.sum(sorted_d))
            - (n + 1.0) / n
        )

    @staticmethod
    def _rank_correlation(p: np.ndarray, q: np.ndarray) -> float:
        """Spearman rank correlation between two distributions.

        Returns a value in [-1, 1].  A high positive value means the
        rank orders are well-preserved; a low value indicates the model
        has reshuffled which tokens are common/rare.
        """
        min_len = min(len(p), len(q))
        rho, _ = stats.spearmanr(p[:min_len], q[:min_len])
        return float(rho)
