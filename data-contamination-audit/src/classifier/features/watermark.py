"""Watermark detection based on Kirchenbauer et al. (2023).

Implements the "green list" statistical test for detecting whether a
sequence of tokens was generated with a watermarking scheme that biases
generation toward a pseudorandom subset of the vocabulary (the green
list) determined by the preceding token.

Reference
---------
Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., & Goldstein, T.
(2023).  A Watermark for Large Language Models.  ICML 2023.
"""

from __future__ import annotations

import math
from typing import Sequence


class WatermarkDetector:
    """Detect Kirchenbauer-style green-list watermarks in token sequences.

    Parameters
    ----------
    vocab_size:
        Size of the token vocabulary (GPT-2 default: 50257).
    gamma:
        Fraction of the vocabulary in the green list at each position.
    hash_key:
        Secret key used in the pseudorandom green list generation.
        (For detection we test a candidate key; if the true key is unknown,
        a sweep over candidate keys is needed.)
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        gamma: float = 0.25,
        hash_key: int = 15485863,
    ) -> None:
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.hash_key = hash_key

    # ------------------------------------------------------------------
    # Green-list generation
    # ------------------------------------------------------------------

    def _green_list(self, prev_token_id: int) -> set[int]:
        """Generate the green-list token set seeded by *prev_token_id*.

        Uses a simple hash of ``(hash_key, prev_token_id)`` to seed a
        deterministic PRNG and selects the first ``gamma * vocab_size``
        tokens from a pseudorandom permutation.
        """
        import random as _random

        seed = self.hash_key * (prev_token_id + 1)
        rng = _random.Random(seed)

        # Generate a pseudorandom permutation of [0, vocab_size).
        indices = list(range(self.vocab_size))
        rng.shuffle(indices)

        green_size = int(self.gamma * self.vocab_size)
        return set(indices[:green_size])

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, token_ids: Sequence[int]) -> dict[str, float]:
        """Compute watermark detection statistics for a token sequence.

        For each token at position *t >= 1*, the previous token
        ``token_ids[t-1]`` seeds the green list.  The token
        ``token_ids[t]`` is checked for membership.

        Parameters
        ----------
        token_ids:
            A list of integer token IDs (e.g., from a GPT-2 tokenizer).

        Returns
        -------
        dict with:
            - ``watermark_z_score``: z-statistic under the null hypothesis
              that tokens are drawn uniformly at random w.r.t. the green
              list.  Values above ~4 strongly suggest watermarking.
            - ``watermark_green_fraction``: observed fraction of tokens
              that fell in their respective green lists.
        """
        if len(token_ids) < 2:
            return {
                "watermark_z_score": 0.0,
                "watermark_green_fraction": 0.0,
            }

        n_green = 0
        total = 0

        for t in range(1, len(token_ids)):
            green = self._green_list(token_ids[t - 1])
            if token_ids[t] in green:
                n_green += 1
            total += 1

        green_fraction = n_green / total if total > 0 else 0.0

        # z-score under H0: each token is green with probability gamma.
        # z = (n_green - gamma * T) / sqrt(T * gamma * (1 - gamma))
        expected = self.gamma * total
        std = math.sqrt(total * self.gamma * (1.0 - self.gamma))
        z_score = (n_green - expected) / std if std > 0 else 0.0

        return {
            "watermark_z_score": z_score,
            "watermark_green_fraction": green_fraction,
        }
