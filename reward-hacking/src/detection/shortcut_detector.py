from __future__ import annotations

"""Detect shortcut learning patterns in model outputs."""

import numpy as np
from dataclasses import dataclass


@dataclass
class ShortcutReport:
    """Report from shortcut detection analysis."""

    length_gaming: bool
    repetition_gaming: bool
    diversity_collapse: bool
    any_detected: bool
    details: dict


class ShortcutDetector:
    """Detects common shortcut learning patterns.

    Checks for:
    - Length gaming: outputs becoming excessively long
    - Repetition: high token/phrase repetition rates
    - Diversity collapse: outputs converging to few patterns
    """

    def __init__(
        self,
        length_ratio_threshold: float = 2.0,
        repetition_threshold: float = 0.5,
        diversity_min: float = 0.3,
    ):
        self._length_ratio = length_ratio_threshold
        self._repetition_threshold = repetition_threshold
        self._diversity_min = diversity_min

    def check_length_gaming(
        self,
        output_lengths: list[int],
        baseline_lengths: list[int],
    ) -> tuple[bool, dict]:
        """Check if outputs are gaming via excessive length.

        Args:
            output_lengths: Current output lengths.
            baseline_lengths: Baseline (pre-RL) output lengths.

        Returns:
            Tuple of (is_gaming, details).
        """
        if not output_lengths or not baseline_lengths:
            return False, {"reason": "empty_data"}

        current_mean = float(np.mean(output_lengths))
        baseline_mean = float(np.mean(baseline_lengths))

        if baseline_mean <= 0:
            return False, {"reason": "zero_baseline"}

        ratio = current_mean / baseline_mean
        is_gaming = ratio > self._length_ratio

        return is_gaming, {
            "current_mean_length": current_mean,
            "baseline_mean_length": baseline_mean,
            "length_ratio": ratio,
            "threshold": self._length_ratio,
        }

    def check_repetition(
        self,
        outputs: list[list[int]],
    ) -> tuple[bool, dict]:
        """Check for excessive token repetition in outputs.

        Args:
            outputs: List of token sequences (list of lists of ints).

        Returns:
            Tuple of (is_repeating, details).
        """
        if not outputs:
            return False, {"reason": "empty_data"}

        repetition_rates = []
        for seq in outputs:
            if len(seq) < 2:
                continue
            # Count adjacent repeats
            repeats = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1])
            rate = repeats / (len(seq) - 1)
            repetition_rates.append(rate)

        if not repetition_rates:
            return False, {"reason": "no_valid_sequences"}

        mean_rate = float(np.mean(repetition_rates))
        max_rate = float(np.max(repetition_rates))
        is_repeating = mean_rate > self._repetition_threshold

        return is_repeating, {
            "mean_repetition_rate": mean_rate,
            "max_repetition_rate": max_rate,
            "threshold": self._repetition_threshold,
            "num_sequences": len(repetition_rates),
        }

    def check_diversity_collapse(
        self,
        outputs: list[list[int]],
        vocab_size: int = 100,
    ) -> tuple[bool, dict]:
        """Check if output diversity has collapsed.

        Args:
            outputs: List of token sequences.
            vocab_size: Size of token vocabulary.

        Returns:
            Tuple of (is_collapsed, details).
        """
        if not outputs:
            return False, {"reason": "empty_data"}

        # Count unique tokens used across all outputs
        all_tokens = set()
        for seq in outputs:
            all_tokens.update(seq)

        diversity = len(all_tokens) / max(vocab_size, 1)
        is_collapsed = diversity < self._diversity_min

        # Also check output similarity (Jaccard between consecutive pairs)
        similarities = []
        for i in range(1, len(outputs)):
            set_a = set(outputs[i - 1])
            set_b = set(outputs[i])
            if set_a or set_b:
                jaccard = len(set_a & set_b) / len(set_a | set_b)
                similarities.append(jaccard)

        mean_similarity = float(np.mean(similarities)) if similarities else 0.0

        return is_collapsed, {
            "token_diversity": diversity,
            "unique_tokens": len(all_tokens),
            "vocab_size": vocab_size,
            "diversity_threshold": self._diversity_min,
            "mean_pairwise_similarity": mean_similarity,
        }

    def run_all(
        self,
        output_lengths: list[int],
        baseline_lengths: list[int],
        outputs: list[list[int]],
        vocab_size: int = 100,
    ) -> ShortcutReport:
        """Run all shortcut detection checks.

        Args:
            output_lengths: Current output lengths.
            baseline_lengths: Baseline output lengths.
            outputs: Token sequences for repetition/diversity checks.
            vocab_size: Vocabulary size.

        Returns:
            ShortcutReport with all detection results.
        """
        length_gaming, length_details = self.check_length_gaming(
            output_lengths, baseline_lengths
        )
        repetition, rep_details = self.check_repetition(outputs)
        diversity_collapse, div_details = self.check_diversity_collapse(
            outputs, vocab_size
        )

        return ShortcutReport(
            length_gaming=length_gaming,
            repetition_gaming=repetition,
            diversity_collapse=diversity_collapse,
            any_detected=length_gaming or repetition or diversity_collapse,
            details={
                "length": length_details,
                "repetition": rep_details,
                "diversity": div_details,
            },
        )
