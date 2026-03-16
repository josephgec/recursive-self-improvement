"""Quality filter for training pairs: validity, length, and alignment checks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.synthesis.synthesizer import TrainingPair


class QualityFilter:
    """Filter training pairs based on quality criteria.

    Checks:
    - Token count bounds (min/max for prompt and completion)
    - Minimum quality score
    - Non-empty content
    - Prompt-completion alignment (basic similarity check)
    """

    def __init__(
        self,
        min_prompt_tokens: int = 10,
        max_prompt_tokens: int = 4096,
        min_completion_tokens: int = 10,
        max_completion_tokens: int = 4096,
        min_quality_score: float = 0.3,
    ):
        self.min_prompt_tokens = min_prompt_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.min_completion_tokens = min_completion_tokens
        self.max_completion_tokens = max_completion_tokens
        self.min_quality_score = min_quality_score
        self._rejected: List[Dict[str, Any]] = []

    @property
    def rejected(self) -> List[Dict[str, Any]]:
        return list(self._rejected)

    def filter(self, pairs: List[TrainingPair]) -> List[TrainingPair]:
        """Filter a list of training pairs, returning only those that pass."""
        self._rejected = []
        accepted = []
        for pair in pairs:
            reasons = self.check(pair)
            if not reasons:
                accepted.append(pair)
            else:
                self._rejected.append({
                    "pair_id": pair.pair_id,
                    "strategy": pair.strategy,
                    "reasons": reasons,
                })
        return accepted

    def check(self, pair: TrainingPair) -> List[str]:
        """Check a single pair and return list of rejection reasons (empty if OK)."""
        reasons = []

        # Non-empty checks
        if not pair.prompt.strip():
            reasons.append("empty_prompt")
        if not pair.completion.strip():
            reasons.append("empty_completion")

        # Token count checks
        if pair.prompt_tokens < self.min_prompt_tokens:
            reasons.append(f"prompt_too_short ({pair.prompt_tokens} < {self.min_prompt_tokens})")
        if pair.prompt_tokens > self.max_prompt_tokens:
            reasons.append(f"prompt_too_long ({pair.prompt_tokens} > {self.max_prompt_tokens})")
        if pair.completion_tokens < self.min_completion_tokens:
            reasons.append(f"completion_too_short ({pair.completion_tokens} < {self.min_completion_tokens})")
        if pair.completion_tokens > self.max_completion_tokens:
            reasons.append(f"completion_too_long ({pair.completion_tokens} > {self.max_completion_tokens})")

        # Quality score check
        if pair.quality_score < self.min_quality_score:
            reasons.append(f"low_quality ({pair.quality_score:.3f} < {self.min_quality_score})")

        # Alignment check: completion should not be identical to prompt
        if pair.prompt.strip() == pair.completion.strip():
            reasons.append("prompt_completion_identical")

        return reasons

    def summary(self) -> Dict[str, Any]:
        """Return summary of filtering results."""
        reason_counts: Dict[str, int] = {}
        for rej in self._rejected:
            for reason in rej["reasons"]:
                # Normalize reason to category
                cat = reason.split(" (")[0]
                reason_counts[cat] = reason_counts.get(cat, 0) + 1
        return {
            "rejected_count": len(self._rejected),
            "rejection_reasons": reason_counts,
        }
