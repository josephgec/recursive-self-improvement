"""ConsistencyConstraint: equivalent inputs must produce equivalent outputs."""

from __future__ import annotations

from typing import Any, List

from src.constraints.base import Constraint, ConstraintResult, CheckContext


class ConsistencyConstraint(Constraint):
    """Equivalent inputs must produce equivalent outputs at >= threshold rate."""

    def __init__(self, threshold: float = 0.85) -> None:
        super().__init__(
            name="consistency",
            description="Equivalent inputs must produce equivalent outputs",
            category="quality",
            threshold=threshold,
        )

    def check(self, agent_state: Any, context: CheckContext) -> ConstraintResult:
        """Evaluate consistency.

        ``agent_state`` must expose:
        * ``evaluate_consistency(pairs) -> List[dict]``
          where each dict has ``equivalent: bool``.
        """
        pairs = self._generate_equivalence_pairs()
        results = agent_state.evaluate_consistency(pairs)

        total = len(results)
        equivalent = sum(1 for r in results if r["equivalent"])
        rate = equivalent / total if total else 0.0

        headroom = self.headroom(rate)
        return ConstraintResult(
            satisfied=rate >= self._threshold,
            measured_value=rate,
            threshold=self._threshold,
            headroom=headroom,
            details={
                "total_pairs": total,
                "equivalent_pairs": equivalent,
                "consistency_rate": rate,
            },
        )

    @staticmethod
    def _generate_equivalence_pairs() -> List[dict]:
        """Generate pairs of semantically equivalent inputs."""
        return [
            {
                "input_a": "What is the capital of France?",
                "input_b": "Name the capital city of France.",
            },
            {
                "input_a": "Explain photosynthesis.",
                "input_b": "Describe how photosynthesis works.",
            },
            {
                "input_a": "What is 2 + 2?",
                "input_b": "Calculate the sum of 2 and 2.",
            },
            {
                "input_a": "Who wrote Romeo and Juliet?",
                "input_b": "Name the author of Romeo and Juliet.",
            },
            {
                "input_a": "What is the speed of light?",
                "input_b": "How fast does light travel?",
            },
            {
                "input_a": "Define machine learning.",
                "input_b": "What is machine learning?",
            },
            {
                "input_a": "List the planets in our solar system.",
                "input_b": "Name all planets in the solar system.",
            },
            {
                "input_a": "What causes rain?",
                "input_b": "Explain why it rains.",
            },
            {
                "input_a": "What is DNA?",
                "input_b": "Describe DNA and its function.",
            },
            {
                "input_a": "How does gravity work?",
                "input_b": "Explain the mechanism of gravity.",
            },
        ]
