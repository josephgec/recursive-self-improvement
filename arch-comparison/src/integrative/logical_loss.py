"""Logical loss: penalizes outputs that violate logical/arithmetic rules."""

from __future__ import annotations

import re
from typing import Optional


class LogicalLoss:
    """Computes a loss score based on logical and arithmetic correctness.

    The loss is 0 when the output is fully correct, and increases
    with each detected violation.
    """

    def __init__(self, arithmetic_weight: float = 0.5, consistency_weight: float = 0.5) -> None:
        self.arithmetic_weight = arithmetic_weight
        self.consistency_weight = consistency_weight

    def compute(self, predicted_text: str, context: Optional[str] = None) -> float:
        """Compute the logical loss for a predicted text.

        Args:
            predicted_text: The model's output text.
            context: Optional context (e.g., the input problem).

        Returns:
            A float loss value >= 0. Lower is better.
        """
        arith_loss = self._arithmetic_loss(predicted_text)
        consist_loss = self._consistency_loss(predicted_text)
        total = (
            self.arithmetic_weight * arith_loss
            + self.consistency_weight * consist_loss
        )
        return total

    def _arithmetic_loss(self, text: str) -> float:
        """Check for incorrect arithmetic in 'a op b = c' patterns.

        Returns loss proportional to number of wrong equations found.
        """
        pattern = re.findall(r"(\d+)\s*([\+\-\*])\s*(\d+)\s*=\s*(\d+)", text)
        if not pattern:
            return 0.0

        wrong = 0
        total = len(pattern)
        for a_str, op, b_str, c_str in pattern:
            a, b, c = int(a_str), int(b_str), int(c_str)
            expected = {"+": a + b, "-": a - b, "*": a * b}.get(op)
            if expected is not None and expected != c:
                wrong += 1

        return wrong / total if total > 0 else 0.0

    def _consistency_loss(self, text: str) -> float:
        """Check for logical contradictions in the text.

        Detects patterns like "X is true" and "X is false" for the same X,
        or "X = a" and "X = b" for a != b.
        """
        violations = 0.0

        # Check truth contradictions
        true_claims = set(re.findall(r"(\w+)\s+is\s+true", text, re.IGNORECASE))
        false_claims = set(re.findall(r"(\w+)\s+is\s+false", text, re.IGNORECASE))
        violations += len(true_claims & false_claims)

        # Check variable assignment contradictions
        assignments = re.findall(r"(\w+)\s*=\s*(\d+)", text)
        var_values: dict = {}
        for var, val in assignments:
            if var in var_values and var_values[var] != val:
                violations += 1
            var_values[var] = val

        # Normalise to [0, 1]
        return min(violations / 3.0, 1.0) if violations > 0 else 0.0
