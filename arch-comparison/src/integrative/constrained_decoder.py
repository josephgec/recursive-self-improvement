"""Constrained decoder: applies logical/arithmetic constraints during generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Constraint:
    """A constraint to apply during decoding."""
    constraint_type: str  # "arithmetic", "logical", "type"
    pattern: str  # regex or rule description
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ConstrainedOutput:
    """Output from constrained decoding."""
    text: str
    constraints_applied: List[Constraint] = field(default_factory=list)
    constraint_violations: int = 0
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


class ConstrainedDecoder:
    """Applies constraints during text generation.

    In a real system, constraints mask invalid tokens during beam search.
    Here we simulate this with post-hoc correction and detection.
    """

    def __init__(
        self,
        model: Optional[Callable] = None,
        constraint_weight: float = 0.3,
    ) -> None:
        self.model = model or self._default_mock_model
        self.constraint_weight = constraint_weight
        self._constraints: List[Constraint] = []

    def add_constraint(self, constraint: Constraint) -> None:
        """Register a constraint."""
        self._constraints.append(constraint)

    def generate(
        self,
        prompt: str,
        constraints: Optional[List[Constraint]] = None,
    ) -> ConstrainedOutput:
        """Generate text with constraints applied.

        Args:
            prompt: Input prompt.
            constraints: Optional additional constraints for this call.

        Returns:
            ConstrainedOutput with corrected text and metadata.
        """
        active = list(self._constraints)
        if constraints:
            active.extend(constraints)

        # Generate base text from model
        base_text = self.model(prompt)

        # Detect computation context and apply constraints
        applied: List[Constraint] = []
        violations = 0
        corrected = base_text

        # Auto-detect arithmetic context
        if self._detect_computation_context(prompt):
            arith_constraint = Constraint(
                constraint_type="arithmetic",
                pattern=r"\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+",
            )
            if arith_constraint not in active:
                active.append(arith_constraint)

        for constraint in active:
            if constraint.constraint_type == "arithmetic":
                corrected, was_applied, n_violations = self._apply_arithmetic_constraint(
                    corrected, constraint
                )
                if was_applied:
                    applied.append(constraint)
                violations += n_violations
            elif constraint.constraint_type == "logical":
                corrected, was_applied, n_violations = self._apply_logical_constraint(
                    corrected, constraint
                )
                if was_applied:
                    applied.append(constraint)
                violations += n_violations

        confidence = max(0.0, 1.0 - violations * 0.1 * self.constraint_weight)

        return ConstrainedOutput(
            text=corrected,
            constraints_applied=applied,
            constraint_violations=violations,
            confidence=confidence,
        )

    def _detect_computation_context(self, text: str) -> bool:
        """Detect if the text involves computation.

        Looks for arithmetic expressions, keywords like 'calculate',
        'compute', 'solve', or number patterns.
        """
        text_lower = text.lower()
        compute_words = ["calculate", "compute", "solve", "evaluate", "what is", "find"]
        if any(w in text_lower for w in compute_words):
            return True
        # Check for arithmetic patterns
        if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", text):
            return True
        return False

    def _apply_arithmetic_constraint(
        self, text: str, constraint: Constraint
    ) -> tuple:
        """Apply arithmetic constraints: detect 'X op Y = Z' and correct Z.

        Returns (corrected_text, was_applied, num_violations).
        """
        was_applied = False
        violations = 0

        def fix_arithmetic(match: re.Match) -> str:
            nonlocal was_applied, violations
            a = int(match.group(1))
            op = match.group(2)
            b = int(match.group(3))
            claimed = int(match.group(4))

            ops = {"+": a + b, "-": a - b, "*": a * b}
            if op == "/" and b != 0:
                ops["/"] = a // b

            correct = ops.get(op)
            if correct is not None:
                was_applied = True
                if claimed != correct:
                    violations += 1
                    return f"{a} {op} {b} = {correct}"
            return match.group(0)

        corrected = re.sub(
            r"(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=\s*(\d+)",
            fix_arithmetic,
            text,
        )
        return corrected, was_applied, violations

    def _apply_logical_constraint(
        self, text: str, constraint: Constraint
    ) -> tuple:
        """Apply logical consistency constraints.

        Returns (corrected_text, was_applied, num_violations).
        """
        was_applied = False
        violations = 0

        # Detect contradictions like "X is true" and "X is false"
        true_claims = set(re.findall(r"(\w+)\s+is\s+true", text, re.IGNORECASE))
        false_claims = set(re.findall(r"(\w+)\s+is\s+false", text, re.IGNORECASE))
        contradictions = true_claims & false_claims
        if contradictions:
            was_applied = True
            violations += len(contradictions)

        return text, was_applied, violations

    @staticmethod
    def _default_mock_model(prompt: str) -> str:
        """Default mock model for constrained generation."""
        prompt_lower = prompt.lower()

        # Arithmetic
        arith = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", prompt)
        if arith:
            a, op, b = int(arith.group(1)), arith.group(2), int(arith.group(3))
            ops = {"+": a + b, "-": a - b, "*": a * b}
            if op == "/" and b != 0:
                ops["/"] = a // b
            result = ops.get(op, 0)
            return f"Computing {a} {op} {b} = {result}. The answer is {result}."

        # Logic
        if "logic" in prompt_lower or "implies" in prompt_lower:
            return "Applying logical reasoning. The conclusion follows from the premises. The answer is true."

        # Default
        return "Reasoning directly from the input. The answer is unknown."
