"""Answer checking with cascading equivalence strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.math_equivalence import (
    fraction_equal,
    normalize_answer,
    numeric_equal,
    set_equal,
    symbolic_equal,
)
from src.utils.latex_parser import extract_boxed, strip_latex_formatting
from src.utils.logging import get_logger

logger = get_logger("answer_checker")


@dataclass
class CheckResult:
    """Detailed result of an answer check."""

    correct: bool
    method: str  # which check method succeeded
    model_answer: str  # normalized model answer
    expected_answer: str  # normalized expected answer
    explanation: str = ""


class AnswerChecker:
    """Check if a model answer matches an expected answer.

    Uses a cascade of comparison methods:
    1. Exact string match (after normalization)
    2. Numeric equality (within tolerance)
    3. Fraction equality
    4. Symbolic equality (SymPy simplify)
    5. Set/list equality (order-independent)
    """

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def check(self, model: str, expected: str) -> bool:
        """Check if model answer matches expected answer.

        Returns True if any comparison method confirms equality.
        """
        return self.check_detailed(model, expected).correct

    def check_detailed(self, model: str, expected: str) -> CheckResult:
        """Check with detailed explanation of which method matched."""
        # Normalize both answers
        m = self._normalize(model)
        e = self._normalize(expected)

        # 1. Exact string match
        if m == e:
            return CheckResult(
                correct=True,
                method="exact_string",
                model_answer=m,
                expected_answer=e,
                explanation="Exact string match after normalization.",
            )

        # 2. Numeric equality
        if numeric_equal(m, e, self.tolerance):
            return CheckResult(
                correct=True,
                method="numeric",
                model_answer=m,
                expected_answer=e,
                explanation=f"Numerically equal within tolerance {self.tolerance}.",
            )

        # 3. Fraction equality
        if fraction_equal(m, e):
            return CheckResult(
                correct=True,
                method="fraction",
                model_answer=m,
                expected_answer=e,
                explanation="Equal as fractions.",
            )

        # 4. Symbolic equality
        if symbolic_equal(m, e):
            return CheckResult(
                correct=True,
                method="symbolic",
                model_answer=m,
                expected_answer=e,
                explanation="Symbolically equal (SymPy simplify).",
            )

        # 5. Set/list equality
        if set_equal(m, e):
            return CheckResult(
                correct=True,
                method="set_match",
                model_answer=m,
                expected_answer=e,
                explanation="Equal as sets (order-independent).",
            )

        # Not equal
        return CheckResult(
            correct=False,
            method="none",
            model_answer=m,
            expected_answer=e,
            explanation=f"No match found. Model: '{m}', Expected: '{e}'.",
        )

    def _normalize(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        s = normalize_answer(str(answer))

        # Handle LaTeX: extract from \boxed{} if present
        boxed = extract_boxed(s)
        if boxed is not None:
            s = boxed

        # Strip remaining LaTeX formatting
        if "\\" in s:
            s = strip_latex_formatting(s)

        return normalize_answer(s)
