"""Final verdict determination."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from src.criteria.base import CriterionResult, Evidence


class VerdictCategory(Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    NOT_MET = "NOT_MET"


@dataclass
class FinalVerdict:
    """Final GO/NO-GO verdict."""

    category: VerdictCategory
    criteria_results: List[CriterionResult]
    n_passed: int
    n_total: int
    overall_confidence: float
    rationale: str
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a one-line summary of the verdict."""
        return (
            f"VERDICT: {self.category.value} "
            f"({self.n_passed}/{self.n_total} criteria passed, "
            f"confidence={self.overall_confidence:.2f}). "
            f"{self.rationale}"
        )

    @property
    def is_go(self) -> bool:
        """True if verdict is SUCCESS (GO decision)."""
        return self.category == VerdictCategory.SUCCESS

    @property
    def passed_criteria(self) -> List[CriterionResult]:
        """Return list of passed criteria."""
        return [r for r in self.criteria_results if r.passed]

    @property
    def failed_criteria(self) -> List[CriterionResult]:
        """Return list of failed criteria."""
        return [r for r in self.criteria_results if not r.passed]


class SuccessVerdict:
    """Determines the final verdict based on criteria results."""

    def __init__(
        self,
        success_threshold: int = 5,
        partial_min: int = 3,
        partial_max: int = 4,
    ):
        self._success_threshold = success_threshold
        self._partial_min = partial_min
        self._partial_max = partial_max

    def evaluate(self, results: List[CriterionResult]) -> FinalVerdict:
        """Determine the final verdict from criterion results.

        SUCCESS: all criteria pass (5/5)
        PARTIAL: 3-4 criteria pass
        NOT_MET: <= 2 criteria pass
        """
        n_total = len(results)
        n_passed = sum(1 for r in results if r.passed)

        # Determine category
        if n_passed >= self._success_threshold:
            category = VerdictCategory.SUCCESS
            rationale = "All pre-registered criteria have been met."
        elif n_passed >= self._partial_min:
            category = VerdictCategory.PARTIAL
            failed_names = [
                r.criterion_name for r in results if not r.passed
            ]
            rationale = (
                f"Partial success: {n_passed}/{n_total} criteria met. "
                f"Failed: {', '.join(failed_names)}."
            )
        else:
            category = VerdictCategory.NOT_MET
            failed_names = [
                r.criterion_name for r in results if not r.passed
            ]
            rationale = (
                f"Success criteria not met: only {n_passed}/{n_total} passed. "
                f"Failed: {', '.join(failed_names)}."
            )

        # Overall confidence
        if results:
            confidences = [r.confidence for r in results]
            overall_confidence = sum(confidences) / len(confidences)
        else:
            overall_confidence = 0.0

        return FinalVerdict(
            category=category,
            criteria_results=results,
            n_passed=n_passed,
            n_total=n_total,
            overall_confidence=overall_confidence,
            rationale=rationale,
            details={
                "passed_names": [
                    r.criterion_name for r in results if r.passed
                ],
                "failed_names": [
                    r.criterion_name for r in results if not r.passed
                ],
                "individual_confidences": {
                    r.criterion_name: r.confidence for r in results
                },
            },
        )
