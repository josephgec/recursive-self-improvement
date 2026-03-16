"""Empirical gate: evaluates candidates by executing test cases."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.outer_loop.strategy_evolver import Candidate
from src.pipeline.state import PipelineState


@dataclass
class EmpiricalResult:
    """Result of empirical evaluation."""
    candidate_id: str = ""
    passed: bool = False
    pass_rate: float = 0.0
    accuracy: float = 0.0
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "accuracy": self.accuracy,
            "error": self.error,
        }


class EmpiricalGate:
    """Evaluates candidates by running them against test cases in a sandbox."""

    def __init__(
        self,
        min_pass_rate: float = 0.8,
        test_runner: Optional[Callable] = None,
        timeout: int = 30,
    ):
        self._min_pass_rate = min_pass_rate
        self._test_runner = test_runner or self._default_test_runner
        self._timeout = timeout

    def evaluate(self, candidate: Candidate, state: PipelineState) -> EmpiricalResult:
        """Evaluate a candidate against test cases."""
        try:
            test_output = self._test_runner(candidate.proposed_code, state)
            pass_rate = test_output.get("pass_rate", 0.0)
            accuracy = test_output.get("accuracy", pass_rate)
            passed = pass_rate >= self._min_pass_rate

            return EmpiricalResult(
                candidate_id=candidate.candidate_id,
                passed=passed,
                pass_rate=pass_rate,
                accuracy=accuracy,
                test_results=test_output.get("details", []),
            )
        except Exception as e:
            return EmpiricalResult(
                candidate_id=candidate.candidate_id,
                passed=False,
                pass_rate=0.0,
                error=str(e),
            )

    @staticmethod
    def _default_test_runner(code: str, state: PipelineState) -> Dict[str, Any]:
        """Mock test runner — passes if code is non-empty and non-trivial."""
        if not code or len(code.strip()) < 5:
            return {"pass_rate": 0.0, "accuracy": 0.0, "details": []}
        # Simple heuristic: longer code = better (for mock)
        length_score = min(len(code) / 100.0, 1.0)
        return {
            "pass_rate": max(0.5, length_score),
            "accuracy": max(0.5, length_score * 0.9),
            "details": [{"test": "mock", "passed": True}],
        }
