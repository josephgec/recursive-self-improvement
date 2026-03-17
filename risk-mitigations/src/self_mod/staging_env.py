"""Staging environment for testing self-modifications before deployment.

Clones the current agent state, applies a candidate modification,
evaluates it on benchmarks, and compares to the original.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StagingResult:
    """Result of testing a modification in staging."""
    passed: bool
    original_score: float
    modified_score: float
    improvement: float
    benchmark_results: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    candidate_id: str = ""
    detail: str = ""

    @property
    def regression(self) -> bool:
        return self.modified_score < self.original_score


class StagingEnvironment:
    """Staging environment for safe modification testing.

    Clones the live agent, applies modifications in isolation,
    and evaluates against benchmarks before promoting to live.
    """

    def __init__(
        self,
        min_benchmark_pass_rate: float = 0.95,
        benchmarks: Optional[List[str]] = None,
    ):
        self.min_benchmark_pass_rate = min_benchmark_pass_rate
        self.benchmarks = benchmarks or [
            "correctness",
            "safety",
            "coherence",
            "efficiency",
            "robustness",
        ]
        self._results_history: List[StagingResult] = []

    def test_modification(
        self,
        agent_state: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> StagingResult:
        """Test a candidate modification in staging.

        Args:
            agent_state: Current live agent state (will NOT be modified).
            candidate: The modification to test. Should have:
                - 'id': Unique identifier
                - 'changes': Dict of parameter changes
                - Optional 'expected_improvement': float

        Returns:
            StagingResult with pass/fail and metrics.
        """
        # Clone - live agent must not be touched
        staged_state = copy.deepcopy(agent_state)

        # Apply modification to clone
        errors = []
        try:
            staged_state = self._apply_modification(staged_state, candidate)
        except Exception as e:
            errors.append(f"Failed to apply modification: {e}")
            result = StagingResult(
                passed=False,
                original_score=0.0,
                modified_score=0.0,
                improvement=0.0,
                errors=errors,
                candidate_id=candidate.get("id", "unknown"),
                detail="Modification could not be applied",
            )
            self._results_history.append(result)
            return result

        # Evaluate both
        original_scores = self._evaluate(agent_state)
        modified_scores = self._evaluate(staged_state)

        original_mean = sum(original_scores.values()) / len(original_scores) if original_scores else 0.0
        modified_mean = sum(modified_scores.values()) / len(modified_scores) if modified_scores else 0.0
        improvement = modified_mean - original_mean

        # Check pass rate
        benchmarks_passed = sum(
            1 for b in self.benchmarks
            if modified_scores.get(b, 0.0) >= original_scores.get(b, 0.0) * 0.95
        )
        pass_rate = benchmarks_passed / len(self.benchmarks) if self.benchmarks else 0.0
        passed = pass_rate >= self.min_benchmark_pass_rate and not errors

        result = StagingResult(
            passed=passed,
            original_score=original_mean,
            modified_score=modified_mean,
            improvement=improvement,
            benchmark_results=modified_scores,
            errors=errors,
            candidate_id=candidate.get("id", "unknown"),
            detail=f"Pass rate: {pass_rate:.1%} ({benchmarks_passed}/{len(self.benchmarks)})",
        )
        self._results_history.append(result)
        return result

    def _apply_modification(
        self, state: Dict[str, Any], candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply a candidate modification to a cloned state."""
        changes = candidate.get("changes", {})
        for key, value in changes.items():
            if key.startswith("__") or key == "destroy":
                raise ValueError(f"Unsafe modification key: {key}")
            state[key] = value
        return state

    def _evaluate(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate an agent state against benchmarks.

        This is a mock evaluation. In production, this would run actual benchmarks.
        """
        scores = {}
        base_score = state.get("quality", 0.8)
        modifier = state.get("modifier", 0.0)

        for i, benchmark in enumerate(self.benchmarks):
            # Deterministic mock scores based on state
            score = base_score + modifier + (i * 0.01)
            scores[benchmark] = min(max(score, 0.0), 1.0)

        return scores

    def get_results_history(self) -> List[StagingResult]:
        """Return all staging results."""
        return list(self._results_history)
