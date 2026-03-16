"""Recursive improver: runs self-improvement within an RLM session."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.pipeline.state import PipelineState
from src.scaling.rlm_wrapper import RLMWrapper


class RecursiveImprover:
    """Runs self-improvement loops within an RLM session context."""

    def __init__(self, rlm_wrapper: Optional[RLMWrapper] = None, max_depth: int = 3):
        self._rlm = rlm_wrapper or RLMWrapper()
        self._max_depth = max_depth
        self._improvement_log: List[Dict[str, Any]] = []

    def improve_within_rlm(
        self,
        state: PipelineState,
        iteration_fn: Callable,
        depth: int = 0,
    ) -> Dict[str, Any]:
        """Run self-improvement within an RLM session.

        Can recursively improve by using the RLM to guide modifications,
        up to max_depth recursions.
        """
        if depth >= self._max_depth:
            return {
                "depth": depth,
                "status": "max_depth_reached",
                "improvements": list(self._improvement_log),
            }

        # Wrap the iteration with RLM context
        accuracy_before = state.performance.accuracy

        result = self._rlm.wrap_iteration(state, iteration_fn)

        accuracy_after = state.performance.accuracy
        improved = accuracy_after > accuracy_before

        log_entry = {
            "depth": depth,
            "accuracy_before": accuracy_before,
            "accuracy_after": accuracy_after,
            "improved": improved,
        }
        self._improvement_log.append(log_entry)

        if improved and depth + 1 < self._max_depth:
            # Recurse: try to improve further
            return self.improve_within_rlm(state, iteration_fn, depth + 1)

        return {
            "depth": depth,
            "status": "improved" if improved else "no_improvement",
            "improvements": list(self._improvement_log),
        }

    @property
    def improvement_log(self) -> List[Dict[str, Any]]:
        return list(self._improvement_log)

    def clear_log(self) -> None:
        self._improvement_log.clear()
