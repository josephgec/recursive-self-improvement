"""Tracks cost metrics: LLM calls, finetuning, totals."""

from typing import Dict, List


class CostTracker:
    """Records and reports cost information."""

    def __init__(self):
        self._llm_costs: List[float] = []
        self._finetuning_costs: List[float] = []

    def record_llm_call(self, cost: float):
        """Record the cost of an LLM API call."""
        self._llm_costs.append(cost)

    def record_finetuning(self, cost: float):
        """Record the cost of a finetuning step."""
        self._finetuning_costs.append(cost)

    def get_total_cost(self) -> float:
        """Get total cost across all categories."""
        return sum(self._llm_costs) + sum(self._finetuning_costs)

    def get_cost_per_iteration(self) -> float:
        """Get average cost per iteration (based on LLM calls as proxy for iterations)."""
        total_iterations = len(self._llm_costs)
        if total_iterations == 0:
            return 0.0
        return self.get_total_cost() / total_iterations

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost broken down by category."""
        return {
            "llm_calls": sum(self._llm_costs),
            "finetuning": sum(self._finetuning_costs),
            "total": self.get_total_cost(),
        }

    def get_llm_call_count(self) -> int:
        """Number of LLM calls recorded."""
        return len(self._llm_costs)

    def get_finetuning_count(self) -> int:
        """Number of finetuning steps recorded."""
        return len(self._finetuning_costs)
