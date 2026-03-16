"""AccuracyFloorConstraint: minimum accuracy on held-out evaluation tasks."""

from __future__ import annotations

from typing import Any, Dict, List

from src.constraints.base import Constraint, ConstraintResult, CheckContext


class AccuracyFloorConstraint(Constraint):
    """Agent must maintain accuracy >= threshold on held-out tasks."""

    def __init__(self, threshold: float = 0.80) -> None:
        super().__init__(
            name="accuracy_floor",
            description="Minimum accuracy on held-out evaluation tasks",
            category="quality",
            threshold=threshold,
        )

    def check(self, agent_state: Any, context: CheckContext) -> ConstraintResult:
        """Evaluate accuracy.

        ``agent_state`` must expose:
        * ``evaluate(tasks) -> List[dict]`` where each dict has ``correct: bool``
          and ``category: str``.
        * If ``agent_state`` has a ``held_out_tasks`` attribute those are used;
          otherwise the built-in held-out suite is loaded.
        """
        tasks = self._get_tasks(agent_state)
        results = agent_state.evaluate(tasks)

        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        accuracy = correct / total if total else 0.0

        # per-category breakdown
        categories: Dict[str, Dict[str, int]] = {}
        for r in results:
            cat = r.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"correct": 0, "total": 0}
            categories[cat]["total"] += 1
            if r["correct"]:
                categories[cat]["correct"] += 1

        category_accuracy = {
            cat: vals["correct"] / vals["total"] if vals["total"] else 0.0
            for cat, vals in categories.items()
        }

        headroom = self.headroom(accuracy)
        return ConstraintResult(
            satisfied=accuracy >= self._threshold,
            measured_value=accuracy,
            threshold=self._threshold,
            headroom=headroom,
            details={
                "total": total,
                "correct": correct,
                "per_category": category_accuracy,
            },
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _get_tasks(agent_state: Any) -> List[dict]:
        if hasattr(agent_state, "held_out_tasks") and agent_state.held_out_tasks:
            return agent_state.held_out_tasks

        # Fall back to built-in suite
        from src.evaluation.held_out_suite import HeldOutSuite
        return HeldOutSuite().load()
