"""Analyze head role evolution over iterations."""

from typing import Dict, List, Tuple, Any
import numpy as np

from src.attention.specialization import HeadTrackingResult, HeadRoleChange


class HeadEvolutionAnalyzer:
    """Analyze how attention head roles evolve over iterations."""

    def __init__(self):
        self._tracking_history: List[HeadTrackingResult] = []

    def add_tracking_result(self, result: HeadTrackingResult) -> None:
        """Add a tracking result to history."""
        self._tracking_history.append(result)

    def get_role_transitions(self) -> List[Dict[str, Any]]:
        """Get all role transitions across history."""
        transitions = []
        for i, result in enumerate(self._tracking_history):
            for rc in result.role_changes:
                transitions.append({
                    "iteration": i,
                    "layer": rc.layer,
                    "head": rc.head,
                    "from": rc.role_before,
                    "to": rc.role_after,
                    "magnitude": rc.magnitude,
                })
        return transitions

    def get_head_stability(self) -> Dict[Tuple[int, int], float]:
        """Compute stability score for each head (fraction of iterations without role change)."""
        if not self._tracking_history:
            return {}

        all_heads = set()
        for result in self._tracking_history:
            for rc in result.role_changes:
                all_heads.add((rc.layer, rc.head))
            for shift in result.shifts:
                all_heads.add((shift.layer, shift.head))

        stability = {}
        n_iters = len(self._tracking_history)
        for head in all_heads:
            changes = sum(
                1 for r in self._tracking_history
                for rc in r.role_changes
                if (rc.layer, rc.head) == head
            )
            stability[head] = 1.0 - changes / max(n_iters, 1)
        return stability

    def get_dying_head_trend(self) -> List[int]:
        """Get trend of dying head count over iterations."""
        return [len(r.dying_heads) for r in self._tracking_history]

    def get_narrowing_head_trend(self) -> List[int]:
        """Get trend of narrowing head count over iterations."""
        return [len(r.narrowing_heads) for r in self._tracking_history]

    def get_entropy_trend(self) -> List[float]:
        """Get trend of mean entropy over iterations."""
        return [
            r.summary.get("mean_entropy", 0.0)
            for r in self._tracking_history
        ]

    def summarize(self) -> Dict[str, Any]:
        """Summarize head evolution analysis."""
        transitions = self.get_role_transitions()
        dying_trend = self.get_dying_head_trend()
        narrowing_trend = self.get_narrowing_head_trend()
        entropy_trend = self.get_entropy_trend()

        return {
            "total_iterations": len(self._tracking_history),
            "total_role_transitions": len(transitions),
            "dying_head_trend": dying_trend,
            "narrowing_head_trend": narrowing_trend,
            "entropy_trend": entropy_trend,
            "mean_entropy_overall": float(np.mean(entropy_trend)) if entropy_trend else 0.0,
        }
