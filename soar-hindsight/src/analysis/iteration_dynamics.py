"""Analyze dynamics across SOAR iterations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class IterationDynamicsAnalyzer:
    """Analyze how metrics evolve across SOAR iterations."""

    def __init__(self) -> None:
        self._iteration_data: List[Dict[str, Any]] = []

    def load_history(self, history: List[Dict[str, Any]]) -> None:
        """Load iteration history from SOARLoop."""
        self._iteration_data = list(history)

    @property
    def n_iterations(self) -> int:
        return len(self._iteration_data)

    def solve_rate_trajectory(self) -> List[Tuple[int, float]]:
        """Extract solve rate trajectory over iterations."""
        return [
            (d.get("iteration", i + 1), d.get("solve_rate", 0.0))
            for i, d in enumerate(self._iteration_data)
        ]

    def data_volume_trajectory(self) -> List[Tuple[int, int]]:
        """Track training data volume over iterations."""
        return [
            (d.get("iteration", i + 1), d.get("n_pairs_after_filter", 0))
            for i, d in enumerate(self._iteration_data)
        ]

    def training_loss_trajectory(self) -> List[Tuple[int, float]]:
        """Track final training loss over iterations."""
        result = []
        for i, d in enumerate(self._iteration_data):
            training = d.get("training", {})
            metrics = training.get("metrics", {})
            loss = metrics.get("train_loss_final", None)
            if loss is not None:
                result.append((d.get("iteration", i + 1), loss))
        return result

    def improvement_rate(self) -> List[Tuple[int, float]]:
        """Compute per-iteration improvement in solve rate."""
        rates = self.solve_rate_trajectory()
        if len(rates) < 2:
            return []
        result = []
        for i in range(1, len(rates)):
            delta = rates[i][1] - rates[i - 1][1]
            result.append((rates[i][0], round(delta, 4)))
        return result

    def cumulative_pairs(self) -> List[Tuple[int, int]]:
        """Cumulative training pairs over iterations."""
        total = 0
        result = []
        for i, d in enumerate(self._iteration_data):
            total += d.get("n_pairs_after_filter", 0)
            result.append((d.get("iteration", i + 1), total))
        return result

    def convergence_analysis(self) -> Dict[str, Any]:
        """Analyze convergence behavior."""
        rates = self.solve_rate_trajectory()
        if len(rates) < 2:
            return {"converged": False, "plateau_start": None}

        # Find where improvement drops below threshold
        improvements = self.improvement_rate()
        threshold = 0.01
        plateau_start = None
        for it, delta in improvements:
            if delta < threshold and plateau_start is None:
                plateau_start = it
            elif delta >= threshold:
                plateau_start = None

        return {
            "converged": any(d.get("converged", False) for d in self._iteration_data),
            "plateau_start": plateau_start,
            "final_solve_rate": rates[-1][1] if rates else 0.0,
            "total_improvement": round(rates[-1][1] - rates[0][1], 4) if len(rates) >= 2 else 0.0,
        }

    def full_report(self) -> Dict[str, Any]:
        return {
            "n_iterations": self.n_iterations,
            "solve_rate_trajectory": self.solve_rate_trajectory(),
            "data_volume_trajectory": self.data_volume_trajectory(),
            "improvement_rate": self.improvement_rate(),
            "convergence": self.convergence_analysis(),
        }
