"""Tracks stability metrics: rollbacks, oscillation, consecutive issues."""

from typing import List


class StabilityTracker:
    """Records rollback events and computes stability metrics."""

    def __init__(self):
        self._rollback_iterations: List[int] = []

    def record_rollback(self, iteration: int):
        """Record that a rollback occurred at this iteration."""
        self._rollback_iterations.append(iteration)

    def total_rollbacks(self) -> int:
        """Total number of rollbacks."""
        return len(self._rollback_iterations)

    def get_rollback_rate(self, total_iterations: int) -> float:
        """Fraction of iterations that had rollbacks."""
        if total_iterations <= 0:
            return 0.0
        return len(self._rollback_iterations) / total_iterations

    def detect_oscillation(self, window: int = 5) -> bool:
        """Detect if rollbacks are oscillating (frequent consecutive rollbacks)."""
        if len(self._rollback_iterations) < 3:
            return False
        # Check if there are clusters of rollbacks within a window
        sorted_iters = sorted(self._rollback_iterations)
        for i in range(len(sorted_iters) - 2):
            if sorted_iters[i + 2] - sorted_iters[i] <= window:
                return True
        return False

    def consecutive_rollbacks(self) -> int:
        """Maximum number of consecutive rollback iterations."""
        if not self._rollback_iterations:
            return 0
        sorted_iters = sorted(self._rollback_iterations)
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(sorted_iters)):
            if sorted_iters[i] == sorted_iters[i - 1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        return max_consecutive

    def stability_score(self, total_iterations: int) -> float:
        """Overall stability score (0-1, higher is more stable).

        Combines rollback rate, oscillation, and consecutive rollbacks.
        """
        if total_iterations <= 0:
            return 1.0

        rollback_penalty = self.get_rollback_rate(total_iterations)
        oscillation_penalty = 0.2 if self.detect_oscillation() else 0.0
        consecutive_penalty = min(self.consecutive_rollbacks() * 0.05, 0.3)

        score = 1.0 - rollback_penalty - oscillation_penalty - consecutive_penalty
        return max(0.0, min(1.0, score))
