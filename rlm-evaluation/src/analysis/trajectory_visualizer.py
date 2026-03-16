"""Visualize RLM execution trajectories."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.benchmarks.task import EvalResult


class TrajectoryVisualizer:
    """Render and select representative trajectory examples."""

    def render(self, result: EvalResult) -> str:
        """Render a single trajectory as formatted text.

        Args:
            result: An evaluation result with trajectory.

        Returns:
            Formatted string representation.
        """
        lines = [
            f"=== Trajectory: {result.task_id} ===",
            f"Strategy: {result.strategy_detected}",
            f"Correct: {result.correct}",
            f"Cost: ${result.cost:.4f}",
            f"Calls: {result.num_calls}",
            "",
        ]

        for i, step in enumerate(result.trajectory, 1):
            lines.append(f"Step {i}:")
            for line in step.split("\n"):
                lines.append(f"  {line}")
            lines.append("")

        lines.append(f"Answer: {result.answer}")
        lines.append("=" * 40)

        return "\n".join(lines)

    def render_batch(self, results: List[EvalResult], max_show: int = 5) -> str:
        """Render multiple trajectories."""
        output_parts = []
        for r in results[:max_show]:
            output_parts.append(self.render(r))
        return "\n\n".join(output_parts)

    def select_representative(
        self,
        results: List[EvalResult],
        num_examples: int = 5,
    ) -> List[EvalResult]:
        """Select representative trajectory examples.

        Tries to include:
        - One per strategy type
        - Both correct and incorrect examples
        - Various trajectory lengths
        """
        if len(results) <= num_examples:
            return list(results)

        selected: List[EvalResult] = []
        seen_strategies: set = set()
        has_correct = False
        has_incorrect = False

        # First pass: one per strategy
        for r in results:
            strategy = r.strategy_detected or "unknown"
            if strategy not in seen_strategies and len(selected) < num_examples:
                selected.append(r)
                seen_strategies.add(strategy)
                if r.correct:
                    has_correct = True
                else:
                    has_incorrect = True

        # Second pass: ensure we have both correct and incorrect
        if not has_correct and len(selected) < num_examples:
            for r in results:
                if r.correct and r not in selected:
                    selected.append(r)
                    break

        if not has_incorrect and len(selected) < num_examples:
            for r in results:
                if not r.correct and r not in selected:
                    selected.append(r)
                    break

        # Fill remaining slots
        for r in results:
            if len(selected) >= num_examples:
                break
            if r not in selected:
                selected.append(r)

        return selected[:num_examples]

    def strategy_examples(
        self,
        results: List[EvalResult],
    ) -> Dict[str, List[EvalResult]]:
        """Group results by strategy for easy browsing."""
        grouped: Dict[str, List[EvalResult]] = {}
        for r in results:
            strategy = r.strategy_detected or "unknown"
            grouped.setdefault(strategy, []).append(r)
        return grouped
