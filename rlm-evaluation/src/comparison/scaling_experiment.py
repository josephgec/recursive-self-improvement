"""Context scaling experiment: how systems perform as context grows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from src.benchmarks.task import EvalTask, EvalResult


@dataclass
class ScalingResult:
    """Result at a single context size."""
    context_size: int
    rlm_accuracy: float
    standard_accuracy: float
    rlm_cost: float
    standard_cost: float
    rlm_results: List[EvalResult] = field(default_factory=list)
    standard_results: List[EvalResult] = field(default_factory=list)


class ContextScalingExperiment:
    """Run context scaling experiments comparing RLM vs standard."""

    def __init__(
        self,
        rlm_executor_fn: Callable[[EvalTask], EvalResult],
        standard_executor_fn: Callable[[EvalTask], EvalResult],
    ) -> None:
        self.rlm_executor_fn = rlm_executor_fn
        self.standard_executor_fn = standard_executor_fn

    def run(
        self,
        base_tasks: List[EvalTask],
        context_sizes: List[int],
    ) -> List[ScalingResult]:
        """Run the scaling experiment across context sizes.

        Args:
            base_tasks: Tasks to scale.
            context_sizes: List of context token counts to test.

        Returns:
            List of ScalingResult, one per context size.
        """
        results: List[ScalingResult] = []

        for size in context_sizes:
            sized_tasks = [t.with_context_size(size) for t in base_tasks]

            rlm_results = [self.rlm_executor_fn(t) for t in sized_tasks]
            std_results = [self.standard_executor_fn(t) for t in sized_tasks]

            rlm_correct = sum(1 for r in rlm_results if r.correct)
            std_correct = sum(1 for r in std_results if r.correct)

            n = len(sized_tasks)
            sr = ScalingResult(
                context_size=size,
                rlm_accuracy=rlm_correct / n if n > 0 else 0.0,
                standard_accuracy=std_correct / n if n > 0 else 0.0,
                rlm_cost=sum(r.cost for r in rlm_results),
                standard_cost=sum(r.cost for r in std_results),
                rlm_results=rlm_results,
                standard_results=std_results,
            )
            results.append(sr)

        return results

    def compute_degradation_curve(
        self,
        results: List[ScalingResult],
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Compute accuracy degradation curves.

        Returns:
            Dict with 'rlm' and 'standard' keys, each mapping to
            list of (context_size, accuracy) tuples.
        """
        return {
            "rlm": [(r.context_size, r.rlm_accuracy) for r in results],
            "standard": [(r.context_size, r.standard_accuracy) for r in results],
        }

    def find_crossover_point(
        self,
        results: List[ScalingResult],
    ) -> Optional[int]:
        """Find the context size where RLM first surpasses standard.

        Returns:
            Context size of crossover, or None if RLM never surpasses.
        """
        for r in results:
            if r.rlm_accuracy > r.standard_accuracy:
                return r.context_size
        return None

    def plot_scaling_curves(
        self,
        results: List[ScalingResult],
    ) -> str:
        """Generate an ASCII plot of scaling curves.

        Returns:
            ASCII art representation of the scaling curves.
        """
        if not results:
            return "No data to plot."

        lines: List[str] = []
        lines.append("Accuracy vs Context Size")
        lines.append("=" * 60)

        max_acc = 1.0
        width = 50

        for r in results:
            rlm_bar = int(r.rlm_accuracy / max_acc * width)
            std_bar = int(r.standard_accuracy / max_acc * width)

            size_label = f"{r.context_size:>7}"
            lines.append(f"{size_label} RLM |{'#' * rlm_bar:<{width}}| {r.rlm_accuracy:.1%}")
            lines.append(f"{'':>7} STD |{'.' * std_bar:<{width}}| {r.standard_accuracy:.1%}")
            lines.append("")

        crossover = self.find_crossover_point(results)
        if crossover is not None:
            lines.append(f"Crossover point: {crossover} tokens")
        else:
            lines.append("No crossover detected")

        return "\n".join(lines)
