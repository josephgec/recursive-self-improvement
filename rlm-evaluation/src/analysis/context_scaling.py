"""Context scaling analysis: accuracy vs context length."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.comparison.scaling_experiment import ScalingResult


class ContextScalingAnalysis:
    """Analyze and visualize how accuracy scales with context length."""

    def plot_ascii(
        self,
        results: List[ScalingResult],
    ) -> str:
        """Generate ASCII plot of accuracy vs context size."""
        if not results:
            return "No scaling data available."

        lines = [
            "Accuracy vs Context Size",
            "=" * 60,
            "",
        ]

        width = 40

        for r in results:
            rlm_bar = int(r.rlm_accuracy * width)
            std_bar = int(r.standard_accuracy * width)

            label = f"{r.context_size:>7}tok"
            lines.append(f"{label} RLM |{'#' * rlm_bar:<{width}}| {r.rlm_accuracy:.0%}")
            lines.append(f"{'':>10} STD |{'.' * std_bar:<{width}}| {r.standard_accuracy:.0%}")
            lines.append("")

        return "\n".join(lines)

    def degradation_analysis(
        self,
        results: List[ScalingResult],
    ) -> Dict[str, Dict[str, float]]:
        """Analyze how each system degrades with context size.

        Returns:
            Dict with 'rlm' and 'standard' keys, each containing:
            - start_accuracy: accuracy at smallest context
            - end_accuracy: accuracy at largest context
            - degradation: total drop
            - degradation_rate: drop per 10k tokens
        """
        analysis: Dict[str, Dict[str, float]] = {}

        if not results:
            return analysis

        for system, acc_key in [("rlm", "rlm_accuracy"), ("standard", "standard_accuracy")]:
            start_acc = getattr(results[0], acc_key)
            end_acc = getattr(results[-1], acc_key)
            degradation = start_acc - end_acc

            context_range = results[-1].context_size - results[0].context_size
            rate = degradation / (context_range / 10000) if context_range > 0 else 0

            analysis[system] = {
                "start_accuracy": start_acc,
                "end_accuracy": end_acc,
                "degradation": degradation,
                "degradation_rate_per_10k": rate,
            }

        return analysis

    def advantage_curve(
        self,
        results: List[ScalingResult],
    ) -> List[Tuple[int, float]]:
        """Compute the RLM advantage (rlm_acc - std_acc) at each context size.

        Returns:
            List of (context_size, advantage) tuples.
        """
        return [
            (r.context_size, r.rlm_accuracy - r.standard_accuracy)
            for r in results
        ]
