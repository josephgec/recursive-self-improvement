"""Cost modeling for RLM vs standard LLM evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.benchmarks.task import EvalResult


@dataclass
class CostBreakdown:
    """Breakdown of costs for a system."""
    system: str
    total_cost: float
    input_cost: float
    output_cost: float
    total_input_tokens: int
    total_output_tokens: int
    num_tasks: int
    num_correct: int
    cost_per_task: float = 0.0
    cost_per_correct: float = 0.0

    def __post_init__(self) -> None:
        if self.num_tasks > 0:
            self.cost_per_task = self.total_cost / self.num_tasks
        if self.num_correct > 0:
            self.cost_per_correct = self.total_cost / self.num_correct


@dataclass
class CostComparison:
    """Comparison of costs between systems."""
    systems: Dict[str, CostBreakdown] = field(default_factory=dict)
    cost_ratio: float = 0.0
    accuracy_ratio: float = 0.0
    efficiency_winner: str = ""

    def summary(self) -> str:
        """Generate comparison summary."""
        lines = ["Cost Comparison:"]
        for name, bd in self.systems.items():
            accuracy = bd.num_correct / bd.num_tasks if bd.num_tasks > 0 else 0
            lines.append(
                f"  {name}: ${bd.total_cost:.4f} total, "
                f"{accuracy:.1%} accuracy, "
                f"${bd.cost_per_correct:.4f}/correct"
            )
        lines.append(f"  Cost ratio: {self.cost_ratio:.2f}x")
        lines.append(f"  Accuracy ratio: {self.accuracy_ratio:.2f}x")
        lines.append(f"  Efficiency winner: {self.efficiency_winner}")
        return "\n".join(lines)


class CostModel:
    """Model and compare costs across different systems."""

    # Default pricing per 1K tokens
    DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
        "rlm": {"input": 0.01, "output": 0.03},
        "standard": {"input": 0.005, "output": 0.015},
        "gpt4": {"input": 0.03, "output": 0.06},
        "claude": {"input": 0.008, "output": 0.024},
    }

    def __init__(self, pricing: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        self.pricing = pricing or dict(self.DEFAULT_PRICING)

    def compute_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "rlm",
    ) -> float:
        """Compute cost for a single call."""
        prices = self.pricing.get(model, self.pricing.get("standard", {"input": 0.01, "output": 0.03}))
        return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1000

    def cost_breakdown(
        self,
        results: List[EvalResult],
        system: str = "rlm",
    ) -> CostBreakdown:
        """Compute detailed cost breakdown for a set of results."""
        total_input = sum(r.input_tokens for r in results)
        total_output = sum(r.output_tokens for r in results)
        num_correct = sum(1 for r in results if r.correct)

        prices = self.pricing.get(system, {"input": 0.01, "output": 0.03})
        input_cost = total_input * prices["input"] / 1000
        output_cost = total_output * prices["output"] / 1000

        return CostBreakdown(
            system=system,
            total_cost=input_cost + output_cost,
            input_cost=input_cost,
            output_cost=output_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            num_tasks=len(results),
            num_correct=num_correct,
        )

    def cost_per_correct(
        self,
        results: List[EvalResult],
        model: str = "rlm",
    ) -> float:
        """Compute cost per correct answer."""
        breakdown = self.cost_breakdown(results, model)
        return breakdown.cost_per_correct

    def accuracy_per_dollar(
        self,
        results: List[EvalResult],
        model: str = "rlm",
    ) -> float:
        """Compute correct answers per dollar."""
        breakdown = self.cost_breakdown(results, model)
        if breakdown.total_cost == 0:
            return 0.0
        return breakdown.num_correct / breakdown.total_cost

    def compare_systems(
        self,
        rlm_results: List[EvalResult],
        standard_results: List[EvalResult],
        rlm_model: str = "rlm",
        standard_model: str = "standard",
    ) -> CostComparison:
        """Compare costs between RLM and standard systems."""
        rlm_bd = self.cost_breakdown(rlm_results, rlm_model)
        std_bd = self.cost_breakdown(standard_results, standard_model)

        cost_ratio = rlm_bd.total_cost / std_bd.total_cost if std_bd.total_cost > 0 else float("inf")

        rlm_accuracy = rlm_bd.num_correct / rlm_bd.num_tasks if rlm_bd.num_tasks > 0 else 0
        std_accuracy = std_bd.num_correct / std_bd.num_tasks if std_bd.num_tasks > 0 else 0
        accuracy_ratio = rlm_accuracy / std_accuracy if std_accuracy > 0 else float("inf")

        # Efficiency: accuracy per dollar
        rlm_eff = rlm_bd.num_correct / rlm_bd.total_cost if rlm_bd.total_cost > 0 else 0
        std_eff = std_bd.num_correct / std_bd.total_cost if std_bd.total_cost > 0 else 0
        winner = rlm_model if rlm_eff >= std_eff else standard_model

        return CostComparison(
            systems={rlm_model: rlm_bd, standard_model: std_bd},
            cost_ratio=cost_ratio,
            accuracy_ratio=accuracy_ratio,
            efficiency_winner=winner,
        )
