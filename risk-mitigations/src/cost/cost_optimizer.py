"""Cost optimizer - suggests cost savings based on forecasts and configuration.

Analyzes spending patterns and suggests optimizations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CostSaving:
    """A suggested cost saving measure."""
    category: str
    description: str
    estimated_savings: float
    effort: str  # "low", "medium", "high"
    risk: str  # "low", "medium", "high"
    priority: int  # 1=highest

    @property
    def roi_estimate(self) -> str:
        """Rough ROI based on savings vs effort."""
        if self.effort == "low" and self.estimated_savings > 0:
            return "high"
        elif self.effort == "high" and self.estimated_savings < 100:
            return "low"
        return "medium"


class CostOptimizer:
    """Suggests cost optimization strategies.

    Analyzes forecasts and configurations to recommend
    cost-saving measures.
    """

    def suggest(
        self,
        forecast: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[CostSaving]:
        """Suggest cost savings based on forecast and config.

        Args:
            forecast: A CostForecast object.
            config: Optional configuration dict.

        Returns:
            List of CostSaving suggestions, ordered by priority.
        """
        config = config or {}
        suggestions = []
        priority = 1

        # Check burn rate
        burn_rate = getattr(forecast, "burn_rate_per_iteration", 0.0)
        budget_remaining = getattr(forecast, "budget_remaining", float("inf"))
        budget_sufficient = getattr(forecast, "budget_sufficient", True)

        if not budget_sufficient:
            suggestions.append(CostSaving(
                category="budget",
                description="Budget will be exhausted before completion. Reduce iteration count or find efficiency gains.",
                estimated_savings=burn_rate * 5,
                effort="medium",
                risk="low",
                priority=priority,
            ))
            priority += 1

        # Suggest caching if burn rate is high
        if burn_rate > 10.0:
            suggestions.append(CostSaving(
                category="caching",
                description="Implement result caching to reduce redundant computations.",
                estimated_savings=burn_rate * 0.3,
                effort="medium",
                risk="low",
                priority=priority,
            ))
            priority += 1

        # Suggest batching
        if burn_rate > 5.0:
            suggestions.append(CostSaving(
                category="batching",
                description="Batch similar queries to reduce per-query overhead.",
                estimated_savings=burn_rate * 0.15,
                effort="low",
                risk="low",
                priority=priority,
            ))
            priority += 1

        # Suggest model downgrade if cost is very high
        if burn_rate > 20.0:
            suggestions.append(CostSaving(
                category="model_selection",
                description="Use a smaller model for non-critical queries.",
                estimated_savings=burn_rate * 0.5,
                effort="medium",
                risk="medium",
                priority=priority,
            ))
            priority += 1

        # Suggest early stopping
        if budget_remaining < burn_rate * 10 and burn_rate > 0:
            suggestions.append(CostSaving(
                category="early_stopping",
                description="Consider early stopping if diminishing returns observed.",
                estimated_savings=budget_remaining * 0.3,
                effort="low",
                risk="medium",
                priority=priority,
            ))
            priority += 1

        # Suggest reducing eval frequency
        eval_frequency = config.get("eval_frequency", 1)
        if eval_frequency == 1 and burn_rate > 3.0:
            suggestions.append(CostSaving(
                category="eval_frequency",
                description="Reduce evaluation frequency from every iteration to every 5th.",
                estimated_savings=burn_rate * 0.2,
                effort="low",
                risk="low",
                priority=priority,
            ))
            priority += 1

        return sorted(suggestions, key=lambda s: s.priority)
