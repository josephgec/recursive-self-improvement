"""Cost forecasting for budget planning.

Predicts future costs based on spending history and remaining work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CostForecast:
    """Forecast of future costs."""
    estimated_total_cost: float
    estimated_remaining_cost: float
    burn_rate_per_iteration: float
    iterations_until_exhaustion: Optional[int]
    budget_remaining: float
    budget_sufficient: bool
    confidence: float  # 0-1
    alert_level: str  # "none", "info", "warning", "critical"

    @property
    def fraction_remaining(self) -> float:
        if self.estimated_total_cost > 0:
            return self.budget_remaining / self.estimated_total_cost
        return 1.0


class CostForecaster:
    """Forecasts future costs based on spending history.

    Uses linear extrapolation from historical spending data.
    """

    def __init__(self, alert_threshold: float = 0.20):
        """
        Args:
            alert_threshold: Fraction of budget remaining to trigger alert.
        """
        self.alert_threshold = alert_threshold
        self._forecast_history: List[CostForecast] = []

    def forecast(
        self,
        history: List[float],
        remaining_iterations: int,
        total_budget: float,
        spent_so_far: float = 0.0,
    ) -> CostForecast:
        """Forecast costs based on spending history.

        Args:
            history: List of per-iteration costs.
            remaining_iterations: How many iterations remain.
            total_budget: Total budget available.
            spent_so_far: Amount already spent.

        Returns:
            CostForecast with projections.
        """
        if not history:
            return CostForecast(
                estimated_total_cost=0.0,
                estimated_remaining_cost=0.0,
                burn_rate_per_iteration=0.0,
                iterations_until_exhaustion=None,
                budget_remaining=total_budget - spent_so_far,
                budget_sufficient=True,
                confidence=0.0,
                alert_level="none",
            )

        # Calculate burn rate
        burn_rate = sum(history) / len(history)

        # Estimate remaining cost
        estimated_remaining = burn_rate * remaining_iterations
        estimated_total = spent_so_far + estimated_remaining

        # Budget remaining
        budget_remaining = total_budget - spent_so_far

        # Iterations until exhaustion
        if burn_rate > 0:
            iterations_until_exhaustion = int(budget_remaining / burn_rate)
        else:
            iterations_until_exhaustion = None

        # Budget sufficient?
        budget_sufficient = estimated_remaining <= budget_remaining

        # Confidence based on history length
        confidence = min(len(history) / 20.0, 1.0)

        # Alert level
        fraction_remaining = budget_remaining / total_budget if total_budget > 0 else 1.0
        alert_level = self._compute_alert_level(fraction_remaining, budget_sufficient)

        forecast = CostForecast(
            estimated_total_cost=estimated_total,
            estimated_remaining_cost=estimated_remaining,
            burn_rate_per_iteration=burn_rate,
            iterations_until_exhaustion=iterations_until_exhaustion,
            budget_remaining=budget_remaining,
            budget_sufficient=budget_sufficient,
            confidence=confidence,
            alert_level=alert_level,
        )
        self._forecast_history.append(forecast)
        return forecast

    def should_alert(self, forecast: CostForecast) -> bool:
        """Determine if an alert should be raised.

        Args:
            forecast: A cost forecast.

        Returns:
            True if budget remaining fraction is below alert threshold.
        """
        if forecast.estimated_total_cost > 0:
            fraction = forecast.budget_remaining / forecast.estimated_total_cost
        else:
            fraction = 1.0

        return fraction <= self.alert_threshold or not forecast.budget_sufficient

    def _compute_alert_level(
        self, fraction_remaining: float, budget_sufficient: bool
    ) -> str:
        """Compute alert level based on remaining budget fraction."""
        if fraction_remaining <= 0.05 or not budget_sufficient:
            return "critical"
        elif fraction_remaining <= self.alert_threshold:
            return "warning"
        elif fraction_remaining <= 0.40:
            return "info"
        else:
            return "none"

    def get_history(self) -> List[CostForecast]:
        """Return forecast history."""
        return list(self._forecast_history)
