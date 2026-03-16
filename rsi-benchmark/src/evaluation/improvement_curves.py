"""Improvement curve tracking, growth model fitting, and analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GrowthModel:
    """Fitted growth model parameters."""
    model_type: str  # logarithmic, power, sigmoid
    params: Dict[str, float]
    r_squared: float
    residuals: List[float] = field(default_factory=list)

    def predict(self, x: float) -> float:
        if self.model_type == "logarithmic":
            a = self.params.get("a", 1.0)
            b = self.params.get("b", 0.0)
            return a * math.log(max(x, 1e-10)) + b
        elif self.model_type == "power":
            a = self.params.get("a", 1.0)
            b = self.params.get("b", 0.5)
            c = self.params.get("c", 0.0)
            return a * (x ** b) + c
        elif self.model_type == "sigmoid":
            l = self.params.get("L", 1.0)
            k = self.params.get("k", 1.0)
            x0 = self.params.get("x0", 5.0)
            return l / (1 + math.exp(-k * (x - x0)))
        return 0.0


class ImprovementCurveTracker:
    """Track and analyze improvement curves across iterations."""

    def __init__(self) -> None:
        self._curves: Dict[str, List[Tuple[int, float]]] = {}

    def record(self, benchmark: str, iteration: int, accuracy: float) -> None:
        """Record an accuracy measurement."""
        if benchmark not in self._curves:
            self._curves[benchmark] = []
        self._curves[benchmark].append((iteration, accuracy))
        # Keep sorted by iteration
        self._curves[benchmark].sort(key=lambda x: x[0])

    def get_curve(self, benchmark: str) -> List[Tuple[int, float]]:
        """Get the improvement curve for a benchmark."""
        return list(self._curves.get(benchmark, []))

    def get_all_curves(self) -> Dict[str, List[Tuple[int, float]]]:
        """Get all improvement curves."""
        return {k: list(v) for k, v in self._curves.items()}

    def is_improving(self, benchmark: str, window: int = 3) -> bool:
        """Check if the benchmark is showing improvement."""
        curve = self._curves.get(benchmark, [])
        if len(curve) < window:
            return False
        recent = [v for _, v in curve[-window:]]
        # Check if trend is upward
        improvements = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        return improvements >= (window - 1) / 2

    def is_degrading(self, benchmark: str, window: int = 3) -> bool:
        """Check if the benchmark is showing degradation."""
        curve = self._curves.get(benchmark, [])
        if len(curve) < window:
            return False
        recent = [v for _, v in curve[-window:]]
        degradations = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])
        return degradations >= (window - 1) / 2

    def is_plateaued(self, benchmark: str, window: int = 3, tolerance: float = 0.005) -> bool:
        """Check if the benchmark has plateaued."""
        curve = self._curves.get(benchmark, [])
        if len(curve) < window:
            return False
        recent = [v for _, v in curve[-window:]]
        max_diff = max(recent) - min(recent)
        return max_diff <= tolerance

    def compute_sustained_improvement(self, benchmark: str) -> int:
        """Compute the longest streak of sustained improvement."""
        curve = self._curves.get(benchmark, [])
        if len(curve) < 2:
            return 0
        values = [v for _, v in curve]
        max_streak = 0
        current_streak = 0
        for i in range(1, len(values)):
            if values[i] > values[i - 1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def compute_total_improvement(self, benchmark: str) -> float:
        """Compute total improvement from first to last measurement."""
        curve = self._curves.get(benchmark, [])
        if len(curve) < 2:
            return 0.0
        first = curve[0][1]
        last = curve[-1][1]
        return last - first

    def fit_growth_model(
        self,
        benchmark: str,
        model_type: str = "logarithmic",
    ) -> GrowthModel:
        """Fit a growth model to the improvement curve.

        Supports: logarithmic, power, sigmoid.
        Uses simple least-squares fitting.
        """
        curve = self._curves.get(benchmark, [])
        if not curve:
            return GrowthModel(model_type=model_type, params={}, r_squared=0.0)

        xs = [float(x) for x, _ in curve]
        ys = [y for _, y in curve]
        n = len(xs)

        if model_type == "logarithmic":
            params, r_sq, residuals = self._fit_logarithmic(xs, ys)
        elif model_type == "power":
            params, r_sq, residuals = self._fit_power(xs, ys)
        elif model_type == "sigmoid":
            params, r_sq, residuals = self._fit_sigmoid(xs, ys)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return GrowthModel(
            model_type=model_type,
            params=params,
            r_squared=r_sq,
            residuals=residuals,
        )

    def _fit_logarithmic(
        self, xs: List[float], ys: List[float]
    ) -> Tuple[Dict[str, float], float, List[float]]:
        """Fit y = a * ln(x) + b."""
        log_xs = [math.log(max(x, 1e-10)) for x in xs]
        a, b = self._linear_regression(log_xs, ys)
        predictions = [a * lx + b for lx in log_xs]
        r_sq = self._r_squared(ys, predictions)
        residuals = [y - p for y, p in zip(ys, predictions)]
        return {"a": a, "b": b}, r_sq, residuals

    def _fit_power(
        self, xs: List[float], ys: List[float]
    ) -> Tuple[Dict[str, float], float, List[float]]:
        """Fit y = a * x^b + c. Simplified: use log-log regression."""
        # Shift ys to be positive for log transform
        min_y = min(ys)
        offset = 0.0
        if min_y <= 0:
            offset = abs(min_y) + 0.01
        shifted_ys = [y + offset for y in ys]

        log_xs = [math.log(max(x, 1e-10)) for x in xs]
        log_ys = [math.log(max(y, 1e-10)) for y in shifted_ys]

        b, log_a = self._linear_regression(log_xs, log_ys)
        a = math.exp(log_a)
        c = -offset

        predictions = [a * (x ** b) + c for x in xs]
        r_sq = self._r_squared(ys, predictions)
        residuals = [y - p for y, p in zip(ys, predictions)]
        return {"a": a, "b": b, "c": c}, r_sq, residuals

    def _fit_sigmoid(
        self, xs: List[float], ys: List[float]
    ) -> Tuple[Dict[str, float], float, List[float]]:
        """Fit y = L / (1 + exp(-k*(x - x0))). Use heuristic parameters."""
        L = max(ys) * 1.1  # Estimated asymptote
        x0 = xs[len(xs) // 2]  # Midpoint
        # Estimate k from steepest slope
        max_slope = 0.0
        for i in range(1, len(xs)):
            slope = (ys[i] - ys[i - 1]) / max(xs[i] - xs[i - 1], 1e-10)
            if abs(slope) > abs(max_slope):
                max_slope = slope
        k = 4 * max_slope / max(L, 1e-10)  # Sigmoid slope at midpoint is L*k/4
        if k == 0:
            k = 0.5

        predictions = [L / (1 + math.exp(-k * (x - x0))) for x in xs]
        r_sq = self._r_squared(ys, predictions)
        residuals = [y - p for y, p in zip(ys, predictions)]
        return {"L": L, "k": k, "x0": x0}, r_sq, residuals

    @staticmethod
    def _linear_regression(
        xs: List[float], ys: List[float]
    ) -> Tuple[float, float]:
        """Simple linear regression: y = a*x + b. Returns (a, b)."""
        n = len(xs)
        if n == 0:
            return 0.0, 0.0
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_x2 = sum(x * x for x in xs)

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-10:
            return 0.0, sum_y / max(n, 1)

        a = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - a * sum_x) / n
        return a, b

    @staticmethod
    def _r_squared(actual: List[float], predicted: List[float]) -> float:
        """Compute R-squared goodness of fit."""
        if len(actual) < 2:
            return 0.0
        mean_y = sum(actual) / len(actual)
        ss_tot = sum((y - mean_y) ** 2 for y in actual)
        ss_res = sum((y - p) ** 2 for y, p in zip(actual, predicted))
        if ss_tot < 1e-10:
            return 1.0 if ss_res < 1e-10 else 0.0
        return 1.0 - ss_res / ss_tot

    def plot_improvement_curves(self) -> Dict[str, Any]:
        """Generate plot data for improvement curves (returns dict, not actual plot)."""
        plot_data: Dict[str, Any] = {}
        for benchmark, curve in self._curves.items():
            xs = [x for x, _ in curve]
            ys = [y for _, y in curve]
            plot_data[benchmark] = {
                "iterations": xs,
                "accuracy": ys,
                "total_improvement": self.compute_total_improvement(benchmark),
                "sustained_improvement": self.compute_sustained_improvement(benchmark),
            }
        return plot_data
