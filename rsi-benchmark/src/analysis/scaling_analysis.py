"""Scaling analysis: fit scaling laws and project forward."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ScalingLaw:
    """A fitted scaling law."""
    law_type: str  # "power", "logarithmic", "linear"
    params: Dict[str, float]
    r_squared: float
    domain: Tuple[float, float]  # (min_iter, max_iter) fitted range

    def predict(self, iteration: float) -> float:
        if self.law_type == "linear":
            a = self.params.get("a", 0.0)
            b = self.params.get("b", 0.0)
            return a * iteration + b
        elif self.law_type == "logarithmic":
            a = self.params.get("a", 0.0)
            b = self.params.get("b", 0.0)
            return a * math.log(max(iteration, 1e-10)) + b
        elif self.law_type == "power":
            a = self.params.get("a", 0.0)
            b = self.params.get("b", 0.5)
            c = self.params.get("c", 0.0)
            return a * (iteration ** b) + c
        return 0.0


def fit_scaling_law(
    iterations: List[int],
    accuracies: List[float],
    law_type: str = "logarithmic",
) -> ScalingLaw:
    """Fit a scaling law to iteration-accuracy data."""
    if not iterations or not accuracies:
        return ScalingLaw(law_type=law_type, params={}, r_squared=0.0, domain=(0, 0))

    xs = [float(x) for x in iterations]
    ys = list(accuracies)
    n = len(xs)

    if law_type == "linear":
        a, b = _linear_regression(xs, ys)
        params = {"a": a, "b": b}
        preds = [a * x + b for x in xs]
    elif law_type == "logarithmic":
        log_xs = [math.log(max(x, 1e-10)) for x in xs]
        a, b = _linear_regression(log_xs, ys)
        params = {"a": a, "b": b}
        preds = [a * lx + b for lx in log_xs]
    elif law_type == "power":
        # Use log-log for power law
        pos_xs = [max(x, 1e-10) for x in xs]
        min_y = min(ys)
        offset = 0.0
        if min_y <= 0:
            offset = abs(min_y) + 0.01
        shifted_ys = [y + offset for y in ys]
        log_xs = [math.log(x) for x in pos_xs]
        log_ys = [math.log(max(y, 1e-10)) for y in shifted_ys]
        b, log_a = _linear_regression(log_xs, log_ys)
        a = math.exp(log_a)
        c = -offset
        params = {"a": a, "b": b, "c": c}
        preds = [a * (x ** b) + c for x in pos_xs]
    else:
        raise ValueError(f"Unknown law type: {law_type}")

    r_sq = _r_squared(ys, preds)
    domain = (min(xs), max(xs))

    return ScalingLaw(
        law_type=law_type, params=params, r_squared=r_sq, domain=domain,
    )


def optimal_iterations(
    scaling_law: ScalingLaw,
    target_accuracy: float,
    max_iterations: int = 100,
) -> Optional[int]:
    """Find the optimal number of iterations to reach target accuracy."""
    for i in range(1, max_iterations + 1):
        if scaling_law.predict(float(i)) >= target_accuracy:
            return i
    return None


def project_forward(
    scaling_law: ScalingLaw,
    num_extra_iterations: int,
) -> List[Tuple[int, float]]:
    """Project the scaling law forward."""
    start = int(scaling_law.domain[1]) + 1
    projections = []
    for i in range(num_extra_iterations):
        iteration = start + i
        projected = scaling_law.predict(float(iteration))
        projections.append((iteration, projected))
    return projections


def _linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float]:
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


def _r_squared(actual: List[float], predicted: List[float]) -> float:
    if len(actual) < 2:
        return 0.0
    mean_y = sum(actual) / len(actual)
    ss_tot = sum((y - mean_y) ** 2 for y in actual)
    ss_res = sum((y - p) ** 2 for y, p in zip(actual, predicted))
    if ss_tot < 1e-10:
        return 1.0 if ss_res < 1e-10 else 0.0
    return 1.0 - ss_res / ss_tot
