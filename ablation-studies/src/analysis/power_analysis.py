"""Statistical power analysis for ablation studies."""

from __future__ import annotations

import math
from typing import Optional


def required_repetitions(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Estimate the number of repetitions needed to detect an effect.

    Uses the formula: n = (z_alpha + z_beta)^2 / d^2
    where d is the standardized effect size (Cohen's d).

    Args:
        effect_size: Expected Cohen's d.
        alpha: Significance level.
        power: Desired statistical power.

    Returns:
        Required number of repetitions per condition.
    """
    if effect_size == 0:
        return 999  # Cannot detect zero effect

    z_alpha = _z_score(1 - alpha / 2)
    z_beta = _z_score(power)

    n = math.ceil((z_alpha + z_beta) ** 2 / (effect_size ** 2))
    return max(n, 2)  # Need at least 2


def achieved_power(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
) -> float:
    """Compute the achieved statistical power for a given sample size.

    Args:
        effect_size: Observed or expected Cohen's d.
        n: Sample size per condition.
        alpha: Significance level.

    Returns:
        Achieved power (0 to 1).
    """
    if n < 2 or effect_size == 0:
        return 0.0

    z_alpha = _z_score(1 - alpha / 2)
    # Non-centrality parameter
    lambda_nc = abs(effect_size) * math.sqrt(n)

    # Power = P(Z > z_alpha - lambda)
    z_power = lambda_nc - z_alpha
    power = _normal_cdf(z_power)

    return min(max(power, 0.0), 1.0)


def minimum_detectable_effect(
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Compute the minimum detectable effect size for a given sample size.

    Args:
        n: Sample size per condition.
        alpha: Significance level.
        power: Desired power.

    Returns:
        Minimum Cohen's d detectable.
    """
    if n < 2:
        return float("inf")

    z_alpha = _z_score(1 - alpha / 2)
    z_beta = _z_score(power)

    d = (z_alpha + z_beta) / math.sqrt(n)
    return d


def _z_score(p: float) -> float:
    """Approximate the z-score for a given cumulative probability.

    Uses the rational approximation by Abramowitz and Stegun.
    """
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p == 0.5:
        return 0.0

    if p > 0.5:
        return -_z_score(1 - p)

    t = math.sqrt(-2 * math.log(p))
    # Coefficients for the rational approximation
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
    return -z


def _normal_cdf(z: float) -> float:
    """Approximate the standard normal CDF using the error function."""
    return 0.5 * (1 + _erf(z / math.sqrt(2)))


def _erf(x: float) -> float:
    """Approximate the error function using Horner's method."""
    # Save the sign
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # Abramowitz and Stegun approximation 7.1.26
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return sign * y
