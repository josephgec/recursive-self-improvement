"""Collapse forecaster - predicts likelihood and timeline of model collapse.

Uses template matching against known collapse baselines to estimate
similarity to collapse trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Mock collapse baselines for template matching
COLLAPSE_BASELINES = [
    {
        "name": "entropy_death",
        "description": "Gradual entropy decrease leading to mode collapse",
        "pattern": {"entropy_trend": "decreasing", "kl_trend": "increasing"},
        "entropy_trajectory": [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.1],
        "kl_trajectory": [0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0],
    },
    {
        "name": "sudden_collapse",
        "description": "Sudden catastrophic collapse after stable period",
        "pattern": {"entropy_trend": "stable_then_drop", "kl_trend": "spike"},
        "entropy_trajectory": [4.0, 3.9, 4.0, 3.9, 4.0, 2.0, 0.5, 0.1, 0.01],
        "kl_trajectory": [0.1, 0.1, 0.1, 0.1, 0.1, 3.0, 8.0, 15.0, 20.0],
    },
    {
        "name": "quality_drift",
        "description": "Slow quality degradation without obvious entropy signals",
        "pattern": {"entropy_trend": "slightly_decreasing", "kl_trend": "slowly_increasing"},
        "entropy_trajectory": [4.0, 3.9, 3.8, 3.7, 3.5, 3.3, 3.0, 2.8, 2.5],
        "kl_trajectory": [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2],
    },
]


@dataclass
class CollapseForecast:
    """Forecast of collapse likelihood and timeline."""
    collapse_probability: float
    estimated_iterations_to_collapse: Optional[int]
    most_similar_baseline: str
    similarity_score: float
    risk_level: str  # "low", "medium", "high", "critical"
    recommendations: List[str] = field(default_factory=list)

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level in ("high", "critical")


class CollapseForecaster:
    """Forecasts model collapse using template matching against baselines."""

    def __init__(self, baselines: Optional[List[Dict[str, Any]]] = None):
        self.baselines = baselines or COLLAPSE_BASELINES
        self._forecast_history: List[CollapseForecast] = []

    def forecast(self, metrics: Dict[str, List[float]]) -> CollapseForecast:
        """Forecast collapse based on current metrics trajectory.

        Args:
            metrics: Dict with 'entropy' and/or 'kl_divergence' lists.

        Returns:
            CollapseForecast with risk assessment.
        """
        best_similarity = 0.0
        best_baseline = "none"

        for baseline in self.baselines:
            sim = self.similarity_to_collapse(metrics, baseline)
            if sim > best_similarity:
                best_similarity = sim
                best_baseline = baseline["name"]

        probability = min(best_similarity, 1.0)

        if probability >= 0.8:
            risk_level = "critical"
            est_iters = 5
        elif probability >= 0.5:
            risk_level = "high"
            est_iters = 20
        elif probability >= 0.3:
            risk_level = "medium"
            est_iters = 50
        else:
            risk_level = "low"
            est_iters = None

        recommendations = self._generate_recommendations(risk_level, metrics)

        forecast = CollapseForecast(
            collapse_probability=probability,
            estimated_iterations_to_collapse=est_iters,
            most_similar_baseline=best_baseline,
            similarity_score=best_similarity,
            risk_level=risk_level,
            recommendations=recommendations,
        )
        self._forecast_history.append(forecast)
        return forecast

    def similarity_to_collapse(
        self,
        metrics: Dict[str, List[float]],
        baseline: Dict[str, Any],
    ) -> float:
        """Compute similarity between current metrics and a collapse baseline.

        Uses normalized correlation between metric trajectories.

        Args:
            metrics: Dict with 'entropy' and/or 'kl_divergence' lists.
            baseline: A collapse baseline dict.

        Returns:
            Similarity score in [0, 1].
        """
        scores = []

        entropy = metrics.get("entropy", [])
        if entropy and "entropy_trajectory" in baseline:
            score = self._trajectory_similarity(
                entropy, baseline["entropy_trajectory"]
            )
            scores.append(score)

        kl = metrics.get("kl_divergence", [])
        if kl and "kl_trajectory" in baseline:
            score = self._trajectory_similarity(kl, baseline["kl_trajectory"])
            scores.append(score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def _trajectory_similarity(
        self, observed: List[float], reference: List[float]
    ) -> float:
        """Compute similarity between two trajectories.

        Truncates to same length, normalizes, and computes correlation.
        """
        min_len = min(len(observed), len(reference))
        if min_len < 2:
            return 0.0

        obs = observed[:min_len]
        ref = reference[:min_len]

        obs_range = max(obs) - min(obs)
        ref_range = max(ref) - min(ref)

        if obs_range == 0 or ref_range == 0:
            return 0.0

        obs_norm = [(x - min(obs)) / obs_range for x in obs]
        ref_norm = [(x - min(ref)) / ref_range for x in ref]

        # Pearson-like correlation
        n = len(obs_norm)
        mean_obs = sum(obs_norm) / n
        mean_ref = sum(ref_norm) / n

        cov = sum((o - mean_obs) * (r - mean_ref) for o, r in zip(obs_norm, ref_norm))
        var_obs = sum((o - mean_obs) ** 2 for o in obs_norm)
        var_ref = sum((r - mean_ref) ** 2 for r in ref_norm)

        if var_obs == 0 or var_ref == 0:
            return 0.0

        correlation = cov / ((var_obs * var_ref) ** 0.5)
        return max(0.0, correlation)

    def _generate_recommendations(
        self, risk_level: str, metrics: Dict[str, List[float]]
    ) -> List[str]:
        """Generate recommendations based on risk level."""
        recs = []
        if risk_level == "critical":
            recs.append("HALT: Immediately stop training and diagnose")
            recs.append("Increase alpha to maximum (more clean data)")
            recs.append("Consider rolling back to last known good checkpoint")
        elif risk_level == "high":
            recs.append("Increase alpha significantly")
            recs.append("Reduce synthetic data fraction")
            recs.append("Monitor entropy closely")
        elif risk_level == "medium":
            recs.append("Consider increasing alpha")
            recs.append("Run additional eval benchmarks")
        else:
            recs.append("Continue monitoring")
        return recs

    def get_history(self) -> List[CollapseForecast]:
        """Return forecast history."""
        return list(self._forecast_history)
