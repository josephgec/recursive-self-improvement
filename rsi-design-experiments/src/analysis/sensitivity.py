"""Sensitivity analysis: how much does each design decision matter?"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.experiments.base import ExperimentResult


@dataclass
class SensitivityResult:
    """Sensitivity analysis result for one experiment."""

    experiment_name: str
    sensitivity: float  # (max - min) / mean
    metric: str
    max_value: float
    min_value: float
    mean_value: float


class SensitivityAnalyzer:
    """Analyzes sensitivity of outcomes to design decisions."""

    def compute_sensitivity(
        self,
        experiment_result: ExperimentResult,
        metric: str = "composite_score",
    ) -> SensitivityResult:
        """Compute sensitivity for an experiment.

        Sensitivity = (max_mean - min_mean) / grand_mean across conditions.
        Higher sensitivity means the design decision matters more.
        """
        condition_means = []
        for cond_name, cond_results in experiment_result.per_condition_results.items():
            values = [getattr(r, metric, 0.0) for r in cond_results]
            if values:
                condition_means.append(sum(values) / len(values))

        if not condition_means:
            return SensitivityResult(
                experiment_name=experiment_result.experiment_name,
                sensitivity=0.0,
                metric=metric,
                max_value=0.0,
                min_value=0.0,
                mean_value=0.0,
            )

        max_val = max(condition_means)
        min_val = min(condition_means)
        mean_val = sum(condition_means) / len(condition_means)

        sensitivity = (max_val - min_val) / mean_val if mean_val != 0 else 0.0

        return SensitivityResult(
            experiment_name=experiment_result.experiment_name,
            sensitivity=sensitivity,
            metric=metric,
            max_value=max_val,
            min_value=min_val,
            mean_value=mean_val,
        )

    def rank_experiments(
        self,
        all_results: List[ExperimentResult],
        metric: str = "composite_score",
    ) -> List[SensitivityResult]:
        """Rank experiments by sensitivity (highest first).

        This tells us which design decisions matter most.
        """
        sensitivities = [
            self.compute_sensitivity(result, metric) for result in all_results
        ]
        sensitivities.sort(key=lambda s: s.sensitivity, reverse=True)
        return sensitivities
