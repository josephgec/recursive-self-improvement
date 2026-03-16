"""Finding optimal conditions and Pareto-optimal points."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.experiments.base import ExperimentResult


@dataclass
class OptimalResult:
    """Result of finding the optimal condition."""

    best_condition: str
    best_score: float
    all_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""

    condition: str
    scores: Dict[str, float] = field(default_factory=dict)


class OptimalFinder:
    """Finds optimal and Pareto-optimal conditions from experiment results."""

    def find_best_condition(
        self, result: ExperimentResult, metric: str = "composite_score"
    ) -> OptimalResult:
        """Find the condition with the best mean score on a given metric.

        Args:
            result: Experiment result with per-condition data.
            metric: Name of the metric attribute on ConditionResult.
        """
        all_scores = {}
        for cond_name, cond_results in result.per_condition_results.items():
            values = [getattr(r, metric, 0.0) for r in cond_results]
            all_scores[cond_name] = sum(values) / len(values) if values else 0.0

        best_cond = max(all_scores, key=all_scores.get)  # type: ignore
        return OptimalResult(
            best_condition=best_cond,
            best_score=all_scores[best_cond],
            all_scores=all_scores,
        )

    def find_pareto_optimal(
        self,
        result: ExperimentResult,
        metrics: List[str],
    ) -> List[ParetoPoint]:
        """Find Pareto-optimal conditions across multiple metrics.

        A condition is Pareto-optimal if no other condition dominates it
        (i.e., is better on all metrics).
        """
        # Compute mean scores per condition per metric
        condition_scores: Dict[str, Dict[str, float]] = {}
        for cond_name, cond_results in result.per_condition_results.items():
            scores = {}
            for metric in metrics:
                values = [getattr(r, metric, 0.0) for r in cond_results]
                scores[metric] = sum(values) / len(values) if values else 0.0
            condition_scores[cond_name] = scores

        # Find Pareto-optimal points
        pareto = []
        conditions = list(condition_scores.keys())

        for cond in conditions:
            dominated = False
            for other in conditions:
                if other == cond:
                    continue
                # Check if 'other' dominates 'cond'
                all_better = all(
                    condition_scores[other][m] >= condition_scores[cond][m]
                    for m in metrics
                )
                any_strictly_better = any(
                    condition_scores[other][m] > condition_scores[cond][m]
                    for m in metrics
                )
                if all_better and any_strictly_better:
                    dominated = True
                    break
            if not dominated:
                pareto.append(
                    ParetoPoint(
                        condition=cond,
                        scores=condition_scores[cond],
                    )
                )

        return pareto

    def recommend(
        self,
        result: ExperimentResult,
        primary_metric: str = "composite_score",
        secondary_metrics: Optional[List[str]] = None,
    ) -> str:
        """Generate a recommendation based on the experiment results."""
        best = self.find_best_condition(result, primary_metric)

        if secondary_metrics:
            pareto = self.find_pareto_optimal(
                result, [primary_metric] + secondary_metrics
            )
            pareto_names = [p.condition for p in pareto]
            if best.best_condition in pareto_names:
                return best.best_condition
            # If best on primary is not Pareto-optimal, pick the Pareto point
            # with the highest primary metric
            best_pareto = max(
                pareto, key=lambda p: p.scores.get(primary_metric, 0.0)
            )
            return best_pareto.condition

        return best.best_condition
