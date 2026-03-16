"""Recommendation generation from experiment results."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from src.experiments.base import ExperimentResult
from src.analysis.optimal_finder import OptimalFinder
from src.analysis.sensitivity import SensitivityAnalyzer


@dataclass
class PipelineRecommendation:
    """A complete pipeline configuration recommendation."""

    modification_frequency: str = ""
    hindsight_target: str = ""
    rlm_depth: int = 0
    confidence_levels: Dict[str, str] = field(default_factory=dict)
    reasoning: Dict[str, str] = field(default_factory=dict)
    sensitivity_ranking: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "pipeline_config": {
                "modification_frequency": self.modification_frequency,
                "hindsight_target": self.hindsight_target,
                "rlm_depth": self.rlm_depth,
            },
            "confidence_levels": self.confidence_levels,
            "reasoning": self.reasoning,
            "sensitivity_ranking": self.sensitivity_ranking,
            "metadata": self.metadata,
        }

    def to_yaml(self) -> str:
        """Export as YAML config string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


class RecommendationGenerator:
    """Generates pipeline recommendations from experiment results."""

    def __init__(self):
        self._finder = OptimalFinder()
        self._sensitivity = SensitivityAnalyzer()

    def generate(
        self,
        all_results: List[ExperimentResult],
        metric: str = "composite_score",
    ) -> PipelineRecommendation:
        """Generate a pipeline recommendation from all experiment results.

        Args:
            all_results: list of ExperimentResult, one per experiment.
            metric: metric to optimize.

        Returns:
            PipelineRecommendation with optimal settings and confidence levels.
        """
        recommendation = PipelineRecommendation()

        # Sensitivity ranking
        sensitivities = self._sensitivity.rank_experiments(all_results, metric)
        recommendation.sensitivity_ranking = [s.experiment_name for s in sensitivities]

        # Find best condition for each experiment
        for result in all_results:
            best = self._finder.find_best_condition(result, metric)
            exp_name = result.experiment_name

            # Compute confidence based on how much better the best is
            scores = list(best.all_scores.values())
            if len(scores) > 1 and best.best_score > 0:
                second_best = sorted(scores, reverse=True)[1]
                margin = (best.best_score - second_best) / best.best_score
                if margin > 0.1:
                    confidence = "high"
                elif margin > 0.03:
                    confidence = "medium"
                else:
                    confidence = "low"
            else:
                confidence = "low"

            if "frequency" in exp_name or "modification" in exp_name:
                recommendation.modification_frequency = best.best_condition
                recommendation.confidence_levels["modification_frequency"] = confidence
                recommendation.reasoning["modification_frequency"] = (
                    f"Best condition: {best.best_condition} "
                    f"(score: {best.best_score:.4f})"
                )
            elif "hindsight" in exp_name:
                recommendation.hindsight_target = best.best_condition
                recommendation.confidence_levels["hindsight_target"] = confidence
                recommendation.reasoning["hindsight_target"] = (
                    f"Best condition: {best.best_condition} "
                    f"(score: {best.best_score:.4f})"
                )
            elif "depth" in exp_name or "rlm" in exp_name:
                # Extract depth number from condition name
                depth_str = best.best_condition.replace("depth_", "")
                try:
                    recommendation.rlm_depth = int(depth_str)
                except ValueError:
                    recommendation.rlm_depth = 0
                recommendation.confidence_levels["rlm_depth"] = confidence
                recommendation.reasoning["rlm_depth"] = (
                    f"Best condition: {best.best_condition} "
                    f"(score: {best.best_score:.4f})"
                )

        return recommendation
