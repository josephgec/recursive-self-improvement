"""Modification frequency experiment: how often should the pipeline self-modify?"""

from typing import Any, Dict, List, Optional

from src.experiments.base import Experiment, ConditionResult
from src.conditions.frequency_conditions import (
    FrequencyCondition,
    ModificationFrequencyPolicy,
    build_frequency_conditions,
)
from src.measurement.accuracy_tracker import AccuracyTracker
from src.measurement.stability_tracker import StabilityTracker
from src.measurement.cost_tracker import CostTracker
from src.measurement.improvement_rate import ImprovementRateTracker
from src.measurement.composite_scorer import CompositeScorer


class ModificationFrequencyExperiment(Experiment):
    """Experiment testing different modification frequency policies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "modification_frequency"
        self.iterations = (config or {}).get("iterations_per_condition", 20)

    def get_conditions(self) -> List[FrequencyCondition]:
        return build_frequency_conditions()

    def configure_pipeline(self, condition: FrequencyCondition, pipeline: Any) -> Any:
        """Configure the pipeline with this condition's modification policy."""
        pipeline.set_modification_policy(condition.policy)
        pipeline.set_condition_name(condition.name)
        return pipeline

    def measure(self, pipeline: Any, condition: FrequencyCondition) -> ConditionResult:
        """Run the pipeline and measure accuracy, stability, cost."""
        accuracy_tracker = AccuracyTracker()
        stability_tracker = StabilityTracker()
        cost_tracker = CostTracker()
        improvement_tracker = ImprovementRateTracker()
        scorer = CompositeScorer()

        condition.policy.reset()

        for i in range(self.iterations):
            step_result = pipeline.step(i)

            accuracy_tracker.record(
                step_result["accuracy"],
                step_result.get("accuracy_type", "in_distribution"),
            )
            if step_result.get("rollback", False):
                stability_tracker.record_rollback(i)
            cost_tracker.record_llm_call(step_result.get("llm_cost", 0.01))
            if step_result.get("finetuning_cost", 0):
                cost_tracker.record_finetuning(step_result["finetuning_cost"])
            improvement_tracker.record(step_result["accuracy"])

        overall_acc = accuracy_tracker.get_overall()
        stab_score = stability_tracker.stability_score(self.iterations)
        total_cost = cost_tracker.get_total_cost()
        imp_rate = improvement_tracker.compute_rolling_delta()

        # Generalization: OOD accuracy relative to ID accuracy
        ood_acc = accuracy_tracker.get_per_type("out_of_distribution")
        id_acc = accuracy_tracker.get_per_type("in_distribution")
        gen_score = ood_acc / id_acc if id_acc > 0 else 0.0

        # Efficiency: inverse of normalized cost
        max_cost = 10.0  # upper bound normalization
        efficiency = 1.0 - min(total_cost / max_cost, 1.0)

        composite = scorer.score(overall_acc, stab_score, efficiency, gen_score)

        return ConditionResult(
            condition_name=condition.name,
            accuracy_trajectory=accuracy_tracker.get_trajectory(),
            final_accuracy=overall_acc,
            stability_score=stab_score,
            rollback_count=stability_tracker.total_rollbacks(),
            total_cost=total_cost,
            improvement_rate=imp_rate,
            generalization_score=gen_score,
            composite_score=composite,
        )
