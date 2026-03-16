"""Base experiment class and result dataclasses."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConditionResult:
    """Results from running a single condition."""

    condition_name: str
    accuracy_trajectory: List[float] = field(default_factory=list)
    final_accuracy: float = 0.0
    stability_score: float = 0.0
    rollback_count: int = 0
    total_cost: float = 0.0
    improvement_rate: float = 0.0
    generalization_score: float = 0.0
    composite_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Aggregated results from an experiment across all conditions and repetitions."""

    experiment_name: str
    conditions: List[str]
    per_condition_results: Dict[str, List[ConditionResult]] = field(default_factory=dict)
    repetitions: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_mean_accuracy(self, condition: str) -> float:
        """Get mean final accuracy across repetitions for a condition."""
        results = self.per_condition_results.get(condition, [])
        if not results:
            return 0.0
        return sum(r.final_accuracy for r in results) / len(results)

    def get_mean_composite(self, condition: str) -> float:
        """Get mean composite score across repetitions for a condition."""
        results = self.per_condition_results.get(condition, [])
        if not results:
            return 0.0
        return sum(r.composite_score for r in results) / len(results)

    def get_all_scores(self, metric: str = "final_accuracy") -> Dict[str, List[float]]:
        """Get all scores for a metric, grouped by condition."""
        result = {}
        for cond, cond_results in self.per_condition_results.items():
            result[cond] = [getattr(r, metric, 0.0) for r in cond_results]
        return result


class Experiment(ABC):
    """Abstract base class for all experiments."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def get_conditions(self) -> List[Any]:
        """Return the list of conditions to test."""
        ...

    @abstractmethod
    def configure_pipeline(self, condition: Any, pipeline: Any) -> Any:
        """Configure the pipeline for a specific condition."""
        ...

    @abstractmethod
    def measure(self, pipeline: Any, condition: Any) -> ConditionResult:
        """Measure the pipeline's performance under a condition."""
        ...

    def run(self, pipeline: Any, repetitions: int = 5, seed: int = 42) -> ExperimentResult:
        """Run the experiment across all conditions and repetitions."""
        conditions = self.get_conditions()
        condition_names = [c.name for c in conditions]
        result = ExperimentResult(
            experiment_name=self.name,
            conditions=condition_names,
            repetitions=repetitions,
        )

        for condition in conditions:
            condition_results = []
            for rep in range(repetitions):
                rep_seed = seed + rep
                configured = self.configure_pipeline(condition, pipeline)
                configured.set_seed(rep_seed)
                measurement = self.measure(configured, condition)
                condition_results.append(measurement)
            result.per_condition_results[condition.name] = condition_results

        return result
