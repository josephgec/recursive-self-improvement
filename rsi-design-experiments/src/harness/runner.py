"""Experiment runner: orchestrates experiment execution."""

from typing import Any, Dict, List, Optional

from src.experiments.base import Experiment, ExperimentResult
from src.harness.controlled_pipeline import MockPipeline


class ExperimentRunner:
    """Runs experiments and collects results."""

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._results: Dict[str, ExperimentResult] = {}

    def run_experiment(
        self,
        experiment: Experiment,
        repetitions: int = 5,
    ) -> ExperimentResult:
        """Run a single experiment with the given number of repetitions."""
        pipeline = MockPipeline(self._seed)
        result = experiment.run(pipeline, repetitions=repetitions, seed=self._seed)
        self._results[result.experiment_name] = result
        return result

    def run_all(
        self,
        experiments: List[Experiment],
        repetitions: int = 5,
    ) -> List[ExperimentResult]:
        """Run all experiments and return results."""
        results = []
        for experiment in experiments:
            result = self.run_experiment(experiment, repetitions)
            results.append(result)
        return results

    def get_results(self) -> Dict[str, ExperimentResult]:
        """Get all stored results."""
        return dict(self._results)

    def get_result(self, experiment_name: str) -> Optional[ExperimentResult]:
        """Get result for a specific experiment."""
        return self._results.get(experiment_name)
