"""Benchmark suite: orchestrates evaluation across all axes and systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.evaluation.generalization import GeneralizationEvaluator, GeneralizationResult
from src.evaluation.interpretability import InterpretabilityEvaluator, InterpretabilityResult
from src.evaluation.robustness import RobustnessEvaluator, RobustnessResult
from src.utils.task_domains import MultiDomainTaskLoader, Task


@dataclass
class BenchmarkResults:
    """Results from a full benchmark run."""
    generalization: Dict[str, GeneralizationResult] = field(default_factory=dict)
    interpretability: Dict[str, InterpretabilityResult] = field(default_factory=dict)
    robustness: Dict[str, RobustnessResult] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary table of results."""
        summary: Dict[str, Dict[str, float]] = {}
        for system in set(
            list(self.generalization.keys())
            + list(self.interpretability.keys())
            + list(self.robustness.keys())
        ):
            summary[system] = {}
            if system in self.generalization:
                g = self.generalization[system]
                summary[system]["in_domain_accuracy"] = g.in_domain_accuracy
                summary[system]["out_of_domain_accuracy"] = g.out_of_domain_accuracy
                summary[system]["generalization_gap"] = g.generalization_gap
            if system in self.interpretability:
                i = self.interpretability[system]
                summary[system]["interpretability"] = i.overall_score
            if system in self.robustness:
                r = self.robustness[system]
                summary[system]["consistency"] = r.consistency
                summary[system]["degradation"] = r.degradation
        return summary


class BenchmarkSuite:
    """Orchestrates the full benchmark across systems and evaluation axes."""

    def __init__(self) -> None:
        self._systems: Dict[str, Any] = {}  # name -> pipeline
        self._gen_evaluator = GeneralizationEvaluator()
        self._interp_evaluator = InterpretabilityEvaluator()
        self._robust_evaluator = RobustnessEvaluator()
        self._task_loader = MultiDomainTaskLoader()

    def register_system(self, name: str, pipeline: Any) -> None:
        """Register a system for benchmarking."""
        self._systems[name] = pipeline

    @property
    def registered_systems(self) -> List[str]:
        return list(self._systems.keys())

    def run_full_suite(
        self,
        domains: Optional[List[str]] = None,
    ) -> BenchmarkResults:
        """Run the full benchmark suite across all registered systems.

        Args:
            domains: Optional list of domains to evaluate.

        Returns:
            BenchmarkResults with all evaluation results.
        """
        domains = domains or ["arithmetic", "algebra"]
        results = BenchmarkResults()

        for name, pipeline in self._systems.items():
            # Generalization
            gen_result = self._run_generalization(name, pipeline, domains)
            results.generalization[name] = gen_result

            # Interpretability
            interp_result = self._run_interpretability(name, pipeline, domains)
            results.interpretability[name] = interp_result

            # Robustness
            robust_result = self._run_robustness(name, pipeline, domains)
            results.robustness[name] = robust_result

        return results

    def run_single_axis(
        self,
        axis: str,
        domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run a single evaluation axis across all systems.

        Args:
            axis: One of "generalization", "interpretability", "robustness".
            domains: Optional list of domains.

        Returns:
            Dict mapping system name to axis-specific result.
        """
        domains = domains or ["arithmetic"]
        axis_results: Dict[str, Any] = {}

        for name, pipeline in self._systems.items():
            if axis == "generalization":
                axis_results[name] = self._run_generalization(name, pipeline, domains)
            elif axis == "interpretability":
                axis_results[name] = self._run_interpretability(name, pipeline, domains)
            elif axis == "robustness":
                axis_results[name] = self._run_robustness(name, pipeline, domains)
            else:
                raise ValueError(f"Unknown axis: {axis}")

        return axis_results

    def _run_generalization(
        self, name: str, pipeline: Any, domains: List[str]
    ) -> GeneralizationResult:
        """Run generalization evaluation."""
        train_tasks: List[Task] = []
        test_tasks: List[Task] = []

        for domain in domains:
            tasks = self._task_loader.load_domain(domain)
            mid = max(1, len(tasks) // 2)
            train_tasks.extend(tasks[:mid])
            test_tasks.extend(tasks[mid:])

        if not train_tasks:
            train_tasks = self._task_loader.load_domain("arithmetic")[:3]
        if not test_tasks:
            test_tasks = self._task_loader.load_domain("algebra")[:3]

        return self._gen_evaluator.evaluate(name, pipeline, train_tasks, test_tasks)

    def _run_interpretability(
        self, name: str, pipeline: Any, domains: List[str]
    ) -> InterpretabilityResult:
        """Run interpretability evaluation."""
        tasks: List[Task] = []
        for domain in domains:
            tasks.extend(self._task_loader.load_domain(domain)[:5])

        if not tasks:
            tasks = self._task_loader.load_domain("arithmetic")[:3]

        results = [pipeline.solve(t.problem) for t in tasks]
        return self._interp_evaluator.evaluate(name, results)

    def _run_robustness(
        self, name: str, pipeline: Any, domains: List[str]
    ) -> RobustnessResult:
        """Run robustness evaluation."""
        domain = domains[0] if domains else "arithmetic"
        originals, perturbed = self._task_loader.load_paired_perturbations(domain)

        if not originals:
            originals = self._task_loader.load_domain("arithmetic")[:3]
            perturbed = self._task_loader.load_domain("arithmetic")[:3]

        return self._robust_evaluator.evaluate(
            name, pipeline, originals, perturbed, perturbation_type="rephrase"
        )
