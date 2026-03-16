"""Paradigm ablation study: run ablation experiments across conditions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.ablation.conditions import AblationCondition, build_all_conditions, configure_pipeline_for_condition
from src.benchmarks.registry import BaseBenchmark, BenchmarkResult


@dataclass
class AblationRun:
    """A single ablation run (one condition, one benchmark, all iterations)."""
    condition: str
    benchmark: str
    iterations: List[int]
    accuracies: List[float]
    total_improvement: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationResult:
    """Complete ablation study result."""
    conditions: List[str]
    benchmarks: List[str]
    runs: Dict[str, Dict[str, AblationRun]]  # condition -> benchmark -> run
    summary: Dict[str, float] = field(default_factory=dict)  # condition -> avg improvement


class ParadigmAblationStudy:
    """Run paradigm ablation study across 7 conditions."""

    def __init__(
        self,
        agent_factory: Callable[[str], Any],
        num_iterations: int = 15,
        conditions: Optional[List[AblationCondition]] = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._num_iterations = num_iterations
        self._conditions = conditions or build_all_conditions()

    def run(
        self,
        benchmarks: Dict[str, BaseBenchmark],
    ) -> AblationResult:
        """Run the full ablation study."""
        condition_names = [c.name for c in self._conditions]
        benchmark_names = list(benchmarks.keys())
        runs: Dict[str, Dict[str, AblationRun]] = {}
        summary: Dict[str, float] = {}

        for condition in self._conditions:
            config = configure_pipeline_for_condition(condition)
            agent = self._agent_factory(condition.name)
            runs[condition.name] = {}

            condition_improvements = []

            for bm_name, benchmark in benchmarks.items():
                iterations_list: List[int] = []
                accuracies: List[float] = []
                tasks = benchmark.tasks

                for iteration in range(self._num_iterations):
                    # Set iteration on agent if supported
                    if hasattr(agent, "set_iteration"):
                        agent.set_iteration(iteration)

                    results = benchmark.evaluate(agent, tasks)
                    correct = sum(1 for r in results if r.correct)
                    accuracy = correct / len(results) if results else 0.0

                    iterations_list.append(iteration)
                    accuracies.append(accuracy)

                total_improvement = accuracies[-1] - accuracies[0] if accuracies else 0.0
                condition_improvements.append(total_improvement)

                runs[condition.name][bm_name] = AblationRun(
                    condition=condition.name,
                    benchmark=bm_name,
                    iterations=iterations_list,
                    accuracies=accuracies,
                    total_improvement=total_improvement,
                )

            summary[condition.name] = (
                sum(condition_improvements) / len(condition_improvements)
                if condition_improvements
                else 0.0
            )

        return AblationResult(
            conditions=condition_names,
            benchmarks=benchmark_names,
            runs=runs,
            summary=summary,
        )
