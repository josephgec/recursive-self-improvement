"""Ablation study framework for comparing evolution conditions."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.ga.engine import GAEngine, EvolutionResult
from src.operators.thinking_evaluator import FitnessDetails


@dataclass
class ConditionResult:
    """Results for a single ablation condition."""

    condition_name: str
    fitness_scores: List[float] = field(default_factory=list)
    mean_fitness: float = 0.0
    std_fitness: float = 0.0
    best_fitness: float = 0.0
    evolution_results: List[EvolutionResult] = field(default_factory=list)

    def compute_stats(self):
        """Compute summary statistics from fitness scores."""
        if not self.fitness_scores:
            return
        self.mean_fitness = sum(self.fitness_scores) / len(self.fitness_scores)
        self.best_fitness = max(self.fitness_scores)
        variance = sum(
            (x - self.mean_fitness) ** 2 for x in self.fitness_scores
        ) / max(len(self.fitness_scores) - 1, 1)
        self.std_fitness = variance ** 0.5


@dataclass
class AblationResult:
    """Results from a complete ablation study."""

    conditions: Dict[str, ConditionResult] = field(default_factory=dict)
    summary: str = ""

    def get_ranking(self) -> List[str]:
        """Get conditions ranked by mean fitness (descending)."""
        return sorted(
            self.conditions.keys(),
            key=lambda c: self.conditions[c].mean_fitness,
            reverse=True,
        )

    def generate_summary(self) -> str:
        """Generate a text summary of the ablation results."""
        lines = ["Ablation Study Results", "=" * 40]

        ranking = self.get_ranking()
        for i, cond in enumerate(ranking, 1):
            cr = self.conditions[cond]
            lines.append(
                f"{i}. {cond}: mean={cr.mean_fitness:.4f} "
                f"(+/-{cr.std_fitness:.4f}), best={cr.best_fitness:.4f}"
            )

        self.summary = "\n".join(lines)
        return self.summary


# Ablation condition names
CONDITION_FULL_THINKING = "full_thinking"
CONDITION_NO_THINKING = "no_thinking"
CONDITION_RANDOM_MUTATION = "random_mutation"
CONDITION_NO_CROSSOVER = "no_crossover"
CONDITION_NO_ELITISM = "no_elitism"
CONDITION_SMALL_POPULATION = "small_population"
CONDITION_LARGE_POPULATION = "large_population"

ALL_CONDITIONS = [
    CONDITION_FULL_THINKING,
    CONDITION_NO_THINKING,
    CONDITION_RANDOM_MUTATION,
    CONDITION_NO_CROSSOVER,
    CONDITION_NO_ELITISM,
    CONDITION_SMALL_POPULATION,
    CONDITION_LARGE_POPULATION,
]


class AblationStudy:
    """Run ablation studies comparing different evolution conditions.

    Supports 7 conditions testing the contribution of each component.
    """

    def __init__(
        self,
        engine_factory: Callable[..., GAEngine],
        domain_desc: str = "",
        example_tasks: Optional[List[str]] = None,
    ):
        """
        Args:
            engine_factory: Function that takes condition kwargs and returns a GAEngine.
            domain_desc: Domain description for evolution
            example_tasks: Example task descriptions
        """
        self.engine_factory = engine_factory
        self.domain_desc = domain_desc
        self.example_tasks = example_tasks or []

    def run(
        self,
        eval_tasks: List[Dict[str, Any]],
        repetitions: int = 5,
        conditions: Optional[List[str]] = None,
    ) -> AblationResult:
        """Run the ablation study.

        Args:
            eval_tasks: Tasks for evaluation
            repetitions: Number of repetitions per condition
            conditions: Which conditions to test (default: all)

        Returns:
            AblationResult with per-condition statistics.
        """
        if conditions is None:
            conditions = ALL_CONDITIONS

        result = AblationResult()

        for condition in conditions:
            cond_result = ConditionResult(condition_name=condition)

            for rep in range(repetitions):
                engine = self._create_engine_for_condition(condition)
                evo_result = engine.evolve(
                    self.domain_desc,
                    self.example_tasks,
                    eval_tasks,
                )
                cond_result.fitness_scores.append(evo_result.best_fitness)
                cond_result.evolution_results.append(evo_result)

            cond_result.compute_stats()
            result.conditions[condition] = cond_result

        result.generate_summary()
        return result

    def _create_engine_for_condition(self, condition: str) -> GAEngine:
        """Create a GAEngine configured for the given ablation condition."""
        kwargs: Dict[str, Any] = {}

        if condition == CONDITION_NO_THINKING:
            kwargs["use_thinking"] = False
        elif condition == CONDITION_RANDOM_MUTATION:
            kwargs["random_mutation"] = True
        elif condition == CONDITION_NO_CROSSOVER:
            kwargs["crossover_rate"] = 0.0
        elif condition == CONDITION_NO_ELITISM:
            kwargs["elitism_count"] = 0
        elif condition == CONDITION_SMALL_POPULATION:
            kwargs["population_size"] = 4
        elif condition == CONDITION_LARGE_POPULATION:
            kwargs["population_size"] = 30
        # CONDITION_FULL_THINKING uses defaults

        return self.engine_factory(**kwargs)
