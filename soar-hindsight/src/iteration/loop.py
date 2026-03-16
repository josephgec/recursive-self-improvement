"""SOAR Loop: search -> collect -> synthesize -> train -> evaluate -> repeat."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from src.collection.collector import TrajectoryCollector
from src.collection.trajectory import (
    IndividualRecord,
    SearchTrajectory,
    TaskSpec,
)
from src.finetuning.evaluation import Evaluator
from src.finetuning.trainer import Trainer
from src.finetuning.model_registry import ModelRegistry
from src.iteration.convergence import ConvergenceDetector
from src.iteration.improvement_tracker import ImprovementTracker
from src.synthesis.deduplicator import Deduplicator
from src.synthesis.quality_filter import QualityFilter
from src.synthesis.synthesizer import Synthesizer


class SOARLoop:
    """Orchestrates the full SOAR self-improvement loop.

    Each iteration:
    1. Search: Run evolutionary search (mocked) to generate trajectories
    2. Collect: Harvest and process trajectories
    3. Synthesize: Convert trajectories to training data
    4. Train: Fine-tune model on new data
    5. Evaluate: Compare new model against previous
    6. Decide: Continue or stop based on convergence
    """

    def __init__(
        self,
        synthesizer: Synthesizer,
        trainer: Optional[Trainer] = None,
        evaluator: Optional[Evaluator] = None,
        quality_filter: Optional[QualityFilter] = None,
        deduplicator: Optional[Deduplicator] = None,
        convergence: Optional[ConvergenceDetector] = None,
        max_iterations: int = 10,
        seed: int = 42,
    ):
        self.synthesizer = synthesizer
        self.trainer = trainer or Trainer()
        self.evaluator = evaluator or Evaluator()
        self.quality_filter = quality_filter or QualityFilter(
            min_prompt_tokens=5, min_completion_tokens=5, min_quality_score=0.1
        )
        self.deduplicator = deduplicator or Deduplicator()
        self.convergence = convergence or ConvergenceDetector()
        self.max_iterations = max_iterations
        self._seed = seed
        self._tracker = ImprovementTracker()
        self._iteration = 0
        self._history: List[Dict[str, Any]] = []

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    @property
    def tracker(self) -> ImprovementTracker:
        return self._tracker

    def mock_search(self, n_trajectories: int = 10, model_quality: float = 0.3) -> List[SearchTrajectory]:
        """Generate mock search trajectories.

        model_quality controls the base quality - higher means better solutions.
        """
        rng = random.Random(self._seed + self._iteration)
        trajectories = []

        for i in range(n_trajectories):
            task = TaskSpec(
                task_id=f"task-{self._iteration}-{i}",
                description=f"Implement a function that solves problem {i} in iteration {self._iteration}",
                difficulty=rng.choice(["easy", "medium", "hard"]),
                tags=rng.sample(["sorting", "search", "dp", "graph", "string"], k=2),
            )

            individuals = []
            best_fitness = 0.0

            for gen in range(rng.randint(3, 8)):
                n_pop = rng.randint(2, 5)
                for _ in range(n_pop):
                    fitness = min(1.0, model_quality + rng.gauss(0, 0.2) + gen * 0.05)
                    fitness = max(0.0, fitness)
                    best_fitness = max(best_fitness, fitness)

                    has_error = rng.random() < 0.2
                    parent_ids = []
                    operator = rng.choice(["mutation", "crossover", "init"])
                    if operator == "crossover" and len(individuals) >= 2:
                        parents = rng.sample(individuals, 2)
                        parent_ids = [p.individual_id for p in parents]

                    ind = IndividualRecord(
                        generation=gen,
                        code=f"def solve_{i}(x):\n    # gen {gen}\n    return x * {gen + 1}",
                        fitness=fitness,
                        parent_ids=parent_ids,
                        operator=operator,
                        error=f"TypeError: invalid operation at gen {gen}" if has_error else None,
                    )
                    individuals.append(ind)

            traj = SearchTrajectory(
                task=task,
                individuals=individuals,
                best_fitness=best_fitness,
                total_generations=max((ind.generation for ind in individuals), default=0) + 1,
                solved=best_fitness >= 1.0,
            )
            traj.extract_improvement_chain()
            trajectories.append(traj)

        return trajectories

    def run_iteration(self, trajectories: Optional[List[SearchTrajectory]] = None) -> Dict[str, Any]:
        """Run a single SOAR iteration."""
        self._iteration += 1

        # Step 1: Search (use provided or mock)
        if trajectories is None:
            model_quality = 0.3 + self._iteration * 0.05
            trajectories = self.mock_search(model_quality=model_quality)

        # Step 2: Synthesize
        pairs = self.synthesizer.synthesize(trajectories)

        # Step 3: Filter and deduplicate
        pairs = self.quality_filter.filter(pairs)
        pairs = self.deduplicator.deduplicate(pairs)

        # Step 4: Train
        train_result = {}
        if pairs:
            model_name = f"soar-iter-{self._iteration}"
            train_result = self.trainer.train(pairs, model_name=model_name)

        # Step 5: Evaluate
        base_metrics = self.evaluator.evaluate("base", is_base=True)
        ft_name = train_result.get("model_name", f"soar-iter-{self._iteration}")
        ft_metrics = self.evaluator.evaluate(ft_name, is_base=False)
        comparison = self.evaluator.compare("base", ft_name)

        # Step 6: Track improvement
        solve_rate = ft_metrics.get("zero_shot_solve_rate", 0.0)
        self._tracker.record(self._iteration, solve_rate)

        # Check convergence
        converged = self.convergence.check(solve_rate)

        result = {
            "iteration": self._iteration,
            "n_trajectories": len(trajectories),
            "n_pairs_synthesized": len(self.synthesizer.pairs),
            "n_pairs_after_filter": len(pairs),
            "training": train_result,
            "base_metrics": base_metrics,
            "finetuned_metrics": ft_metrics,
            "comparison": comparison,
            "solve_rate": solve_rate,
            "converged": converged,
        }

        self._history.append(result)
        return result

    def run(self, max_iterations: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run the full SOAR loop until convergence or max iterations."""
        limit = max_iterations or self.max_iterations
        for _ in range(limit):
            result = self.run_iteration()
            if result.get("converged", False):
                break
        return self._history

    def summary(self) -> Dict[str, Any]:
        """Return summary of the SOAR loop run."""
        return {
            "total_iterations": self._iteration,
            "converged": self.convergence.is_converged,
            "final_solve_rate": self._tracker.latest_value,
            "improvement_history": self._tracker.history,
        }
