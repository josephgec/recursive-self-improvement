"""Evaluation: compare fine-tuned models against base performance."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from src.collection.trajectory import TaskSpec


class Evaluator:
    """Compare fine-tuned vs base model on key metrics.

    Metrics:
    - zero_shot_solve_rate: fraction of tasks solved without search
    - initial_quality: average fitness of first-generation individuals
    - mutation_quality: average fitness improvement from mutations

    All evaluations are mocked for offline testing.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._results: Dict[str, Dict[str, float]] = {}

    def evaluate(
        self,
        model_name: str,
        tasks: Optional[List[TaskSpec]] = None,
        n_tasks: int = 50,
        is_base: bool = False,
    ) -> Dict[str, float]:
        """Evaluate a model on a set of tasks (mocked)."""
        rng = random.Random(self._seed + hash(model_name))

        # Base models have lower performance
        if is_base:
            zero_shot = rng.uniform(0.05, 0.15)
            initial_q = rng.uniform(0.1, 0.3)
            mutation_q = rng.uniform(0.02, 0.08)
        else:
            # Fine-tuned models improve across all metrics
            zero_shot = rng.uniform(0.15, 0.45)
            initial_q = rng.uniform(0.25, 0.55)
            mutation_q = rng.uniform(0.05, 0.15)

        metrics = {
            "zero_shot_solve_rate": round(zero_shot, 4),
            "initial_quality": round(initial_q, 4),
            "mutation_quality": round(mutation_q, 4),
            "n_tasks": n_tasks,
        }

        self._results[model_name] = metrics
        return metrics

    def compare(self, base_name: str, finetuned_name: str) -> Dict[str, Any]:
        """Compare fine-tuned model against base model."""
        base = self._results.get(base_name)
        ft = self._results.get(finetuned_name)

        if base is None or ft is None:
            missing = []
            if base is None:
                missing.append(base_name)
            if ft is None:
                missing.append(finetuned_name)
            return {"error": f"Missing evaluations for: {', '.join(missing)}"}

        comparison = {}
        for metric in ["zero_shot_solve_rate", "initial_quality", "mutation_quality"]:
            base_val = base.get(metric, 0.0)
            ft_val = ft.get(metric, 0.0)
            improvement = ft_val - base_val
            relative = improvement / base_val if base_val > 0 else 0.0
            comparison[metric] = {
                "base": base_val,
                "finetuned": ft_val,
                "improvement": round(improvement, 4),
                "relative_improvement": round(relative, 4),
            }

        comparison["overall_improved"] = all(
            comparison[m]["improvement"] > 0
            for m in ["zero_shot_solve_rate", "initial_quality", "mutation_quality"]
        )

        return comparison

    @property
    def results(self) -> Dict[str, Dict[str, float]]:
        return dict(self._results)

    def clear(self) -> None:
        self._results.clear()
