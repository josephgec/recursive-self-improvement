"""Fitness computation combining pixel accuracy, simplicity, and consistency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.arc.evaluator import ProgramEvalResult, ProgramEvaluator
from src.arc.grid import ARCTask
from src.population.individual import Individual


@dataclass
class FitnessWeights:
    """Weights for fitness components."""

    pixel_accuracy: float = 0.7
    simplicity: float = 0.15
    consistency: float = 0.15
    max_code_length: int = 2000


class FitnessComputer:
    """Computes multi-objective fitness for individuals."""

    def __init__(
        self,
        weights: Optional[FitnessWeights] = None,
        evaluator: Optional[ProgramEvaluator] = None,
    ):
        self.weights = weights or FitnessWeights()
        self.evaluator = evaluator or ProgramEvaluator()

    def compute_simplicity(self, code: str) -> float:
        """Score code simplicity (shorter, cleaner code = higher score)."""
        length = len(code)
        if length == 0:
            return 0.0

        # Length penalty
        length_score = max(0.0, 1.0 - length / self.weights.max_code_length)

        # Line count factor
        lines = code.strip().splitlines()
        line_score = max(0.0, 1.0 - len(lines) / 100.0)

        # Nesting depth penalty
        max_indent = 0
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                max_indent = max(max_indent, indent)
        indent_score = max(0.0, 1.0 - max_indent / 32.0)

        return (length_score + line_score + indent_score) / 3.0

    def compute_consistency(self, eval_result: ProgramEvalResult) -> float:
        """Score consistency across training examples."""
        if not eval_result.train_results:
            return 0.0

        accuracies = [r.pixel_accuracy for r in eval_result.train_results]

        if len(accuracies) <= 1:
            return accuracies[0] if accuracies else 0.0

        # Mean accuracy
        mean_acc = sum(accuracies) / len(accuracies)

        # Variance penalty - prefer consistent performance
        variance = sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies)
        consistency = max(0.0, 1.0 - variance * 4.0)  # Scale variance

        return mean_acc * consistency

    def evaluate_individual(
        self,
        individual: Individual,
        task: ARCTask,
    ) -> Individual:
        """Evaluate an individual and update its fitness scores."""
        eval_result = self.evaluator.evaluate_task(individual.code, task)

        # Set compile error if any
        individual.compile_error = eval_result.compile_error

        # Collect runtime errors
        individual.runtime_errors = []
        for r in eval_result.train_results + eval_result.test_results:
            if r.error:
                individual.runtime_errors.append(r.error)

        # Compute component scores
        individual.pixel_accuracy = eval_result.train_accuracy
        individual.train_accuracy = eval_result.train_accuracy
        individual.test_accuracy = eval_result.test_accuracy
        individual.simplicity_score = self.compute_simplicity(individual.code)
        individual.consistency_score = self.compute_consistency(eval_result)

        # Compute weighted fitness
        individual.fitness = (
            self.weights.pixel_accuracy * individual.pixel_accuracy
            + self.weights.simplicity * individual.simplicity_score
            + self.weights.consistency * individual.consistency_score
        )

        individual.evaluated = True
        return individual

    def evaluate_population(
        self,
        individuals: list,
        task: ARCTask,
    ) -> list:
        """Evaluate all individuals in a population."""
        for ind in individuals:
            self.evaluate_individual(ind, task)
        return individuals
