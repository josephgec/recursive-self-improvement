"""Perturbation generator: creates modified versions of tasks."""

from __future__ import annotations

import random
import re
from typing import List, Optional

from src.utils.task_domains import Task


class PerturbationGenerator:
    """Generates perturbed versions of tasks for robustness evaluation.

    Perturbation types:
    - rephrase: Change wording without changing semantics
    - noise: Add irrelevant information
    - domain_shift: Change domain-specific terminology
    - adversarial: Add misleading information
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def rephrase(self, task: Task) -> Task:
        """Rephrase the problem without changing semantics."""
        problem = task.problem
        swaps = [
            ("What is", "Calculate"),
            ("Calculate", "Find the value of"),
            ("Find the value of", "What is"),
            ("Compute", "Evaluate"),
            ("Evaluate", "Compute"),
            ("Solve for", "Find the value of"),
            ("Find", "Determine"),
            ("Determine", "Find"),
            ("Is it true that", "Does it hold that"),
            ("What is the probability", "Find the probability"),
        ]
        for old, new in swaps:
            if old.lower() in problem.lower():
                idx = problem.lower().index(old.lower())
                problem = problem[:idx] + new + problem[idx + len(old):]
                break
        else:
            problem = f"Please answer the following: {problem}"

        return Task(
            task_id=f"{task.task_id}_rephrase",
            domain=task.domain,
            problem=problem,
            expected_answer=task.expected_answer,
            difficulty=task.difficulty,
            metadata={**task.metadata, "perturbation": "rephrase", "original_id": task.task_id},
        )

    def add_noise(self, task: Task) -> Task:
        """Add irrelevant information to the problem."""
        noise_phrases = [
            "Note that the sky is blue.",
            "Assume the temperature is 20 degrees Celsius.",
            "Consider that it is a Tuesday.",
            "The population of Earth is about 8 billion.",
            "Given that water boils at 100 degrees Celsius,",
        ]
        noise = self._rng.choice(noise_phrases)
        problem = f"{noise} {task.problem}"

        return Task(
            task_id=f"{task.task_id}_noise",
            domain=task.domain,
            problem=problem,
            expected_answer=task.expected_answer,
            difficulty=task.difficulty,
            metadata={**task.metadata, "perturbation": "noise", "original_id": task.task_id},
        )

    def domain_shift(self, task: Task) -> Task:
        """Apply domain-specific terminology changes."""
        problem = task.problem
        shifts = [
            ("probability", "chance"),
            ("compute", "calculate"),
            ("equation", "expression"),
            ("solve", "work out"),
            ("true", "correct"),
            ("false", "incorrect"),
        ]
        for old, new in shifts:
            if old in problem.lower():
                problem = re.sub(re.escape(old), new, problem, flags=re.IGNORECASE, count=1)
                break
        else:
            problem = f"In a different context: {problem}"

        return Task(
            task_id=f"{task.task_id}_domain_shift",
            domain=task.domain,
            problem=problem,
            expected_answer=task.expected_answer,
            difficulty=task.difficulty,
            metadata={**task.metadata, "perturbation": "domain_shift", "original_id": task.task_id},
        )

    def adversarial(self, task: Task) -> Task:
        """Add adversarial / misleading information."""
        misleading = [
            "Some people might say the answer is 0, but",
            "A common mistake is to think the answer is -1, however",
            "Although it might seem like the answer is undefined,",
            "Despite appearances,",
        ]
        prefix = self._rng.choice(misleading)
        problem = f"{prefix} {task.problem}"

        return Task(
            task_id=f"{task.task_id}_adversarial",
            domain=task.domain,
            problem=problem,
            expected_answer=task.expected_answer,
            difficulty=task.difficulty,
            metadata={**task.metadata, "perturbation": "adversarial", "original_id": task.task_id},
        )

    def generate_all(self, task: Task) -> List[Task]:
        """Generate all perturbation types for a task."""
        return [
            self.rephrase(task),
            self.add_noise(task),
            self.domain_shift(task),
            self.adversarial(task),
        ]
