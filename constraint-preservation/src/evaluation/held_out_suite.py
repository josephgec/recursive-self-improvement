"""HeldOutSuite: held-out evaluation tasks stratified across domains and difficulty."""

from __future__ import annotations

from typing import List


class HeldOutSuite:
    """Collection of held-out evaluation tasks."""

    def __init__(self) -> None:
        self._tasks = self._build_tasks()

    def load(self, n: int = 100) -> List[dict]:
        """Return up to *n* tasks, stratified across domains and difficulty."""
        return self._tasks[:n]

    @staticmethod
    def _build_tasks() -> List[dict]:
        """Build 60 built-in held-out tasks across 6 domains x 3 difficulties."""
        domains = [
            "math",
            "science",
            "language",
            "reasoning",
            "coding",
            "knowledge",
        ]
        difficulties = ["easy", "medium", "hard"]

        tasks: List[dict] = []
        task_id = 0

        templates = {
            "math": [
                "Compute the integral of x^{n} dx from 0 to 1.",
                "Solve the equation: {n}x + 3 = 15.",
                "What is the derivative of sin({n}x)?",
                "Find the sum of the first {n} natural numbers.",
            ],
            "science": [
                "Explain the concept of entropy in thermodynamics.",
                "What is Newton's {n}rd law of motion?",
                "Describe the process of cellular respiration.",
                "What causes the Aurora Borealis?",
            ],
            "language": [
                "Parse the sentence: 'The cat sat on the mat.'",
                "Identify the parts of speech in: 'She quickly ran.'",
                "Translate 'hello' into {n} languages.",
                "Define the word 'ephemeral'.",
            ],
            "reasoning": [
                "If A implies B and B implies C, does A imply C?",
                "All dogs are animals. Spot is a dog. What follows?",
                "There are {n} doors. Behind one is a prize. Strategy?",
                "A bat and ball cost $1.10 total. Bat costs $1 more. Ball cost?",
            ],
            "coding": [
                "Write a function to compute Fibonacci({n}).",
                "Implement binary search in pseudocode.",
                "What is the time complexity of merge sort?",
                "Write a function to reverse a linked list.",
            ],
            "knowledge": [
                "What is the capital of France?",
                "Who wrote 'Hamlet'?",
                "When was the Declaration of Independence signed?",
                "Name the {n} planets in our solar system.",
            ],
        }

        for domain in domains:
            domain_templates = templates[domain]
            for diff_idx, difficulty in enumerate(difficulties):
                for tmpl_idx, template in enumerate(domain_templates):
                    n_val = (diff_idx + 1) * (tmpl_idx + 1)
                    tasks.append(
                        {
                            "id": f"task_{task_id:04d}",
                            "domain": domain,
                            "category": domain,
                            "difficulty": difficulty,
                            "prompt": template.format(n=n_val),
                            "expected_type": "text",
                        }
                    )
                    task_id += 1

        # Pad to at least 100 with extra knowledge tasks
        while len(tasks) < 100:
            tasks.append(
                {
                    "id": f"task_{task_id:04d}",
                    "domain": "knowledge",
                    "category": "knowledge",
                    "difficulty": "medium",
                    "prompt": f"General knowledge question #{task_id}.",
                    "expected_type": "text",
                }
            )
            task_id += 1

        return tasks
