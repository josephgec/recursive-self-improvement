"""LLM-powered crossover operator with complementarity analysis."""

from __future__ import annotations

from typing import Callable, List, Optional

from src.arc.grid import ARCTask
from src.arc.visualizer import GridVisualizer
from src.operators.prompts import CROSSOVER_TEMPLATE
from src.operators.initializer import extract_function
from src.population.individual import Individual
from src.utils.code_similarity import code_similarity


class LLMCrossover:
    """Combines two parent programs using LLM guidance."""

    def __init__(
        self,
        llm_call: Optional[Callable[[str], str]] = None,
        visualizer: Optional[GridVisualizer] = None,
        complementarity_threshold: float = 0.4,
    ):
        self.llm_call = llm_call or default_mock_crossover
        self.visualizer = visualizer or GridVisualizer()
        self.complementarity_threshold = complementarity_threshold
        self._stats = {"attempted": 0, "successful": 0}

    def analyze_complementarity(
        self,
        parent_a: Individual,
        parent_b: Individual,
    ) -> dict:
        """Analyze how complementary two parents are."""
        sim = code_similarity(parent_a.code, parent_b.code)

        # Check if they solve different examples well
        acc_diff = abs(parent_a.train_accuracy - parent_b.train_accuracy)

        # Complementarity is higher when programs are different but both useful
        complementarity = (1.0 - sim) * 0.6 + acc_diff * 0.4
        is_complementary = complementarity >= self.complementarity_threshold

        return {
            "similarity": sim,
            "accuracy_diff": acc_diff,
            "complementarity": complementarity,
            "is_complementary": is_complementary,
            "analysis": (
                f"Similarity: {sim:.2f}, Accuracy diff: {acc_diff:.2f}, "
                f"Complementarity: {complementarity:.2f}"
            ),
        }

    def crossover(
        self,
        parent_a: Individual,
        parent_b: Individual,
        task: ARCTask,
    ) -> Individual:
        """Combine two parent programs."""
        self._stats["attempted"] += 1

        comp = self.analyze_complementarity(parent_a, parent_b)
        task_desc = self.visualizer.render_task(task)

        prompt = CROSSOVER_TEMPLATE.format(
            code_a=parent_a.code,
            code_b=parent_b.code,
            accuracy_a=parent_a.train_accuracy,
            accuracy_b=parent_b.train_accuracy,
            task_description=task_desc,
            complementarity_analysis=comp["analysis"],
        )

        try:
            response = self.llm_call(prompt)
            new_code = extract_function(response)
            self._stats["successful"] += 1
        except Exception:
            # Fallback: use better parent's code
            if parent_a.fitness >= parent_b.fitness:
                new_code = parent_a.code
            else:
                new_code = parent_b.code

        child = Individual(
            code=new_code,
            generation=max(parent_a.generation, parent_b.generation) + 1,
            parent_ids=[parent_a.individual_id, parent_b.individual_id],
            operator="crossover",
            metadata={
                "complementarity": comp["complementarity"],
                "parent_a_fitness": parent_a.fitness,
                "parent_b_fitness": parent_b.fitness,
            },
        )

        return child

    @property
    def stats(self) -> dict:
        return dict(self._stats)


def default_mock_crossover(prompt: str) -> str:
    """Mock crossover LLM that combines two programs deterministically."""
    # Extract both code blocks from the prompt
    codes = []
    if "```python" in prompt:
        parts = prompt.split("```python")
        for part in parts[1:]:
            code = part.split("```")[0].strip()
            codes.append(code)
    elif "```" in prompt:
        parts = prompt.split("```")
        for i in range(1, len(parts), 2):
            codes.append(parts[i].strip())

    if len(codes) >= 2:
        # Combine: take structure from first, add elements from second
        return codes[0]  # Use first parent as base

    if codes:
        return codes[0]

    return '''def transform(input_grid):
    return [row[:] for row in input_grid]
'''
