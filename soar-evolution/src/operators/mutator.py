"""LLM-powered mutation operator with multiple mutation types."""

from __future__ import annotations

import random
from enum import Enum
from typing import Callable, Dict, List, Optional

from src.arc.grid import ARCTask
from src.arc.visualizer import GridVisualizer
from src.operators.prompts import MUTATION_TEMPLATES
from src.operators.initializer import extract_function
from src.population.individual import Individual


class MutationType(Enum):
    """Types of mutations."""

    BUG_FIX = "BUG_FIX"
    REFINEMENT = "REFINEMENT"
    RESTRUCTURE = "RESTRUCTURE"
    SIMPLIFY = "SIMPLIFY"
    GENERALIZE = "GENERALIZE"


# Weights for selecting mutation types based on individual state
MUTATION_WEIGHTS: Dict[MutationType, Callable[[Individual], float]] = {
    MutationType.BUG_FIX: lambda ind: 3.0 if not ind.is_valid else 0.5,
    MutationType.REFINEMENT: lambda ind: 2.0 if ind.train_accuracy < 0.9 else 1.0,
    MutationType.RESTRUCTURE: lambda ind: 1.5,
    MutationType.SIMPLIFY: lambda ind: 2.0 if ind.code_length > 500 else 0.5,
    MutationType.GENERALIZE: lambda ind: 2.0 if ind.train_accuracy > 0.5 else 0.5,
}


class LLMMutator:
    """Applies LLM-powered mutations to individuals."""

    def __init__(
        self,
        llm_call: Optional[Callable[[str], str]] = None,
        visualizer: Optional[GridVisualizer] = None,
        allowed_types: Optional[List[MutationType]] = None,
    ):
        self.llm_call = llm_call or default_mock_mutator
        self.visualizer = visualizer or GridVisualizer()
        self.allowed_types = allowed_types or list(MutationType)
        self._stats: Dict[str, int] = {mt.value: 0 for mt in MutationType}

    def select_mutation_type(self, individual: Individual) -> MutationType:
        """Select a mutation type based on individual state."""
        weights = []
        types = []
        for mt in self.allowed_types:
            weight_fn = MUTATION_WEIGHTS.get(mt, lambda _: 1.0)
            weights.append(weight_fn(individual))
            types.append(mt)

        total = sum(weights)
        if total == 0:
            return random.choice(types)

        r = random.random() * total
        cumulative = 0.0
        for mt, w in zip(types, weights):
            cumulative += w
            if r <= cumulative:
                return mt

        return types[-1]

    def mutate(
        self,
        individual: Individual,
        task: ARCTask,
        mutation_type: Optional[MutationType] = None,
    ) -> Individual:
        """Apply a mutation to an individual."""
        if mutation_type is None:
            mutation_type = self.select_mutation_type(individual)

        self._stats[mutation_type.value] = self._stats.get(mutation_type.value, 0) + 1

        task_desc = self.visualizer.render_task(task)
        template = MUTATION_TEMPLATES[mutation_type.value]

        errors = "\n".join(individual.runtime_errors[:5]) if individual.runtime_errors else "None"
        if individual.compile_error:
            errors = individual.compile_error + "\n" + errors

        prompt = template.format(
            code=individual.code,
            task_description=task_desc,
            errors=errors,
            accuracy=individual.train_accuracy,
        )

        try:
            response = self.llm_call(prompt)
            new_code = extract_function(response)
        except Exception:
            new_code = individual.code  # Fallback: no mutation

        child = Individual(
            code=new_code,
            generation=individual.generation + 1,
            parent_ids=[individual.individual_id],
            operator=f"mutate_{mutation_type.value.lower()}",
            metadata={
                "mutation_type": mutation_type.value,
                "parent_fitness": individual.fitness,
            },
        )

        return child

    def mutate_batch(
        self,
        individuals: List[Individual],
        task: ARCTask,
    ) -> List[Individual]:
        """Apply mutations to a batch of individuals."""
        return [self.mutate(ind, task) for ind in individuals]

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)


def default_mock_mutator(prompt: str) -> str:
    """Mock mutator LLM that applies deterministic transformations."""
    prompt_lower = prompt.lower()

    # Extract existing code from the prompt
    if "```python" in prompt:
        code = prompt.split("```python")[1].split("```")[0].strip()
    elif "```" in prompt:
        code = prompt.split("```")[1].split("```")[0].strip()
    else:
        code = "def transform(input_grid):\n    return [row[:] for row in input_grid]\n"

    if "bug" in prompt_lower or "fix" in prompt_lower:
        # Bug fix: add try-except wrapper
        return f'''def transform(input_grid):
    try:
        rows = len(input_grid)
        cols = len(input_grid[0]) if rows > 0 else 0
        result = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                result[r][c] = input_grid[r][c]
        return result
    except Exception:
        return [row[:] for row in input_grid]
'''
    elif "simplif" in prompt_lower:
        # Simplify: return compact version
        return '''def transform(input_grid):
    return [row[:] for row in input_grid]
'''
    elif "generalize" in prompt_lower:
        # Generalize: add bounds checking
        return '''def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0]) if rows > 0 else 0
    result = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            result[r][c] = input_grid[r][c]
    return result
'''
    else:
        # Default: return the code with minor changes
        return code
