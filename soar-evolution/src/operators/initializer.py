"""LLM-powered population initializer with multiple prompt variants."""

from __future__ import annotations

from typing import Callable, List, Optional

from src.arc.grid import ARCTask
from src.arc.visualizer import GridVisualizer
from src.operators.prompts import INIT_VARIANTS
from src.population.individual import Individual


class LLMInitializer:
    """Generates initial population using LLM with diverse prompt strategies."""

    def __init__(
        self,
        llm_call: Optional[Callable[[str], str]] = None,
        num_variants: int = 5,
        visualizer: Optional[GridVisualizer] = None,
    ):
        self.llm_call = llm_call or default_mock_llm
        self.num_variants = min(num_variants, len(INIT_VARIANTS))
        self.visualizer = visualizer or GridVisualizer()

    def generate(self, task: ARCTask, count: int = 10) -> List[Individual]:
        """Generate initial population of candidate programs."""
        task_desc = self.visualizer.render_task(task)
        individuals = []

        for i in range(count):
            variant_idx = i % self.num_variants
            prompt_template = INIT_VARIANTS[variant_idx]
            prompt = prompt_template.format(task_description=task_desc)

            try:
                code = self.llm_call(prompt)
                code = extract_function(code)
                individual = Individual(
                    code=code,
                    generation=0,
                    operator=f"init_v{variant_idx}",
                    metadata={"prompt_variant": variant_idx},
                )
                individuals.append(individual)
            except Exception as e:
                # Create a fallback individual with identity transform
                individual = Individual(
                    code=_identity_transform(),
                    generation=0,
                    operator="init_fallback",
                    metadata={"error": str(e)},
                )
                individuals.append(individual)

        return individuals

    def generate_single(
        self, task: ARCTask, variant: int = 0
    ) -> Individual:
        """Generate a single candidate program."""
        result = self.generate(task, count=1)
        return result[0]


def extract_function(text: str) -> str:
    """Extract Python function from LLM response text."""
    # Try to extract from code blocks
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0].strip()
            return code
    if "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            code = parts[1].split("```")[0].strip()
            return code

    # If text already looks like a function, return as-is
    if "def transform" in text:
        lines = text.splitlines()
        func_lines = []
        in_function = False
        for line in lines:
            if line.strip().startswith("def transform"):
                in_function = True
            if in_function:
                func_lines.append(line)
        if func_lines:
            return "\n".join(func_lines)

    return text.strip()


def _identity_transform() -> str:
    """Return a default identity transform function."""
    return (
        "def transform(input_grid):\n"
        "    return [row[:] for row in input_grid]\n"
    )


def default_mock_llm(prompt: str) -> str:
    """Default mock LLM that returns deterministic programs based on prompt content."""
    # Analyze the prompt to determine what kind of task it is
    prompt_lower = prompt.lower()

    if "color" in prompt_lower or "swap" in prompt_lower:
        return _color_swap_program(prompt)
    elif "fill" in prompt_lower or "flood" in prompt_lower:
        return _fill_program(prompt)
    elif "rotate" in prompt_lower or "transform" in prompt_lower:
        return _rotate_program(prompt)
    else:
        return _generic_program(prompt)


def _color_swap_program(prompt: str) -> str:
    """Generate a color swap program."""
    return '''def transform(input_grid):
    result = []
    for row in input_grid:
        new_row = []
        for cell in row:
            if cell == 1:
                new_row.append(2)
            elif cell == 2:
                new_row.append(1)
            else:
                new_row.append(cell)
        result.append(new_row)
    return result
'''


def _fill_program(prompt: str) -> str:
    """Generate a fill program."""
    return '''def transform(input_grid):
    # Find the non-zero color
    fill_color = 0
    for row in input_grid:
        for cell in row:
            if cell != 0:
                fill_color = cell
                break
        if fill_color:
            break
    rows = len(input_grid)
    cols = len(input_grid[0])
    return [[fill_color] * cols for _ in range(rows)]
'''


def _rotate_program(prompt: str) -> str:
    """Generate a rotation program."""
    return '''def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    result = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            result[c][rows - 1 - r] = input_grid[r][c]
    return result
'''


def _generic_program(prompt: str) -> str:
    """Generate a generic transform program."""
    return '''def transform(input_grid):
    return [row[:] for row in input_grid]
'''
