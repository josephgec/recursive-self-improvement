"""Thinking-model based prompt initializer."""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Dict, List, Optional

from src.genome.prompt_genome import PromptGenome
from src.genome.sections import DEFAULT_SECTIONS, SECTION_ORDER
from src.operators.prompts import META_INIT_PROMPT

STRATEGIES = [
    "structured_analytical",
    "step_by_step_methodical",
    "formula_focused",
    "example_driven",
    "verification_heavy",
]

STRATEGY_DESCRIPTIONS = {
    "structured_analytical": (
        "Focus on clear analytical frameworks. Decompose problems systematically."
    ),
    "step_by_step_methodical": (
        "Emphasize methodical step-by-step problem solving with clear intermediate results."
    ),
    "formula_focused": (
        "Center the approach around identifying and applying the correct formulas."
    ),
    "example_driven": (
        "Use worked examples as templates. Show similar solved problems as reference."
    ),
    "verification_heavy": (
        "Emphasize verification and cross-checking. Solve problems multiple ways."
    ),
}


class ThinkingInitializer:
    """Initialize prompt genomes using a thinking-model LLM."""

    def __init__(self, llm_call: Callable[..., str]):
        """
        Args:
            llm_call: Function that takes (prompt, **kwargs) and returns a string.
                      Should be a thinking-model that returns structured JSON.
        """
        self.llm_call = llm_call

    def initialize(
        self,
        n: int,
        domain_desc: str,
        example_tasks: List[str],
    ) -> List[PromptGenome]:
        """Generate n diverse initial genomes using thinking-model strategies.

        Uses 5 strategy variants to create diverse prompts.
        """
        genomes = []
        for i in range(n):
            strategy_idx = i % len(STRATEGIES)
            strategy_name = STRATEGIES[strategy_idx]
            strategy_desc = STRATEGY_DESCRIPTIONS[strategy_name]

            prompt = META_INIT_PROMPT.format(
                domain_description=domain_desc,
                example_tasks="\n".join(f"- {t}" for t in example_tasks),
                strategy=strategy_desc,
            )

            response = self.llm_call(prompt, strategy=strategy_name)
            genome = self._parse_response(response, strategy_name)
            genome.generation = 0
            genome.operator = f"init_{strategy_name}"
            genomes.append(genome)

        return genomes

    def _parse_response(
        self, response: str, strategy: str
    ) -> PromptGenome:
        """Parse LLM response into a PromptGenome."""
        genome = PromptGenome(genome_id=str(uuid.uuid4())[:8])

        try:
            # Try to parse as JSON
            data = json.loads(response)
            if isinstance(data, dict):
                # Handle nested response format
                sections = data.get("sections", data)
                if isinstance(sections, dict):
                    for section_name, content in sections.items():
                        if isinstance(content, str) and section_name in SECTION_ORDER:
                            genome.set_section(section_name, content)
        except (json.JSONDecodeError, TypeError):
            # Fallback: use default sections with strategy flavor
            pass

        # Fill in any missing required sections with defaults
        for section_name in ["identity", "task_description", "methodology"]:
            if section_name not in genome.sections:
                default = DEFAULT_SECTIONS.get(section_name, "")
                genome.set_section(section_name, default)

        # Fill in optional sections with defaults if missing
        for section_name in SECTION_ORDER:
            if section_name not in genome.sections:
                default = DEFAULT_SECTIONS.get(section_name, "")
                genome.set_section(section_name, default)

        return genome
