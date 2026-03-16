"""Thinking-model based prompt mutator."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.genome.prompt_genome import PromptGenome
from src.genome.sections import SECTION_IMPORTANCE_WEIGHTS, SECTION_ORDER
from src.operators.prompts import META_MUTATION_PROMPT


@dataclass
class WeaknessAnalysis:
    """Analysis of weaknesses in a genome's performance."""

    weakest_section: str
    weakness_score: float
    details: str
    suggested_improvements: List[str] = field(default_factory=list)


class ThinkingMutator:
    """Mutate prompt genomes using a thinking-model LLM.

    Analyzes weaknesses in the current prompt and targets mutations
    at the weakest section.
    """

    def __init__(self, llm_call: Callable[..., str]):
        self.llm_call = llm_call

    def analyze_weakness(
        self,
        genome: PromptGenome,
        fitness_details: Optional[Dict[str, Any]] = None,
    ) -> WeaknessAnalysis:
        """Analyze which section of the genome is weakest.

        Uses fitness details if available, otherwise uses heuristic analysis.
        """
        if fitness_details and "section_scores" in fitness_details:
            section_scores = fitness_details["section_scores"]
            weakest = min(section_scores, key=section_scores.get)
            return WeaknessAnalysis(
                weakest_section=weakest,
                weakness_score=section_scores[weakest],
                details=f"Section '{weakest}' has lowest score: {section_scores[weakest]:.3f}",
                suggested_improvements=[
                    f"Improve clarity in {weakest}",
                    f"Add more specific guidance in {weakest}",
                ],
            )

        # Heuristic: score sections by content quality indicators
        section_scores = {}
        for name, section in genome.sections.items():
            if not section.is_mutable:
                continue
            content = section.content
            score = 0.0
            # Longer content generally better (up to a point)
            word_count = len(content.split())
            score += min(word_count / 50.0, 1.0) * 0.3
            # Specificity: presence of concrete terms
            specific_terms = [
                "step", "formula", "calculate", "verify", "check",
                "example", "result", "value", "answer",
            ]
            matches = sum(1 for t in specific_terms if t in content.lower())
            score += (matches / len(specific_terms)) * 0.4
            # Structure: presence of punctuation/lists
            if "." in content:
                score += 0.1
            if any(c in content for c in ["-", "*", "1.", "2."]):
                score += 0.2
            section_scores[name] = score

        if not section_scores:
            return WeaknessAnalysis(
                weakest_section="methodology",
                weakness_score=0.0,
                details="No mutable sections found, defaulting to methodology",
            )

        weakest = min(section_scores, key=section_scores.get)
        return WeaknessAnalysis(
            weakest_section=weakest,
            weakness_score=section_scores[weakest],
            details=f"Section '{weakest}' scored lowest on quality heuristics: "
                    f"{section_scores[weakest]:.3f}",
            suggested_improvements=[
                f"Add more specific guidance in {weakest}",
                f"Restructure {weakest} for clarity",
            ],
        )

    def mutate(
        self,
        genome: PromptGenome,
        fitness_details: Optional[Dict[str, Any]] = None,
    ) -> PromptGenome:
        """Mutate a genome by targeting its weakest section.

        Returns a new genome with the mutation applied.
        """
        weakness = self.analyze_weakness(genome, fitness_details)

        # Build sections description for prompt
        sections_desc = ""
        for name, section in genome.sections.items():
            sections_desc += f"\n[{name}]: {section.content[:200]}..."

        prompt = META_MUTATION_PROMPT.format(
            current_sections=sections_desc,
            fitness=genome.fitness,
            weakest_section=weakness.weakest_section,
            weakness_details=weakness.details,
        )

        response = self.llm_call(prompt, target_section=weakness.weakest_section)

        # Create mutated copy
        new_genome = genome.copy()
        new_genome.generation = genome.generation + 1
        new_genome.parent_ids = [genome.genome_id]
        new_genome.operator = f"mutate_{weakness.weakest_section}"
        new_genome.fitness = 0.0  # Reset fitness for re-evaluation

        # Apply mutation from response
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                mutations = data.get("mutations", data)
                if isinstance(mutations, dict):
                    for section_name, content in mutations.items():
                        if (
                            isinstance(content, str)
                            and section_name in new_genome.sections
                            and new_genome.sections[section_name].is_mutable
                        ):
                            new_genome.set_section(section_name, content)
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, apply a simple heuristic mutation
            target = weakness.weakest_section
            if target in new_genome.sections:
                old_content = new_genome.sections[target].content
                improved = old_content + " Be precise and show all work."
                new_genome.set_section(target, improved)

        return new_genome
