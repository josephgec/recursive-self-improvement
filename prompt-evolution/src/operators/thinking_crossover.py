"""Thinking-model based prompt crossover."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.genome.prompt_genome import PromptGenome
from src.genome.sections import SECTION_ORDER
from src.genome.similarity import section_similarity
from src.operators.prompts import META_CROSSOVER_PROMPT


@dataclass
class CrossoverDecision:
    """Decision for a single section during crossover."""

    section_name: str
    action: str  # TAKE_A, TAKE_B, SYNTHESIZE
    content: str = ""
    reasoning: str = ""


class ThinkingCrossover:
    """Crossover operator using a thinking-model LLM.

    Analyzes complementarity between parents and makes per-section
    decisions about how to combine them.
    """

    def __init__(self, llm_call: Callable[..., str]):
        self.llm_call = llm_call

    def analyze_complementarity(
        self,
        parent_a: PromptGenome,
        parent_b: PromptGenome,
    ) -> Dict[str, float]:
        """Analyze how complementary two parents' sections are.

        Returns per-section complementarity scores (0 = identical, 1 = very different).
        """
        result = {}
        all_sections = set(parent_a.sections.keys()) | set(parent_b.sections.keys())

        for section_name in all_sections:
            text_a = ""
            text_b = ""
            if section_name in parent_a.sections:
                text_a = parent_a.sections[section_name].content
            if section_name in parent_b.sections:
                text_b = parent_b.sections[section_name].content

            sim = section_similarity(text_a, text_b)
            result[section_name] = 1.0 - sim

        return result

    def crossover(
        self,
        parent_a: PromptGenome,
        parent_b: PromptGenome,
        fitness_a: float = 0.0,
        fitness_b: float = 0.0,
    ) -> PromptGenome:
        """Produce an offspring genome by combining two parents.

        Uses thinking-model to decide per-section strategy.
        """
        complementarity = self.analyze_complementarity(parent_a, parent_b)

        sections_a_desc = ""
        for name, section in parent_a.sections.items():
            sections_a_desc += f"\n[{name}]: {section.content[:200]}"

        sections_b_desc = ""
        for name, section in parent_b.sections.items():
            sections_b_desc += f"\n[{name}]: {section.content[:200]}"

        comp_desc = "\n".join(
            f"  {k}: {v:.2f} complementarity" for k, v in complementarity.items()
        )

        prompt = META_CROSSOVER_PROMPT.format(
            fitness_a=fitness_a,
            fitness_b=fitness_b,
            sections_a=sections_a_desc,
            sections_b=sections_b_desc,
            complementarity=comp_desc,
        )

        response = self.llm_call(
            prompt,
            parent_a_id=parent_a.genome_id,
            parent_b_id=parent_b.genome_id,
        )

        # Create offspring
        offspring = PromptGenome(
            genome_id=str(uuid.uuid4())[:8],
            generation=max(parent_a.generation, parent_b.generation) + 1,
            parent_ids=[parent_a.genome_id, parent_b.genome_id],
            operator="crossover",
            fitness=0.0,
        )

        # Parse decisions
        decisions = self._parse_decisions(response, parent_a, parent_b)

        # Apply decisions
        all_sections = set(parent_a.sections.keys()) | set(parent_b.sections.keys())
        for section_name in all_sections:
            decision = decisions.get(section_name)
            if decision is None:
                # Default: take from fitter parent
                if fitness_a >= fitness_b and section_name in parent_a.sections:
                    offspring.set_section(
                        section_name,
                        parent_a.sections[section_name].content,
                    )
                elif section_name in parent_b.sections:
                    offspring.set_section(
                        section_name,
                        parent_b.sections[section_name].content,
                    )
                elif section_name in parent_a.sections:
                    offspring.set_section(
                        section_name,
                        parent_a.sections[section_name].content,
                    )
            elif decision.action == "TAKE_A" and section_name in parent_a.sections:
                offspring.set_section(
                    section_name,
                    parent_a.sections[section_name].content,
                )
            elif decision.action == "TAKE_B" and section_name in parent_b.sections:
                offspring.set_section(
                    section_name,
                    parent_b.sections[section_name].content,
                )
            elif decision.action == "SYNTHESIZE" and decision.content:
                offspring.set_section(section_name, decision.content)
            else:
                # Fallback
                if section_name in parent_a.sections:
                    offspring.set_section(
                        section_name,
                        parent_a.sections[section_name].content,
                    )
                elif section_name in parent_b.sections:
                    offspring.set_section(
                        section_name,
                        parent_b.sections[section_name].content,
                    )

        return offspring

    def _parse_decisions(
        self,
        response: str,
        parent_a: PromptGenome,
        parent_b: PromptGenome,
    ) -> Dict[str, CrossoverDecision]:
        """Parse LLM response into crossover decisions."""
        decisions = {}

        try:
            data = json.loads(response)
            if isinstance(data, dict):
                raw_decisions = data.get("decisions", {})
                synthesized = data.get("synthesized", {})

                for section_name, action in raw_decisions.items():
                    if action in ("TAKE_A", "TAKE_B", "SYNTHESIZE"):
                        content = ""
                        if action == "SYNTHESIZE":
                            content = synthesized.get(section_name, "")
                        decisions[section_name] = CrossoverDecision(
                            section_name=section_name,
                            action=action,
                            content=content,
                        )
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        return decisions
