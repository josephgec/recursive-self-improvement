"""Non-thinking (template/random-based) operators for ablation comparison."""

from __future__ import annotations

import random
import uuid
from typing import Any, Callable, Dict, List, Optional

from src.genome.prompt_genome import PromptGenome
from src.genome.sections import DEFAULT_SECTIONS, SECTION_ORDER
from src.operators.thinking_evaluator import FitnessDetails, ReasoningQuality, TaskEvaluation

# Template variations for non-thinking initialization
_IDENTITY_TEMPLATES = [
    "You are a financial math assistant.",
    "You are an AI that solves financial problems.",
    "You are a calculator for financial mathematics.",
    "You are a helpful math tutor for finance.",
]

_METHODOLOGY_TEMPLATES = [
    "Solve problems step by step.",
    "Use formulas to solve problems.",
    "Break down the problem and calculate.",
    "Apply the relevant formula and compute.",
]

_REASONING_TEMPLATES = [
    "Show your work.",
    "Think carefully about each step.",
    "Explain your reasoning.",
    "Walk through the solution methodically.",
]


class NonThinkingInitializer:
    """Template-based initializer that does not use LLM reasoning."""

    def initialize(
        self,
        n: int,
        domain_desc: str = "",
        example_tasks: Optional[List[str]] = None,
    ) -> List[PromptGenome]:
        """Generate n genomes using random template combinations."""
        genomes = []
        for _ in range(n):
            genome = PromptGenome(genome_id=str(uuid.uuid4())[:8])
            genome.generation = 0
            genome.operator = "init_template"

            genome.set_section(
                "identity", random.choice(_IDENTITY_TEMPLATES)
            )
            genome.set_section(
                "task_description",
                DEFAULT_SECTIONS.get("task_description", "Solve math problems."),
            )
            genome.set_section(
                "methodology", random.choice(_METHODOLOGY_TEMPLATES)
            )
            genome.set_section(
                "reasoning_style", random.choice(_REASONING_TEMPLATES)
            )
            genome.set_section(
                "output_format",
                DEFAULT_SECTIONS.get("output_format", "Show your answer."),
            )
            genome.set_section(
                "constraints",
                DEFAULT_SECTIONS.get("constraints", "Use given values only."),
            )
            genome.set_section("examples", "")
            genome.set_section(
                "error_handling",
                DEFAULT_SECTIONS.get("error_handling", "State if unsolvable."),
            )
            genomes.append(genome)

        return genomes


class NonThinkingMutator:
    """Random mutation operator that does not use LLM reasoning."""

    MUTATIONS = [
        " Be precise.",
        " Show all calculations.",
        " Double-check your work.",
        " Use standard formulas.",
        " Verify your answer.",
        " State assumptions clearly.",
    ]

    def mutate(
        self,
        genome: PromptGenome,
        fitness_details: Optional[Dict[str, Any]] = None,
    ) -> PromptGenome:
        """Apply random mutation to a random mutable section."""
        new_genome = genome.copy()
        new_genome.generation = genome.generation + 1
        new_genome.parent_ids = [genome.genome_id]
        new_genome.operator = "mutate_random"
        new_genome.fitness = 0.0

        mutable = [
            name
            for name, section in new_genome.sections.items()
            if section.is_mutable
        ]

        if mutable:
            target = random.choice(mutable)
            old_content = new_genome.sections[target].content
            mutation = random.choice(self.MUTATIONS)
            new_genome.set_section(target, old_content + mutation)

        return new_genome


class NonThinkingEvaluator:
    """Simplified evaluator that does basic accuracy checking."""

    def __init__(self, answer_checker: Optional[Any] = None):
        self.answer_checker = answer_checker

    def evaluate(
        self,
        genome: PromptGenome,
        tasks: List[Dict[str, Any]],
        llm_call: Optional[Callable] = None,
    ) -> FitnessDetails:
        """Evaluate genome with simplified metrics.

        If llm_call is provided, uses it. Otherwise returns baseline fitness.
        """
        details = FitnessDetails()

        if llm_call is None:
            # Return baseline fitness
            details.accuracy = 0.5
            details.reasoning_score = 0.3
            details.consistency_score = 0.8
            details.compute_composite()
            return details

        system_prompt = genome.to_system_prompt()
        correct = 0

        for task in tasks:
            task_id = task.get("task_id", "unknown")
            question = task.get("question", "")
            expected = task.get("expected_answer", "")

            response = llm_call(question, system_prompt=system_prompt, task_id=task_id)

            is_correct = False
            if self.answer_checker:
                is_correct = self.answer_checker.check(response, expected)
            else:
                is_correct = str(expected).lower() in response.lower()

            if is_correct:
                correct += 1

            rq = ReasoningQuality()
            rq.compute_score()

            details.task_evaluations.append(
                TaskEvaluation(
                    task_id=task_id,
                    is_correct=is_correct,
                    expected_answer=str(expected),
                    actual_answer=response[:200],
                    reasoning_quality=rq,
                )
            )

        details.accuracy = correct / max(len(tasks), 1)
        details.reasoning_score = 0.3  # Fixed low score (no reasoning)
        details.consistency_score = 0.8
        details.compute_composite()
        return details


class SimpleCrossover:
    """Simple crossover that randomly picks sections from parents."""

    def crossover(
        self,
        parent_a: PromptGenome,
        parent_b: PromptGenome,
        fitness_a: float = 0.0,
        fitness_b: float = 0.0,
    ) -> PromptGenome:
        """Create offspring by randomly selecting sections from parents."""
        offspring = PromptGenome(
            genome_id=str(uuid.uuid4())[:8],
            generation=max(parent_a.generation, parent_b.generation) + 1,
            parent_ids=[parent_a.genome_id, parent_b.genome_id],
            operator="crossover_simple",
            fitness=0.0,
        )

        all_sections = set(parent_a.sections.keys()) | set(parent_b.sections.keys())

        for section_name in all_sections:
            # Randomly pick from one parent
            has_a = section_name in parent_a.sections
            has_b = section_name in parent_b.sections

            if has_a and has_b:
                chosen = random.choice([parent_a, parent_b])
                offspring.set_section(
                    section_name,
                    chosen.sections[section_name].content,
                )
            elif has_a:
                offspring.set_section(
                    section_name,
                    parent_a.sections[section_name].content,
                )
            else:
                offspring.set_section(
                    section_name,
                    parent_b.sections[section_name].content,
                )

        return offspring
