"""Test fixtures: MockThinkingLLM, MockOutputLLM, sample data."""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Dict, List, Optional

import pytest

from src.genome.prompt_genome import PromptGenome
from src.genome.sections import DEFAULT_SECTIONS, SECTION_ORDER
from src.evaluation.financial_math import FinancialMathBenchmark


# ---------------------------------------------------------------------------
# Mock LLMs
# ---------------------------------------------------------------------------

class MockThinkingLLM:
    """Mock thinking-model LLM that returns structured JSON with reasoning.

    Simulates a thinking model that produces:
    - Initialization: structured prompt sections
    - Mutation: improved section content
    - Crossover: per-section decisions
    - Evaluation: task assessments
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._call_count = 0

    def __call__(self, prompt: str, **kwargs) -> str:
        self._call_count += 1

        strategy = kwargs.get("strategy", "")
        target_section = kwargs.get("target_section", "")
        parent_a_id = kwargs.get("parent_a_id", "")
        parent_b_id = kwargs.get("parent_b_id", "")
        system_prompt = kwargs.get("system_prompt", "")
        task_id = kwargs.get("task_id", "")

        # Detect what kind of call this is
        if strategy:
            return self._handle_init(prompt, strategy)
        elif target_section:
            return self._handle_mutation(prompt, target_section)
        elif parent_a_id:
            return self._handle_crossover(prompt)
        elif task_id:
            return self._handle_evaluation(prompt, system_prompt, task_id)
        else:
            return self._handle_generic(prompt)

    def _handle_init(self, prompt: str, strategy: str) -> str:
        """Generate initialization response with structured sections."""
        strategy_flavors = {
            "structured_analytical": "Use analytical frameworks to decompose problems systematically.",
            "step_by_step_methodical": "Follow a strict step-by-step methodology for every problem.",
            "formula_focused": "Identify the correct formula first, then substitute values carefully.",
            "example_driven": "Reference similar worked examples when solving new problems.",
            "verification_heavy": "Always verify your answer by solving the problem a second way.",
        }
        flavor = strategy_flavors.get(strategy, "Solve problems carefully and accurately.")

        sections = {
            "identity": f"You are an expert financial mathematics assistant using a {strategy} approach.",
            "task_description": "Solve financial mathematics problems with precision and clarity. " + flavor,
            "methodology": (
                f"Apply the {strategy} strategy: {flavor} "
                "Break each problem into clear steps. Use standard financial formulas. "
                "Verify calculations by checking units and reasonableness of results."
            ),
            "reasoning_style": (
                "Think step by step. State your approach before calculating. "
                "Show all intermediate values. Double-check your final answer."
            ),
            "output_format": (
                "Present your answer clearly with: 1) The formula used, "
                "2) Substitution of values, 3) Step-by-step calculation, "
                "4) Final numeric result rounded to 2 decimal places."
            ),
            "constraints": (
                "Use exact values from the problem. State assumptions clearly. "
                "Do not round intermediate calculations."
            ),
            "examples": "",
            "error_handling": (
                "If information is missing, state what is needed. "
                "If a problem has multiple interpretations, solve for the most common one."
            ),
        }

        return json.dumps({"sections": sections})

    def _handle_mutation(self, prompt: str, target_section: str) -> str:
        """Generate mutation response improving the target section."""
        improvements = {
            "identity": "You are a highly specialized financial mathematics expert with deep knowledge of compound interest, present value analysis, loan amortization, option pricing, and risk assessment.",
            "task_description": "Solve financial mathematics problems with rigorous step-by-step analysis. Always identify the problem type first, then apply the appropriate formula with careful attention to detail.",
            "methodology": (
                "1. Identify the problem category (interest, valuation, risk, etc.). "
                "2. Select the appropriate formula. "
                "3. List all given values with units. "
                "4. Substitute values into the formula. "
                "5. Compute step by step showing intermediate results. "
                "6. Verify the answer is reasonable."
            ),
            "reasoning_style": (
                "Think systematically: First understand what is being asked. "
                "Then identify the relevant formula. Show all work with clear labels. "
                "Verify by checking: Does the answer make intuitive sense? "
                "Are the units correct? Is the magnitude reasonable?"
            ),
            "output_format": (
                "Structure your response as: Problem Analysis > Formula > "
                "Values > Calculation > Result > Verification. "
                "Round final answers to 2 decimal places unless specified otherwise."
            ),
            "constraints": (
                "Never assume values not given in the problem. "
                "Use exact arithmetic for intermediate steps. "
                "Clearly state any assumptions made."
            ),
            "examples": "When solving compound interest: A = P(1 + r/n)^(nt). Always identify P, r, n, and t first.",
            "error_handling": (
                "If the problem is under-specified, list what additional information is needed. "
                "If multiple interpretations exist, solve the most standard interpretation first."
            ),
        }

        content = improvements.get(
            target_section,
            f"Improved content for {target_section}: Be more precise and thorough."
        )

        return json.dumps({"mutations": {target_section: content}})

    def _handle_crossover(self, prompt: str) -> str:
        """Generate crossover decisions."""
        decisions = {}
        synthesized = {}

        for section in SECTION_ORDER:
            r = self._rng.random()
            if r < 0.35:
                decisions[section] = "TAKE_A"
            elif r < 0.70:
                decisions[section] = "TAKE_B"
            else:
                decisions[section] = "SYNTHESIZE"
                synthesized[section] = (
                    f"Combined best practices for {section}: "
                    "Use systematic analysis with step-by-step verification. "
                    "Apply domain expertise with careful attention to detail."
                )

        return json.dumps({
            "decisions": decisions,
            "synthesized": synthesized,
        })

    def _handle_evaluation(
        self, question: str, system_prompt: str, task_id: str
    ) -> str:
        """Generate evaluation response.

        Better system prompts (longer, more structured) produce more correct answers.
        """
        # Determine correctness based on prompt quality
        prompt_quality = self._assess_prompt_quality(system_prompt)

        # Extract expected answer from task context if available
        # In mock mode, we generate responses that sometimes contain the right answer
        response_parts = [
            "Step 1: Identify the problem type.",
            "Step 2: Select the appropriate formula.",
        ]

        # Higher quality prompts -> more likely to include the answer
        if self._rng.random() < prompt_quality:
            # Try to extract numbers from the question to make a plausible answer
            numbers = re.findall(r"\d+\.?\d*", question)
            if numbers:
                # Use the numbers to compute something plausible
                response_parts.append(
                    f"Step 3: Applying the formula with the given values."
                )
                response_parts.append(
                    f"Therefore, the answer is {numbers[-1]}"
                )
                response_parts.append(
                    "Step 4: Verify - this result is reasonable given the inputs."
                )

        response_parts.append("Final answer: See calculation above.")

        return "\n".join(response_parts)

    def _handle_generic(self, prompt: str) -> str:
        """Handle generic prompts."""
        return json.dumps({"response": "Processed successfully."})

    def _assess_prompt_quality(self, system_prompt: str) -> float:
        """Assess quality of a system prompt (0-1).

        Higher quality prompts lead to more correct answers.
        """
        if not system_prompt:
            return 0.3

        score = 0.3  # Base score
        prompt_lower = system_prompt.lower()

        # Length bonus (more detailed prompts score higher)
        words = len(system_prompt.split())
        score += min(words / 200.0, 0.2)

        # Structure bonus
        if "##" in system_prompt:
            score += 0.1

        # Quality indicators
        quality_terms = [
            "step by step", "formula", "verify", "calculate",
            "check", "systematic", "precise", "methodology",
        ]
        for term in quality_terms:
            if term in prompt_lower:
                score += 0.03

        return min(score, 0.85)


class MockOutputLLM:
    """Mock output LLM that generates responses with controlled accuracy.

    Better prompts (as measured by quality heuristics) produce more
    correct responses. Correct responses contain the expected answer value.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._call_count = 0

    def __call__(self, prompt: str, **kwargs) -> str:
        self._call_count += 1

        system_prompt = kwargs.get("system_prompt", "")
        task_id = kwargs.get("task_id", "")

        # Assess prompt quality to determine accuracy
        quality = self._assess_prompt_quality(system_prompt)

        return self._generate_response(prompt, quality, task_id)

    def _assess_prompt_quality(self, system_prompt: str) -> float:
        """Assess quality of the system prompt."""
        if not system_prompt:
            return 0.3

        score = 0.3
        prompt_lower = system_prompt.lower()
        words = len(system_prompt.split())
        score += min(words / 200.0, 0.2)

        if "##" in system_prompt:
            score += 0.1

        quality_terms = [
            "step by step", "formula", "verify", "calculate",
            "check", "systematic", "precise", "methodology",
        ]
        for term in quality_terms:
            if term in prompt_lower:
                score += 0.03

        return min(score, 0.85)

    def _generate_response(
        self, question: str, quality: float, task_id: str
    ) -> str:
        """Generate a response with accuracy proportional to prompt quality."""
        parts = []

        # Add reasoning markers based on quality
        if quality > 0.4:
            parts.append("Step 1: Let me identify the problem type and relevant formula.")
        if quality > 0.5:
            parts.append("Step 2: Substituting the given values into the formula.")
            parts.append("where the key variables are identified from the problem.")
        if quality > 0.6:
            parts.append("Step 3: Computing the result step by step.")

        # Extract numbers from question for response generation
        numbers = re.findall(r"[\d,]+\.?\d*", question)
        cleaned_numbers = [n.replace(",", "") for n in numbers]

        if self._rng.random() < quality and cleaned_numbers:
            # Include a plausible numeric answer
            parts.append(
                f"The answer is ${cleaned_numbers[-1]}"
            )
            parts.append(
                "Let me verify this result is reasonable - confirmed."
            )
        else:
            parts.append("Based on the calculation, the result equals the computed value.")

        if quality > 0.5:
            parts.append("Final verification: The answer checks out.")

        return "\n".join(parts)


def create_mock_thinking_llm(seed: int = 42) -> MockThinkingLLM:
    """Create a mock thinking LLM for testing."""
    return MockThinkingLLM(seed=seed)


def create_mock_output_llm(seed: int = 42) -> MockOutputLLM:
    """Create a mock output LLM for testing."""
    return MockOutputLLM(seed=seed)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_thinking_llm():
    """Provide a MockThinkingLLM."""
    return MockThinkingLLM(seed=42)


@pytest.fixture
def mock_output_llm():
    """Provide a MockOutputLLM."""
    return MockOutputLLM(seed=42)


@pytest.fixture
def sample_genome():
    """Provide a sample PromptGenome with default sections."""
    genome = PromptGenome(genome_id="test_genome")
    for section_name, content in DEFAULT_SECTIONS.items():
        genome.set_section(section_name, content)
    genome.fitness = 0.65
    return genome


@pytest.fixture
def sample_genome_b():
    """Provide a second sample genome (different from sample_genome)."""
    genome = PromptGenome(genome_id="test_genome_b")
    genome.set_section("identity", "You are a financial calculator AI.")
    genome.set_section("task_description", "Compute financial values precisely.")
    genome.set_section(
        "methodology",
        "Apply formulas directly. Show formula, values, and result.",
    )
    genome.set_section("reasoning_style", "Be concise and direct. Show key steps only.")
    genome.set_section("output_format", "Answer: [value]. Show formula used.")
    genome.set_section("constraints", "Use given values only. Round to 2 decimals.")
    genome.set_section("examples", "")
    genome.set_section("error_handling", "Say if unsolvable.")
    genome.fitness = 0.55
    return genome


@pytest.fixture
def sample_tasks():
    """Provide sample evaluation tasks."""
    bench = FinancialMathBenchmark(seed=42)
    tasks = bench.generate_tasks(n_per_category=2)
    return bench.to_eval_tasks(tasks[:10])


@pytest.fixture
def financial_benchmark():
    """Provide a FinancialMathBenchmark."""
    return FinancialMathBenchmark(seed=42)
