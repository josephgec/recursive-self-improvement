"""Thinking-model based prompt evaluator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ReasoningQuality:
    """Assessment of reasoning quality in a response."""

    has_step_by_step: bool = False
    has_formula_reference: bool = False
    has_intermediate_values: bool = False
    has_final_answer: bool = False
    has_verification: bool = False
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute reasoning quality score from boolean flags."""
        checks = [
            self.has_step_by_step,
            self.has_formula_reference,
            self.has_intermediate_values,
            self.has_final_answer,
            self.has_verification,
        ]
        self.score = sum(1.0 for c in checks if c) / len(checks)
        return self.score


@dataclass
class TaskEvaluation:
    """Evaluation result for a single task."""

    task_id: str
    is_correct: bool
    expected_answer: str
    actual_answer: str
    reasoning_quality: ReasoningQuality
    response_text: str = ""


@dataclass
class FitnessDetails:
    """Detailed fitness breakdown."""

    accuracy: float = 0.0
    reasoning_score: float = 0.0
    consistency_score: float = 0.0
    composite_fitness: float = 0.0
    task_evaluations: List[TaskEvaluation] = field(default_factory=list)
    section_scores: Dict[str, float] = field(default_factory=dict)

    def compute_composite(
        self,
        accuracy_weight: float = 0.7,
        reasoning_weight: float = 0.15,
        consistency_weight: float = 0.15,
    ) -> float:
        """Compute weighted composite fitness."""
        self.composite_fitness = (
            accuracy_weight * self.accuracy
            + reasoning_weight * self.reasoning_score
            + consistency_weight * self.consistency_score
        )
        return self.composite_fitness


class ThinkingEvaluator:
    """Evaluate prompt genomes using a thinking-model LLM.

    Assesses accuracy, reasoning quality, and consistency.
    """

    def __init__(
        self,
        llm_call: Callable[..., str],
        answer_checker: Optional[Any] = None,
    ):
        self.llm_call = llm_call
        self.answer_checker = answer_checker

    def evaluate(
        self,
        genome: "PromptGenome",
        tasks: List[Dict[str, Any]],
    ) -> FitnessDetails:
        """Evaluate a genome on a set of tasks.

        Args:
            genome: The prompt genome to evaluate
            tasks: List of task dicts with 'question', 'expected_answer', 'task_id'

        Returns:
            FitnessDetails with accuracy, reasoning quality, and consistency.
        """
        from src.genome.prompt_genome import PromptGenome

        system_prompt = genome.to_system_prompt()
        details = FitnessDetails()
        correct_count = 0
        reasoning_scores = []
        answers_per_task: Dict[str, List[str]] = {}

        for task in tasks:
            task_id = task.get("task_id", "unknown")
            question = task.get("question", "")
            expected = task.get("expected_answer", "")

            # Call LLM with system prompt and question
            response = self.llm_call(
                question,
                system_prompt=system_prompt,
                task_id=task_id,
            )

            # Check answer
            is_correct = False
            if self.answer_checker:
                is_correct = self.answer_checker.check(response, expected)
            else:
                is_correct = self._simple_check(response, expected)

            if is_correct:
                correct_count += 1

            # Assess reasoning quality
            rq = self._assess_reasoning(response)
            reasoning_scores.append(rq.score)

            # Track for consistency
            if task_id not in answers_per_task:
                answers_per_task[task_id] = []
            answers_per_task[task_id].append(response)

            details.task_evaluations.append(
                TaskEvaluation(
                    task_id=task_id,
                    is_correct=is_correct,
                    expected_answer=str(expected),
                    actual_answer=response[:200],
                    reasoning_quality=rq,
                    response_text=response,
                )
            )

        # Compute metrics
        details.accuracy = correct_count / max(len(tasks), 1)
        details.reasoning_score = (
            sum(reasoning_scores) / max(len(reasoning_scores), 1)
        )
        details.consistency_score = self._compute_consistency(answers_per_task)

        # Section scores (heuristic based on prompt quality)
        details.section_scores = self._score_sections(genome)

        details.compute_composite()
        return details

    def _simple_check(self, response: str, expected: str) -> bool:
        """Simple answer checking by string containment."""
        expected_str = str(expected).strip().lower()
        response_lower = response.lower()
        return expected_str in response_lower

    def _assess_reasoning(self, response: str) -> ReasoningQuality:
        """Assess the reasoning quality of a response."""
        rq = ReasoningQuality()
        response_lower = response.lower()

        rq.has_step_by_step = any(
            marker in response_lower
            for marker in ["step 1", "step 2", "first", "then", "next"]
        )
        rq.has_formula_reference = any(
            marker in response_lower
            for marker in ["formula", "equation", "=", "where"]
        )
        rq.has_intermediate_values = any(
            marker in response_lower for marker in ["=", "equals", "gives us"]
        )
        rq.has_final_answer = any(
            marker in response_lower
            for marker in ["therefore", "answer is", "result is", "final"]
        )
        rq.has_verification = any(
            marker in response_lower
            for marker in ["verify", "check", "confirm", "double"]
        )

        rq.compute_score()
        return rq

    def _compute_consistency(
        self, answers_per_task: Dict[str, List[str]]
    ) -> float:
        """Compute consistency score across multiple evaluations of same tasks."""
        if not answers_per_task:
            return 1.0

        consistent_count = 0
        total = 0

        for task_id, answers in answers_per_task.items():
            if len(answers) <= 1:
                consistent_count += 1
                total += 1
                continue

            # Check if answers are similar
            for i in range(1, len(answers)):
                total += 1
                if self._answers_similar(answers[0], answers[i]):
                    consistent_count += 1

        return consistent_count / max(total, 1)

    def _answers_similar(self, a: str, b: str) -> bool:
        """Check if two answers are similar enough to be consistent."""
        # Extract numeric values and compare
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return True
        overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
        return overlap > 0.5

    def _score_sections(self, genome: "PromptGenome") -> Dict[str, float]:
        """Score individual sections for weakness analysis."""
        from src.genome.sections import SECTION_ORDER

        scores = {}
        for name in SECTION_ORDER:
            if name in genome.sections:
                content = genome.sections[name].content
                # Heuristic scoring
                score = 0.5  # Base score
                word_count = len(content.split())
                if word_count > 10:
                    score += 0.1
                if word_count > 30:
                    score += 0.1
                if any(t in content.lower() for t in ["step", "formula", "verify"]):
                    score += 0.1
                if "." in content:
                    score += 0.1
                scores[name] = min(score, 1.0)
            else:
                scores[name] = 0.0

        return scores
