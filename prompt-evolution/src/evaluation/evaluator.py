"""Unified evaluator wrapping thinking and non-thinking evaluators."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.genome.prompt_genome import PromptGenome
from src.operators.thinking_evaluator import ThinkingEvaluator, FitnessDetails
from src.operators.non_thinking import NonThinkingEvaluator


class UnifiedEvaluator:
    """Unified evaluator that can use either thinking or non-thinking evaluation.

    Provides a single interface for the GA engine.
    """

    def __init__(
        self,
        llm_call: Callable[..., str],
        answer_checker: Optional[Any] = None,
        use_thinking: bool = True,
    ):
        self.use_thinking = use_thinking
        self.llm_call = llm_call

        if use_thinking:
            self._evaluator = ThinkingEvaluator(
                llm_call=llm_call,
                answer_checker=answer_checker,
            )
        else:
            self._non_thinking = NonThinkingEvaluator(
                answer_checker=answer_checker,
            )

    def evaluate(
        self,
        genome: PromptGenome,
        tasks: List[Dict[str, Any]],
    ) -> FitnessDetails:
        """Evaluate a genome on tasks.

        Delegates to the appropriate evaluator based on configuration.
        """
        if self.use_thinking:
            return self._evaluator.evaluate(genome, tasks)
        else:
            return self._non_thinking.evaluate(genome, tasks, llm_call=self.llm_call)
