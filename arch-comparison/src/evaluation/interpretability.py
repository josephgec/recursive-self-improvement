"""Interpretability evaluator: verifiability, faithfulness, readability."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InterpretabilityResult:
    """Result from interpretability evaluation."""
    system: str
    step_verifiability: float = 0.0
    faithfulness: float = 0.0
    readability: float = 0.0
    overall_score: float = 0.0
    per_result_scores: List[Dict[str, float]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class InterpretabilityEvaluator:
    """Evaluates how interpretable a system's reasoning is.

    Metrics:
    - Step verifiability: can each step be independently verified?
    - Faithfulness: does the reasoning chain actually lead to the answer?
    - Readability: is the reasoning easy for humans to follow?
    """

    CONNECTIVE_WORDS = [
        "therefore", "because", "since", "thus", "hence",
        "so", "implies", "if", "then", "consequently",
        "given", "follows",
    ]

    def __init__(
        self,
        verifiability_weight: float = 0.4,
        faithfulness_weight: float = 0.35,
        readability_weight: float = 0.25,
    ) -> None:
        self.verifiability_weight = verifiability_weight
        self.faithfulness_weight = faithfulness_weight
        self.readability_weight = readability_weight

    def evaluate(
        self,
        system: str,
        results: List[Any],
    ) -> InterpretabilityResult:
        """Evaluate interpretability across a set of results.

        Args:
            system: System name.
            results: List of pipeline results (HybridResult, IntegrativeResult, etc).

        Returns:
            InterpretabilityResult with aggregated scores.
        """
        per_result: List[Dict[str, float]] = []

        for result in results:
            v = self._score_step_verifiability(result)
            f = self._score_faithfulness(result)
            r = self._score_readability(result)
            per_result.append({
                "verifiability": v,
                "faithfulness": f,
                "readability": r,
            })

        n = len(per_result) if per_result else 1
        avg_v = sum(s["verifiability"] for s in per_result) / n
        avg_f = sum(s["faithfulness"] for s in per_result) / n
        avg_r = sum(s["readability"] for s in per_result) / n

        overall = (
            self.verifiability_weight * avg_v
            + self.faithfulness_weight * avg_f
            + self.readability_weight * avg_r
        )

        return InterpretabilityResult(
            system=system,
            step_verifiability=avg_v,
            faithfulness=avg_f,
            readability=avg_r,
            overall_score=overall,
            per_result_scores=per_result,
        )

    def _score_step_verifiability(self, result: Any) -> float:
        """Score step verifiability.

        Tool calls are inherently verifiable (inputs/outputs are concrete).
        Pure reasoning steps are less verifiable.
        """
        # Check for reasoning chain (hybrid)
        chain = getattr(result, "reasoning_chain", None)
        if chain and len(chain) > 0:
            verifiable = sum(
                1 for step in chain
                if step.step_type in ("tool_call", "tool_result")
            )
            return min(1.0, verifiable / max(len(chain), 1))

        # Check for constrained output (integrative)
        co = getattr(result, "constrained_output", None)
        if co is not None:
            # Constraints applied give partial verifiability
            n_constraints = len(co.constraints_applied)
            return min(1.0, 0.3 + n_constraints * 0.2)

        # Prose baseline: minimal verifiability
        return 0.1

    def _score_faithfulness(self, result: Any) -> float:
        """Score faithfulness: does the chain lead logically to the answer?

        For hybrid: check if the last tool result is reflected in the answer.
        For integrative: check if constrained output is consistent.
        For prose: basic heuristic.
        """
        answer = getattr(result, "answer", "")

        # Hybrid
        chain = getattr(result, "reasoning_chain", None)
        if chain and len(chain) > 1:
            # Check if conclusion references earlier content
            conclusion = chain[-1].content if chain else ""
            earlier_content = " ".join(
                step.content for step in chain[:-1]
                if step.content
            )
            # Check overlap
            answer_tokens = set(answer.lower().split())
            earlier_tokens = set(earlier_content.lower().split())
            if answer_tokens and earlier_tokens:
                overlap = len(answer_tokens & earlier_tokens) / max(len(answer_tokens), 1)
                return min(1.0, 0.3 + overlap)
            return 0.5

        # Integrative
        co = getattr(result, "constrained_output", None)
        if co is not None:
            # Fewer violations = more faithful
            violations = co.constraint_violations
            return max(0.0, 1.0 - violations * 0.2)

        # Prose baseline
        return 0.4

    def _score_readability(self, result: Any) -> float:
        """Score readability based on sentence structure.

        Criteria:
        - Average sentence length (shorter is more readable)
        - Presence of logical connectives
        """
        # Gather all text
        text = ""
        chain = getattr(result, "reasoning_chain", None)
        if chain:
            text = " ".join(step.content for step in chain if step.content)
        else:
            co = getattr(result, "constrained_output", None)
            if co is not None:
                text = co.text
            else:
                text = getattr(result, "answer", "")

        if not text:
            return 0.0

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        # Sentence length score (target: 5-25 words)
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_len <= 25:
            length_score = 1.0
        elif avg_len <= 40:
            length_score = 1.0 - (avg_len - 25) / 30.0
        else:
            length_score = 0.5

        # Connective score
        text_lower = text.lower()
        connective_count = sum(
            1 for w in self.CONNECTIVE_WORDS if w in text_lower
        )
        connective_score = min(1.0, connective_count / 2.0)

        return 0.6 * length_score + 0.4 * connective_score
