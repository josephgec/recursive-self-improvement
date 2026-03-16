"""StrategyDetector: classify the strategy used by an RLM session."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional



class Strategy(Enum):
    PEEK_THEN_GREP = "peek_then_grep"
    MAP_REDUCE = "map_reduce"
    HIERARCHICAL = "hierarchical"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    DIRECT = "direct"


@dataclass
class StrategyClassification:
    """The detected strategy and confidence."""
    strategy: Strategy
    confidence: float  # 0.0 - 1.0
    evidence: List[str]

    def __str__(self) -> str:
        return f"{self.strategy.value} (confidence={self.confidence:.2f})"


class StrategyDetector:
    """Analyze a session trajectory to classify the strategy used."""

    def classify(
        self,
        trajectory: List[Any],
        sub_sessions: Optional[List[Any]] = None,
    ) -> StrategyClassification:
        """Classify the strategy based on the trajectory and sub-sessions."""
        if not trajectory:
            return StrategyClassification(
                strategy=Strategy.DIRECT, confidence=1.0, evidence=["empty trajectory"]
            )

        all_code = " ".join(
            code for step in trajectory for code in step.code_blocks
        )

        # Check for hierarchical (sub-queries)
        if sub_sessions and len(sub_sessions) > 0:
            return StrategyClassification(
                strategy=Strategy.HIERARCHICAL,
                confidence=0.9,
                evidence=["sub-sessions detected", f"{len(sub_sessions)} sub-queries"],
            )

        if "rlm_sub_query" in all_code:
            return StrategyClassification(
                strategy=Strategy.HIERARCHICAL,
                confidence=0.85,
                evidence=["rlm_sub_query call found in code"],
            )

        has_peek = "peek(" in all_code
        has_grep = "grep(" in all_code or "search(" in all_code
        has_chunk = "chunk(" in all_code
        has_aggregate = "aggregate" in all_code.lower() or "combine" in all_code.lower()

        # Peek then grep
        if has_peek and has_grep:
            return StrategyClassification(
                strategy=Strategy.PEEK_THEN_GREP,
                confidence=0.85,
                evidence=["peek() and grep()/search() both used"],
            )

        # Map-reduce
        if has_chunk or (has_aggregate and len(trajectory) > 2):
            return StrategyClassification(
                strategy=Strategy.MAP_REDUCE,
                confidence=0.8,
                evidence=[
                    "chunk() used" if has_chunk else "aggregation pattern detected",
                    f"{len(trajectory)} iterations",
                ],
            )

        # Iterative refinement
        if len(trajectory) > 3:
            return StrategyClassification(
                strategy=Strategy.ITERATIVE_REFINEMENT,
                confidence=0.7,
                evidence=[f"{len(trajectory)} iterations without chunk/sub-query"],
            )

        # Grep only
        if has_grep and not has_peek:
            return StrategyClassification(
                strategy=Strategy.PEEK_THEN_GREP,
                confidence=0.6,
                evidence=["grep/search used without peek"],
            )

        # Direct
        return StrategyClassification(
            strategy=Strategy.DIRECT,
            confidence=0.8,
            evidence=[
                f"{len(trajectory)} iteration(s)",
                "no complex strategy detected",
            ],
        )
