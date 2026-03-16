"""Strategy classifier for RLM trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from src.strategies.code_pattern_detector import CodePatternDetector


class StrategyType(Enum):
    """The six core RLM strategies."""
    DIRECT = "DIRECT"
    PEEK_THEN_GREP = "PEEK_THEN_GREP"
    ITERATIVE_SEARCH = "ITERATIVE_SEARCH"
    MAP_REDUCE = "MAP_REDUCE"
    HIERARCHICAL = "HIERARCHICAL"
    HYBRID = "HYBRID"


@dataclass
class StrategyClassification:
    """Result of classifying a trajectory's strategy."""
    strategy: StrategyType
    confidence: float
    pattern_sequence: List[str]
    evidence: Dict[str, int]
    alternative: Optional[StrategyType] = None

    @property
    def strategy_name(self) -> str:
        return self.strategy.value


class StrategyClassifier:
    """Classify RLM trajectories into one of six strategies."""

    def __init__(self) -> None:
        self.detector = CodePatternDetector()

    def classify(
        self,
        trajectory: List[str],
        sub_sessions: Optional[List[List[str]]] = None,
    ) -> StrategyClassification:
        """Classify a trajectory into a strategy type.

        Strategy detection rules:
        - DIRECT: No search or chunking, just reads and answers
        - PEEK_THEN_GREP: Peek at structure, then grep for specifics
        - ITERATIVE_SEARCH: Loop + grep/search pattern
        - MAP_REDUCE: Chunk + loop + aggregate
        - HIERARCHICAL: Sub-queries + synthesis
        - HYBRID: Combination of multiple strategies
        """
        pattern_seq = self.detector.pattern_sequence(trajectory)
        counts = self.detector.pattern_counts(trajectory)

        # Count strategy-relevant patterns
        has_peek = counts.get("peek", 0) > 0
        has_grep = counts.get("grep", 0) > 0
        has_chunk = counts.get("chunk", 0) > 0
        has_sub_query = counts.get("sub_query", 0) > 0
        has_loop = counts.get("loop", 0) > 0
        has_aggregate = counts.get("aggregate", 0) > 0

        # Score each strategy
        scores: Dict[StrategyType, float] = {
            StrategyType.DIRECT: 0.0,
            StrategyType.PEEK_THEN_GREP: 0.0,
            StrategyType.ITERATIVE_SEARCH: 0.0,
            StrategyType.MAP_REDUCE: 0.0,
            StrategyType.HIERARCHICAL: 0.0,
            StrategyType.HYBRID: 0.0,
        }

        # DIRECT: simple, no complex patterns
        if not has_grep and not has_chunk and not has_sub_query and not has_loop:
            scores[StrategyType.DIRECT] = 0.8
        if len(trajectory) <= 2:
            scores[StrategyType.DIRECT] += 0.3

        # PEEK_THEN_GREP: peek followed by grep
        if has_peek and has_grep:
            scores[StrategyType.PEEK_THEN_GREP] = 0.9
        elif has_grep and not has_chunk and not has_sub_query:
            scores[StrategyType.PEEK_THEN_GREP] = 0.6

        # ITERATIVE_SEARCH: loop + search
        if has_loop and has_grep:
            scores[StrategyType.ITERATIVE_SEARCH] = 0.85
        if has_loop and not has_chunk:
            scores[StrategyType.ITERATIVE_SEARCH] += 0.2

        # MAP_REDUCE: chunk + process + aggregate
        if has_chunk and has_aggregate:
            scores[StrategyType.MAP_REDUCE] = 0.9
        if has_chunk and has_loop:
            scores[StrategyType.MAP_REDUCE] += 0.2

        # HIERARCHICAL: sub-queries + synthesis
        if has_sub_query:
            scores[StrategyType.HIERARCHICAL] = 0.85
        if has_sub_query and has_aggregate:
            scores[StrategyType.HIERARCHICAL] += 0.15

        # HYBRID: multiple high scores
        high_scores = sum(1 for s in scores.values() if s > 0.5)
        if high_scores >= 2:
            scores[StrategyType.HYBRID] = 0.7

        # Select best strategy
        best = max(scores, key=lambda s: scores[s])
        best_score = scores[best]

        # Default to DIRECT if no strong signal
        if best_score < 0.3:
            best = StrategyType.DIRECT
            best_score = 0.5

        # Find alternative
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        alternative = None
        if len(sorted_strategies) > 1 and sorted_strategies[1][1] > 0.3:
            alternative = sorted_strategies[1][0]

        confidence = min(best_score, 1.0)

        return StrategyClassification(
            strategy=best,
            confidence=confidence,
            pattern_sequence=pattern_seq,
            evidence=counts,
            alternative=alternative,
        )

    def classify_batch(
        self,
        trajectories: List[List[str]],
    ) -> List[StrategyClassification]:
        """Classify multiple trajectories."""
        return [self.classify(t) for t in trajectories]
