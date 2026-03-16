"""Track strategy evolution within and across sessions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.benchmarks.task import EvalResult
from src.strategies.classifier import StrategyClassifier, StrategyType


@dataclass
class StrategyTransition:
    """A transition from one strategy to another."""
    from_strategy: str
    to_strategy: str
    task_from: str
    task_to: str


class StrategyEvolutionTracker:
    """Track how strategies evolve within sessions and across runs."""

    def __init__(self) -> None:
        self.classifier = StrategyClassifier()

    def track_within_session(
        self,
        results: List[EvalResult],
    ) -> List[StrategyTransition]:
        """Track strategy transitions within a session (sequential results).

        Returns list of transitions between consecutive tasks.
        """
        transitions: List[StrategyTransition] = []

        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]
            prev_strat = prev.strategy_detected or "unknown"
            curr_strat = curr.strategy_detected or "unknown"

            if prev_strat != curr_strat:
                transitions.append(StrategyTransition(
                    from_strategy=prev_strat,
                    to_strategy=curr_strat,
                    task_from=prev.task_id,
                    task_to=curr.task_id,
                ))

        return transitions

    def transition_matrix(
        self,
        results: List[EvalResult],
    ) -> Dict[str, Dict[str, int]]:
        """Build a transition matrix of strategy changes.

        Returns:
            Dict[from_strategy][to_strategy] -> count
        """
        matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for i in range(1, len(results)):
            prev_strat = results[i - 1].strategy_detected or "unknown"
            curr_strat = results[i].strategy_detected or "unknown"
            matrix[prev_strat][curr_strat] += 1

        return {k: dict(v) for k, v in matrix.items()}

    def strategy_timeline(
        self,
        results: List[EvalResult],
    ) -> List[Tuple[str, str]]:
        """Get the timeline of (task_id, strategy) pairs.

        Returns:
            List of (task_id, strategy_name) tuples in order.
        """
        return [
            (r.task_id, r.strategy_detected or "unknown")
            for r in results
        ]

    def adaptation_events(
        self,
        results: List[EvalResult],
    ) -> List[Dict[str, str]]:
        """Identify adaptation events where strategy changed after failure.

        An adaptation event is when a failure is followed by a strategy change.
        """
        events: List[Dict[str, str]] = []

        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]

            if not prev.correct:
                prev_strat = prev.strategy_detected or "unknown"
                curr_strat = curr.strategy_detected or "unknown"
                if prev_strat != curr_strat:
                    events.append({
                        "failed_task": prev.task_id,
                        "failed_strategy": prev_strat,
                        "adapted_task": curr.task_id,
                        "new_strategy": curr_strat,
                        "adaptation_successful": str(curr.correct),
                    })

        return events

    def strategy_diversity(self, results: List[EvalResult]) -> float:
        """Measure diversity of strategies used (0 = all same, 1 = all different).

        Uses normalized entropy.
        """
        if not results:
            return 0.0

        counts: Dict[str, int] = defaultdict(int)
        for r in results:
            counts[r.strategy_detected or "unknown"] += 1

        total = len(results)
        if total == 0 or len(counts) <= 1:
            return 0.0

        import math
        entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in counts.values()
            if c > 0
        )
        max_entropy = math.log2(len(counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0
