"""Analyze emergent strategy patterns across evaluation results."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.benchmarks.task import EvalResult
from src.strategies.classifier import StrategyClassifier, StrategyType


@dataclass
class EmergenceReport:
    """Report on emergent strategy patterns."""
    strategy_by_task_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    strategy_by_context_size: Dict[str, Dict[str, int]] = field(default_factory=dict)
    strategy_effectiveness: Dict[str, float] = field(default_factory=dict)
    grep_before_read_rate: float = 0.0
    adaptation_score: float = 0.0
    dominant_strategies: Dict[str, str] = field(default_factory=dict)
    total_results_analyzed: int = 0

    def summary(self) -> str:
        """Generate a text summary of the emergence report."""
        lines = [
            f"Emergence Report ({self.total_results_analyzed} results analyzed)",
            f"  Grep-before-read rate: {self.grep_before_read_rate:.1%}",
            f"  Adaptation score: {self.adaptation_score:.2f}",
            "",
            "Strategy by task type:",
        ]
        for task_type, strats in self.strategy_by_task_type.items():
            lines.append(f"  {task_type}: {strats}")
        lines.append("")
        lines.append("Strategy effectiveness (accuracy):")
        for strat, acc in self.strategy_effectiveness.items():
            lines.append(f"  {strat}: {acc:.1%}")
        return "\n".join(lines)


class EmergenceAnalyzer:
    """Analyze emergent strategy patterns from evaluation results."""

    def __init__(self) -> None:
        self.classifier = StrategyClassifier()

    def analyze(
        self,
        results: List[EvalResult],
        task_categories: Optional[Dict[str, str]] = None,
        context_sizes: Optional[Dict[str, int]] = None,
    ) -> EmergenceReport:
        """Run full emergence analysis on results.

        Args:
            results: List of evaluation results with trajectories.
            task_categories: Mapping of task_id -> category.
            context_sizes: Mapping of task_id -> context token count.

        Returns:
            EmergenceReport with all findings.
        """
        report = EmergenceReport(total_results_analyzed=len(results))

        if not results:
            return report

        report.strategy_by_task_type = self.strategy_by_task_type(results, task_categories)
        report.strategy_by_context_size = self.strategy_by_context_size(results, context_sizes)
        report.strategy_effectiveness = self.strategy_effectiveness(results)
        report.grep_before_read_rate = self.grep_before_read_rate(results)
        report.adaptation_score = self.adaptation_within_session(results)
        report.dominant_strategies = self._find_dominant_strategies(report.strategy_by_task_type)

        return report

    def strategy_by_task_type(
        self,
        results: List[EvalResult],
        task_categories: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Map strategy distribution per task type/category."""
        dist: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for r in results:
            category = "unknown"
            if task_categories and r.task_id in task_categories:
                category = task_categories[r.task_id]
            elif r.metadata and "category" in r.metadata:
                category = r.metadata["category"]
            else:
                # Try to extract from task_id
                for cat in ["retrieval", "aggregation", "counting", "reasoning", "needle", "refactoring", "bug_fix"]:
                    if cat in r.task_id:
                        category = cat
                        break

            strategy = r.strategy_detected or "unknown"
            dist[category][strategy] += 1

        return {k: dict(v) for k, v in dist.items()}

    def strategy_by_context_size(
        self,
        results: List[EvalResult],
        context_sizes: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Map strategy distribution by context size bucket."""
        buckets = ["small (<4k)", "medium (4k-16k)", "large (16k-64k)", "xlarge (>64k)"]
        dist: Dict[str, Dict[str, int]] = {b: defaultdict(int) for b in buckets}

        for r in results:
            size = 0
            if context_sizes and r.task_id in context_sizes:
                size = context_sizes[r.task_id]
            elif r.input_tokens > 0:
                size = r.input_tokens

            bucket = self._size_bucket(size)
            strategy = r.strategy_detected or "unknown"
            dist[bucket][strategy] += 1

        return {k: dict(v) for k, v in dist.items()}

    def strategy_effectiveness(self, results: List[EvalResult]) -> Dict[str, float]:
        """Compute accuracy for each strategy type."""
        correct_by_strat: Dict[str, int] = defaultdict(int)
        total_by_strat: Dict[str, int] = defaultdict(int)

        for r in results:
            strategy = r.strategy_detected or "unknown"
            total_by_strat[strategy] += 1
            if r.correct:
                correct_by_strat[strategy] += 1

        return {
            s: correct_by_strat[s] / total_by_strat[s]
            for s in total_by_strat
        }

    def grep_before_read_rate(self, results: List[EvalResult]) -> float:
        """Compute the rate at which grep appears before read/cat in trajectories."""
        count = 0
        total = 0

        for r in results:
            if not r.trajectory:
                continue
            total += 1
            grep_idx = -1
            read_idx = -1
            for i, step in enumerate(r.trajectory):
                step_lower = step.lower()
                if grep_idx < 0 and ("grep" in step_lower or "search" in step_lower):
                    grep_idx = i
                if read_idx < 0 and ("read" in step_lower or "cat " in step_lower or "sed " in step_lower):
                    read_idx = i
            if grep_idx >= 0 and (read_idx < 0 or grep_idx < read_idx):
                count += 1

        return count / total if total > 0 else 0.0

    def adaptation_within_session(self, results: List[EvalResult]) -> float:
        """Measure how much strategy varies within a session (adaptation score).

        Higher score means more adaptive behavior.
        """
        if not results:
            return 0.0

        # Count how many different strategies are used
        strategies_used = set()
        for r in results:
            if r.strategy_detected:
                strategies_used.add(r.strategy_detected)

        # Normalize: 1 strategy = 0, all 6 = 1
        max_strategies = 6
        return min((len(strategies_used) - 1) / (max_strategies - 1), 1.0) if len(strategies_used) > 1 else 0.0

    def _size_bucket(self, tokens: int) -> str:
        """Classify token count into a size bucket."""
        if tokens < 4000:
            return "small (<4k)"
        elif tokens < 16000:
            return "medium (4k-16k)"
        elif tokens < 64000:
            return "large (16k-64k)"
        else:
            return "xlarge (>64k)"

    def _find_dominant_strategies(
        self, by_type: Dict[str, Dict[str, int]]
    ) -> Dict[str, str]:
        """Find the dominant strategy for each task type."""
        dominant: Dict[str, str] = {}
        for task_type, strats in by_type.items():
            if strats:
                dominant[task_type] = max(strats, key=lambda s: strats[s])
        return dominant
