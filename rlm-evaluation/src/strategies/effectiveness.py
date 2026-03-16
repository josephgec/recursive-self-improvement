"""Analyze strategy effectiveness across categories."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from src.benchmarks.task import EvalResult


class StrategyEffectivenessAnalyzer:
    """Analyze how effective each strategy is for different task categories."""

    def accuracy_by_strategy_and_category(
        self,
        results: List[EvalResult],
        task_categories: Dict[str, str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute accuracy for each (strategy, category) pair.

        Returns:
            Nested dict: strategy -> category -> accuracy
        """
        correct: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for r in results:
            strategy = r.strategy_detected or "unknown"
            category = task_categories.get(r.task_id, "unknown")
            total[strategy][category] += 1
            if r.correct:
                correct[strategy][category] += 1

        accuracy: Dict[str, Dict[str, float]] = {}
        for strat in total:
            accuracy[strat] = {}
            for cat in total[strat]:
                accuracy[strat][cat] = (
                    correct[strat][cat] / total[strat][cat]
                    if total[strat][cat] > 0 else 0.0
                )
        return accuracy

    def cost_by_strategy(
        self,
        results: List[EvalResult],
    ) -> Dict[str, float]:
        """Compute average cost per task for each strategy."""
        costs: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            strategy = r.strategy_detected or "unknown"
            costs[strategy].append(r.cost)

        return {
            s: sum(c) / len(c) if c else 0.0
            for s, c in costs.items()
        }

    def optimal_strategy_map(
        self,
        results: List[EvalResult],
        task_categories: Dict[str, str],
    ) -> Dict[str, str]:
        """For each category, find the optimal strategy (highest accuracy).

        Returns:
            Dict mapping category -> best strategy.
        """
        acc = self.accuracy_by_strategy_and_category(results, task_categories)

        # Invert: category -> strategy -> accuracy
        cat_strat: Dict[str, Dict[str, float]] = defaultdict(dict)
        for strat, cats in acc.items():
            for cat, a in cats.items():
                cat_strat[cat][strat] = a

        optimal: Dict[str, str] = {}
        for cat, strats in cat_strat.items():
            if strats:
                optimal[cat] = max(strats, key=lambda s: strats[s])
        return optimal

    def cost_effectiveness(
        self,
        results: List[EvalResult],
    ) -> Dict[str, float]:
        """Compute cost-effectiveness (accuracy / avg_cost) per strategy."""
        costs: Dict[str, List[float]] = defaultdict(list)
        correct: Dict[str, int] = defaultdict(int)
        total: Dict[str, int] = defaultdict(int)

        for r in results:
            strategy = r.strategy_detected or "unknown"
            costs[strategy].append(r.cost)
            total[strategy] += 1
            if r.correct:
                correct[strategy] += 1

        effectiveness: Dict[str, float] = {}
        for s in total:
            avg_cost = sum(costs[s]) / len(costs[s]) if costs[s] else 0.001
            accuracy = correct[s] / total[s] if total[s] > 0 else 0.0
            effectiveness[s] = accuracy / avg_cost if avg_cost > 0 else 0.0
        return effectiveness
