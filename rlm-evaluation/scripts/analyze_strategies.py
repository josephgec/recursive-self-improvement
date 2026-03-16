#!/usr/bin/env python3
"""Analyze RLM strategies from evaluation results."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.registry import create_default_registry
from src.execution.rlm_executor import RLMExecutor
from src.strategies.emergence_analyzer import EmergenceAnalyzer
from src.strategies.effectiveness import StrategyEffectivenessAnalyzer


def main() -> None:
    """Run strategy analysis."""
    registry = create_default_registry()
    tasks = registry.load_all()

    rlm = RLMExecutor()
    results = [rlm.execute(t) for t in tasks]

    task_categories = {t.task_id: t.category for t in tasks}

    # Emergence analysis
    analyzer = EmergenceAnalyzer()
    report = analyzer.analyze(results, task_categories)
    print(report.summary())

    # Effectiveness
    eff = StrategyEffectivenessAnalyzer()
    optimal = eff.optimal_strategy_map(results, task_categories)
    print("\nOptimal strategies per category:")
    for cat, strat in optimal.items():
        print(f"  {cat}: {strat}")


if __name__ == "__main__":
    main()
