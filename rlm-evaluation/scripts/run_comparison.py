#!/usr/bin/env python3
"""Run head-to-head comparison between RLM and standard."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.registry import create_default_registry
from src.execution.rlm_executor import RLMExecutor
from src.execution.standard_executor import StandardExecutor
from src.comparison.head_to_head import HeadToHeadComparator
from src.comparison.cost_model import CostModel


def main() -> None:
    """Run comparison."""
    registry = create_default_registry()
    tasks = registry.load_all()

    rlm = RLMExecutor()
    std = StandardExecutor()

    print(f"Evaluating {len(tasks)} tasks with both systems...")
    rlm_results = [rlm.execute(t) for t in tasks]
    std_results = [std.execute(t) for t in tasks]

    # Head-to-head
    comparator = HeadToHeadComparator()
    report = comparator.compare(rlm_results, std_results)
    print("\n" + report.summary())

    # Cost comparison
    cost_model = CostModel()
    cost_cmp = cost_model.compare_systems(rlm_results, std_results)
    print("\n" + cost_cmp.summary())


if __name__ == "__main__":
    main()
