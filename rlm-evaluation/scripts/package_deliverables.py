#!/usr/bin/env python3
"""Package Phase 2b deliverables."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.registry import create_default_registry
from src.execution.rlm_executor import RLMExecutor
from src.execution.standard_executor import StandardExecutor
from src.deliverables.rlm_wrapper_summary import RLMWrapperSummary
from src.deliverables.repl_summary import REPLSummary
from src.deliverables.final_report import FinalReport
from src.comparison.cost_model import CostModel
from src.comparison.head_to_head import HeadToHeadComparator
from src.strategies.emergence_analyzer import EmergenceAnalyzer


def main() -> None:
    """Package deliverables."""
    registry = create_default_registry()
    tasks = registry.load_all()

    rlm = RLMExecutor()
    std = StandardExecutor()
    rlm_results = [rlm.execute(t) for t in tasks]
    std_results = [std.execute(t) for t in tasks]
    task_categories = {t.task_id: t.category for t in tasks}

    # Generate deliverables
    wrapper = RLMWrapperSummary(rlm_results)
    repl = REPLSummary(rlm_results, registry.available_benchmarks)

    h2h = HeadToHeadComparator().compare(rlm_results, std_results, task_categories)
    cost_cmp = CostModel().compare_systems(rlm_results, std_results)
    emergence = EmergenceAnalyzer().analyze(rlm_results, task_categories)

    report = FinalReport(rlm_results, std_results, cost_cmp, h2h, emergence)

    output_dir = "data/reports"
    os.makedirs(output_dir, exist_ok=True)

    for name, content in [
        ("rlm_wrapper_summary.md", wrapper.generate()),
        ("repl_summary.md", repl.generate()),
        ("final_report.md", report.generate()),
    ]:
        path = os.path.join(output_dir, name)
        with open(path, "w") as f:
            f.write(content)
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
