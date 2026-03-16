#!/usr/bin/env python3
"""Generate comprehensive evaluation report."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.registry import create_default_registry
from src.execution.rlm_executor import RLMExecutor
from src.execution.standard_executor import StandardExecutor
from src.analysis.report import ReportGenerator


def main() -> None:
    """Generate report."""
    registry = create_default_registry()
    tasks = registry.load_all()

    rlm = RLMExecutor()
    std = StandardExecutor()
    rlm_results = [rlm.execute(t) for t in tasks]
    std_results = [std.execute(t) for t in tasks]
    task_categories = {t.task_id: t.category for t in tasks}

    generator = ReportGenerator()
    report = generator.generate(rlm_results, std_results, task_categories)

    output_dir = "data/reports"
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "evaluation_report.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"Report generated: {path}")
    print(f"\nReport length: {len(report)} characters")


if __name__ == "__main__":
    main()
