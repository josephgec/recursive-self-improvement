#!/usr/bin/env python3
"""Generate the full comparison report."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid.pipeline import HybridPipeline
from src.integrative.pipeline import IntegrativePipeline
from src.evaluation.benchmark_suite import BenchmarkSuite
from src.analysis.report import generate_report


def make_prose_pipeline():
    """Create a simple prose baseline pipeline."""
    import re
    from dataclasses import dataclass, field

    @dataclass
    class ProseResult:
        answer: str = ""
        correct: bool = False
        metadata: dict = field(default_factory=dict)

    class ProsePipeline:
        def solve(self, problem: str):
            arith = re.search(r"(\d+)\s*([\+\-\*])\s*(\d+)", problem)
            if arith:
                a, op, b = int(arith.group(1)), arith.group(2), int(arith.group(3))
                ops = {"+": a + b, "-": a - b, "*": a * b}
                result = ops.get(op, 0)
                return ProseResult(answer=str(result))
            return ProseResult(answer="unknown")

    return ProsePipeline()


def main() -> None:
    suite = BenchmarkSuite()
    suite.register_system("hybrid", HybridPipeline())
    suite.register_system("integrative", IntegrativePipeline())
    suite.register_system("prose", make_prose_pipeline())

    print("Running benchmark...")
    results = suite.run_full_suite(domains=["arithmetic", "algebra"])

    report = generate_report(results, title="Phase 1a Architecture Comparison")

    os.makedirs("reports", exist_ok=True)
    with open("reports/comparison_report.md", "w") as f:
        f.write(report)
    print("Report saved to reports/comparison_report.md")
    print("\nPreview:")
    print(report[:2000])


if __name__ == "__main__":
    main()
