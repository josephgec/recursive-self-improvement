#!/usr/bin/env python3
"""Run full 3-way comparison between hybrid, integrative, and prose."""

from __future__ import annotations

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid.pipeline import HybridPipeline
from src.integrative.pipeline import IntegrativePipeline
from src.evaluation.benchmark_suite import BenchmarkSuite
from src.analysis.report import generate_report


def make_prose_pipeline():
    """Create a simple prose baseline pipeline."""
    import re

    class ProsePipeline:
        def solve(self, problem: str):
            from dataclasses import dataclass, field

            @dataclass
            class ProseResult:
                answer: str = ""
                correct: bool = False
                metadata: dict = field(default_factory=dict)

            # Simple pattern matching
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

    print("Running full benchmark suite...")
    results = suite.run_full_suite(domains=["arithmetic", "algebra"])

    report = generate_report(results)
    print(report)

    # Save results
    os.makedirs("data/output", exist_ok=True)
    summary = results.get_summary()
    with open("data/output/comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nResults saved to data/output/comparison_results.json")


if __name__ == "__main__":
    main()
