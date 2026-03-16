#!/usr/bin/env python3
"""Run a single evaluation axis."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid.pipeline import HybridPipeline
from src.integrative.pipeline import IntegrativePipeline
from src.evaluation.benchmark_suite import BenchmarkSuite


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
    import argparse
    parser = argparse.ArgumentParser(description="Run single evaluation axis")
    parser.add_argument("--axis", default="generalization",
                       choices=["generalization", "interpretability", "robustness"])
    parser.add_argument("--domain", default="arithmetic")
    args = parser.parse_args()

    suite = BenchmarkSuite()
    suite.register_system("hybrid", HybridPipeline())
    suite.register_system("integrative", IntegrativePipeline())
    suite.register_system("prose", make_prose_pipeline())

    print(f"Running {args.axis} evaluation on {args.domain}...")
    results = suite.run_single_axis(args.axis, domains=[args.domain])

    for system, result in results.items():
        print(f"\n{system}:")
        for attr in dir(result):
            if not attr.startswith("_") and attr not in ("metadata",):
                val = getattr(result, attr)
                if isinstance(val, (int, float, str)):
                    print(f"  {attr}: {val}")


if __name__ == "__main__":
    main()
