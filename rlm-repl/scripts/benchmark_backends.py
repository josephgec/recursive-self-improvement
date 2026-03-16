#!/usr/bin/env python3
"""Benchmark different REPL backends."""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backends.factory import REPLFactory
from src.safety.policy import SafetyPolicy


BENCHMARK_CODE = [
    ("simple_assign", "x = 42"),
    ("arithmetic", "result = sum(range(1000))"),
    ("string_ops", "s = 'hello ' * 100; result = s.upper()"),
    ("list_comp", "result = [x**2 for x in range(100)]"),
    ("dict_comp", "result = {str(k): k**2 for k in range(100)}"),
    ("function_def", "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\nresult = fib(20)"),
]


def benchmark_backend(backend_name: str, iterations: int = 10):
    """Benchmark a single backend."""
    policy = SafetyPolicy()
    results = {}

    try:
        repl = REPLFactory.create(backend_name, policy)
    except Exception as e:
        print(f"  Skipping {backend_name}: {e}")
        return results

    for name, code in BENCHMARK_CODE:
        times = []
        for _ in range(iterations):
            start = time.time()
            try:
                result = repl.execute(code)
            except Exception:
                break
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

        if times:
            results[name] = {
                "avg_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
            }
            print(f"  {name}: avg={results[name]['avg_ms']:.2f}ms")

    try:
        repl.shutdown()
    except Exception:
        pass

    return results


def main():
    print("RLM-REPL Backend Benchmark")
    print("=" * 50)

    all_results = {}
    for backend in ["local"]:
        print(f"\nBackend: {backend}")
        all_results[backend] = benchmark_backend(backend)

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "benchmark_results", "latest.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
