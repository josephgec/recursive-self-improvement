"""RegressionSuite: per-benchmark evaluation tasks."""

from __future__ import annotations

from typing import Dict, List


class RegressionSuite:
    """Collection of benchmark tasks for regression testing."""

    BENCHMARKS = [
        "mmlu",
        "hellaswag",
        "arc_challenge",
        "truthfulqa",
        "winogrande",
        "gsm8k",
    ]

    def __init__(self) -> None:
        self._tasks = self._build_tasks()

    def load(self) -> Dict[str, List[dict]]:
        """Return tasks grouped by benchmark name."""
        return self._tasks

    def get_benchmark_names(self) -> List[str]:
        """Return list of benchmark names."""
        return list(self.BENCHMARKS)

    @staticmethod
    def _build_tasks() -> Dict[str, List[dict]]:
        """Build per-benchmark evaluation tasks."""
        tasks: Dict[str, List[dict]] = {}

        for bench in RegressionSuite.BENCHMARKS:
            bench_tasks: List[dict] = []
            for i in range(10):
                bench_tasks.append(
                    {
                        "id": f"{bench}_{i:03d}",
                        "benchmark": bench,
                        "prompt": f"Benchmark {bench} question {i}",
                        "expected_type": "choice",
                    }
                )
            tasks[bench] = bench_tasks

        return tasks
