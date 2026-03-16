"""Task generation utilities."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.evaluation.financial_math import FinancialMathBenchmark, FinancialTask


def generate_task_batch(
    category: str,
    n: int = 10,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Generate a batch of tasks for a specific category.

    Args:
        category: One of FinancialMathBenchmark.CATEGORIES
        n: Number of tasks to generate
        seed: Random seed for reproducibility

    Returns:
        List of task dicts with question, expected_answer, task_id.
    """
    bench = FinancialMathBenchmark(seed=seed)
    bench.generate_tasks(n_per_category=n)

    tasks = bench.get_tasks_by_category(category)
    return bench.to_eval_tasks(tasks[:n])


def generate_mixed_batch(
    n: int = 20,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Generate a mixed batch of tasks from all categories.

    Args:
        n: Total number of tasks desired
        seed: Random seed for reproducibility

    Returns:
        List of task dicts.
    """
    bench = FinancialMathBenchmark(seed=seed)
    per_cat = max(1, n // len(bench.CATEGORIES))
    bench.generate_tasks(n_per_category=per_cat)

    all_tasks = bench.load()
    return bench.to_eval_tasks(all_tasks[:n])
