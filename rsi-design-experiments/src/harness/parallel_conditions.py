"""Parallel condition runner using ThreadPoolExecutor."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.experiments.base import ConditionResult


class ParallelConditionRunner:
    """Runs multiple conditions concurrently using ThreadPoolExecutor."""

    def __init__(self, max_workers: Optional[int] = None):
        self._max_workers = max_workers

    def run(
        self,
        condition_funcs: List[Tuple[str, Callable[[], ConditionResult]]],
    ) -> Dict[str, ConditionResult]:
        """Run condition functions in parallel.

        Args:
            condition_funcs: list of (condition_name, callable) pairs.
                Each callable should return a ConditionResult.

        Returns:
            Dict mapping condition names to their results.
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_name = {}
            for name, func in condition_funcs:
                future = executor.submit(func)
                future_to_name[future] = name

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    results[name] = result
                except Exception as e:
                    # On error, store a failed result
                    results[name] = ConditionResult(
                        condition_name=name,
                        metadata={"error": str(e)},
                    )

        return results

    def run_repetitions(
        self,
        func: Callable[[int], ConditionResult],
        repetitions: int,
    ) -> List[ConditionResult]:
        """Run the same function multiple times in parallel with different rep indices.

        Args:
            func: callable that takes a repetition index and returns ConditionResult.
            repetitions: number of repetitions.

        Returns:
            List of ConditionResults, one per repetition.
        """
        results = [None] * repetitions

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {}
            for i in range(repetitions):
                future = executor.submit(func, i)
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = ConditionResult(
                        condition_name=f"rep_{idx}",
                        metadata={"error": str(e)},
                    )

        return results  # type: ignore
