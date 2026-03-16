"""Parallel execution of evaluation tasks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

from src.benchmarks.task import EvalTask, EvalResult


class ParallelExecutor:
    """Execute tasks in parallel using ThreadPoolExecutor."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def execute_all(
        self,
        tasks: List[EvalTask],
        executor_fn: Callable[[EvalTask], EvalResult],
        on_complete: Optional[Callable[[EvalResult], None]] = None,
    ) -> List[EvalResult]:
        """Execute all tasks in parallel.

        Args:
            tasks: List of tasks to execute.
            executor_fn: Function that takes a task and returns a result.
            on_complete: Optional callback invoked when each task completes.

        Returns:
            List of results in completion order.
        """
        results: List[EvalResult] = []

        if not tasks:
            return results

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_task = {
                pool.submit(executor_fn, task): task
                for task in tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    if on_complete is not None:
                        on_complete(result)
                except Exception as e:
                    error_result = EvalResult(
                        task_id=task.task_id,
                        benchmark=task.benchmark,
                        answer="",
                        correct=False,
                        error=str(e),
                    )
                    results.append(error_result)
                    if on_complete is not None:
                        on_complete(error_result)

        return results

    def execute_batch(
        self,
        tasks: List[EvalTask],
        executor_fn: Callable[[EvalTask], EvalResult],
        batch_size: int = 10,
    ) -> List[EvalResult]:
        """Execute tasks in batches for better resource management.

        Args:
            tasks: All tasks to execute.
            executor_fn: Function that takes a task and returns a result.
            batch_size: Number of tasks per batch.

        Returns:
            List of all results.
        """
        all_results: List[EvalResult] = []

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = self.execute_all(batch, executor_fn)
            all_results.extend(batch_results)

        return all_results
