"""Concurrent task search using thread pool."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.arc.grid import ARCTask
from src.search.engine import EvolutionarySearchEngine, SearchConfig, SearchResult
from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ParallelSearchResult:
    """Results from searching multiple tasks in parallel."""

    results: Dict[str, SearchResult] = field(default_factory=dict)
    total_time: float = 0.0

    @property
    def num_solved(self) -> int:
        return sum(1 for r in self.results.values() if r.solved)

    @property
    def num_tasks(self) -> int:
        return len(self.results)

    @property
    def solve_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.num_solved / self.num_tasks

    def summary(self) -> str:
        return (
            f"ParallelSearchResult(tasks={self.num_tasks}, "
            f"solved={self.num_solved}, rate={self.solve_rate:.1%}, "
            f"time={self.total_time:.1f}s)"
        )


class ParallelTaskSearch:
    """Run evolutionary search on multiple tasks concurrently."""

    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        max_workers: int = 1,
        llm_call: Optional[Callable[[str], str]] = None,
    ):
        self.config = config or SearchConfig()
        self.max_workers = max_workers
        self.llm_call = llm_call

    def _search_task(self, task: ARCTask) -> SearchResult:
        """Search a single task (for use in thread pool)."""
        engine = EvolutionarySearchEngine(
            config=self.config,
            llm_call=self.llm_call,
        )
        return engine.search(task)

    def search_all(self, tasks: List[ARCTask]) -> ParallelSearchResult:
        """Search all tasks, potentially in parallel."""
        import time

        start = time.time()
        result = ParallelSearchResult()

        if self.max_workers <= 1:
            # Sequential execution
            for task in tasks:
                logger.info(f"Searching task: {task.task_id}")
                sr = self._search_task(task)
                result.results[task.task_id] = sr
        else:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                future_to_task = {
                    executor.submit(self._search_task, task): task
                    for task in tasks
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        sr = future.result()
                        result.results[task.task_id] = sr
                    except Exception as e:
                        logger.error(
                            f"Task {task.task_id} failed: {e}"
                        )
                        result.results[task.task_id] = SearchResult(
                            stop_reason=f"error: {e}"
                        )

        result.total_time = time.time() - start
        logger.info(result.summary())
        return result
