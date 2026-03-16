"""Parallel condition runner using ThreadPoolExecutor."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from src.suites.base import AblationCondition, AblationSuite, AblationSuiteResult, ConditionRun
from src.execution.runner import MockPipeline


class ParallelConditionRunner:
    """Runs ablation conditions in parallel using ThreadPoolExecutor."""

    def __init__(self, max_workers: int = 4,
                 pipeline: Optional[Any] = None):
        self.max_workers = max_workers
        self.pipeline = pipeline or MockPipeline()

    def _run_single_condition(
        self,
        condition: AblationCondition,
        repetitions: int,
        seed: int,
    ) -> List[ConditionRun]:
        """Run a single condition for all repetitions."""
        runs = []
        for rep in range(repetitions):
            rep_seed = seed + rep
            accuracy = self.pipeline.run(condition, seed=rep_seed)
            runs.append(ConditionRun(
                condition_name=condition.name,
                repetition=rep,
                accuracy=accuracy,
                seed=rep_seed,
                metrics={"accuracy": accuracy},
            ))
        return runs

    def run_suite_parallel(
        self,
        suite: AblationSuite,
        repetitions: int = 5,
        seed: int = 42,
    ) -> AblationSuiteResult:
        """Run all conditions in a suite in parallel."""
        result = AblationSuiteResult(suite_name=suite.get_paper_name())
        conditions = suite.get_conditions()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for condition in conditions:
                future = executor.submit(
                    self._run_single_condition,
                    condition, repetitions, seed,
                )
                futures[future] = condition.name

            for future in as_completed(futures):
                condition_name = futures[future]
                runs = future.result()
                result.condition_runs[condition_name] = runs

        return result

    def run_multiple_suites(
        self,
        suites: List[AblationSuite],
        repetitions: int = 5,
        seed: int = 42,
    ) -> Dict[str, AblationSuiteResult]:
        """Run multiple suites, each suite's conditions in parallel."""
        results = {}
        for suite in suites:
            results[suite.get_paper_name()] = self.run_suite_parallel(
                suite, repetitions=repetitions, seed=seed
            )
        return results
