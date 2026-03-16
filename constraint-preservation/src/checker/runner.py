"""ConstraintRunner: runs all constraints in parallel with optional caching."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from src.constraints.base import CheckContext, ConstraintResult
from src.checker.suite import ConstraintSuite
from src.checker.verdict import SuiteVerdict
from src.checker.cache import ConstraintCache


class ConstraintRunner:
    """Execute an entire ConstraintSuite against an agent state."""

    def __init__(
        self,
        suite: ConstraintSuite,
        parallel: bool = True,
        max_workers: int = 4,
        cache: Optional[ConstraintCache] = None,
    ) -> None:
        self._suite = suite
        self._parallel = parallel
        self._max_workers = max_workers
        self._cache = cache

    def run(
        self,
        agent_state: Any,
        context: Optional[CheckContext] = None,
    ) -> SuiteVerdict:
        """Run all constraints and return a SuiteVerdict."""
        if context is None:
            context = CheckContext()

        results: Dict[str, ConstraintResult] = {}

        if self._parallel and len(self._suite) > 1:
            results = self._run_parallel(agent_state, context)
        else:
            results = self._run_sequential(agent_state, context)

        passed = all(r.satisfied for r in results.values())
        return SuiteVerdict(passed=passed, results=results)

    def _run_sequential(
        self, agent_state: Any, context: CheckContext
    ) -> Dict[str, ConstraintResult]:
        results: Dict[str, ConstraintResult] = {}
        for constraint in self._suite:
            result = self._check_with_cache(constraint, agent_state, context)
            results[constraint.name] = result
        return results

    def _run_parallel(
        self, agent_state: Any, context: CheckContext
    ) -> Dict[str, ConstraintResult]:
        results: Dict[str, ConstraintResult] = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(
                    self._check_with_cache, constraint, agent_state, context
                ): constraint.name
                for constraint in self._suite
            }
            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()
        return results

    def _check_with_cache(
        self, constraint: Any, agent_state: Any, context: CheckContext
    ) -> ConstraintResult:
        if self._cache is not None:
            cached = self._cache.get(constraint.name)
            if cached is not None:
                return cached

        result = constraint.check(agent_state, context)

        if self._cache is not None:
            self._cache.put(constraint.name, result)

        return result
