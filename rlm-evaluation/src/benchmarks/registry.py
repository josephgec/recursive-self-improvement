"""Benchmark registry for loading and filtering tasks."""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol

from src.benchmarks.task import EvalTask


class BenchmarkLoader(Protocol):
    """Protocol for benchmark loaders."""

    name: str

    def load(self) -> List[EvalTask]:
        """Load tasks from this benchmark."""
        ...


class BenchmarkRegistry:
    """Central registry for all benchmarks."""

    def __init__(self) -> None:
        self._benchmarks: Dict[str, BenchmarkLoader] = {}

    def register(self, loader: BenchmarkLoader) -> None:
        """Register a benchmark loader."""
        self._benchmarks[loader.name] = loader

    def load(self, benchmark: str) -> List[EvalTask]:
        """Load tasks from a specific benchmark."""
        if benchmark not in self._benchmarks:
            raise KeyError(f"Unknown benchmark: {benchmark}. Available: {list(self._benchmarks.keys())}")
        return self._benchmarks[benchmark].load()

    def load_all(self) -> List[EvalTask]:
        """Load tasks from all registered benchmarks."""
        tasks: List[EvalTask] = []
        for loader in self._benchmarks.values():
            tasks.extend(loader.load())
        return tasks

    def filter(
        self,
        tasks: List[EvalTask],
        category: Optional[str] = None,
        min_context_size: Optional[int] = None,
        max_context_size: Optional[int] = None,
        difficulty: Optional[str] = None,
        benchmark: Optional[str] = None,
    ) -> List[EvalTask]:
        """Filter tasks by criteria."""
        filtered = tasks
        if category is not None:
            filtered = [t for t in filtered if t.category == category]
        if min_context_size is not None:
            filtered = [t for t in filtered if t.context_tokens >= min_context_size]
        if max_context_size is not None:
            filtered = [t for t in filtered if t.context_tokens <= max_context_size]
        if difficulty is not None:
            filtered = [t for t in filtered if t.difficulty == difficulty]
        if benchmark is not None:
            filtered = [t for t in filtered if t.benchmark == benchmark]
        return filtered

    @property
    def available_benchmarks(self) -> List[str]:
        """List registered benchmark names."""
        return list(self._benchmarks.keys())


def create_default_registry() -> BenchmarkRegistry:
    """Create a registry with all default benchmarks."""
    from src.benchmarks.oolong import OOLONGBenchmark
    from src.benchmarks.locodiff import LoCoDiffBenchmark
    from src.benchmarks.synthetic import SyntheticTaskGenerator

    registry = BenchmarkRegistry()
    registry.register(OOLONGBenchmark())
    registry.register(LoCoDiffBenchmark())
    registry.register(SyntheticTaskGenerator())
    return registry
