"""Benchmark registry for loading and evaluating benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    task_id: str
    benchmark: str
    category: str
    prompt: str
    expected_answer: Any
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result from evaluating a single task."""
    task_id: str
    benchmark: str
    correct: bool
    predicted_answer: Any = None
    expected_answer: Any = None
    score: float = 0.0
    time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentProtocol(Protocol):
    """Protocol that agents must implement."""

    def solve(self, task: BenchmarkTask) -> Any:
        ...


class BaseBenchmark:
    """Base class for all benchmarks."""

    name: str = "base"

    def __init__(self) -> None:
        self._tasks: List[BenchmarkTask] = []
        self._build_tasks()

    def _build_tasks(self) -> None:
        """Subclasses override to populate self._tasks."""
        pass

    @property
    def tasks(self) -> List[BenchmarkTask]:
        return list(self._tasks)

    def get_tasks(self, category: Optional[str] = None) -> List[BenchmarkTask]:
        if category is None:
            return list(self._tasks)
        return [t for t in self._tasks if t.category == category]

    @property
    def categories(self) -> List[str]:
        return sorted(set(t.category for t in self._tasks))

    def evaluate(self, agent: Any, tasks: Optional[List[BenchmarkTask]] = None) -> List[BenchmarkResult]:
        if tasks is None:
            tasks = self._tasks
        results = []
        for task in tasks:
            predicted = agent.solve(task)
            correct = self.check_answer(task, predicted)
            score = 1.0 if correct else 0.0
            results.append(BenchmarkResult(
                task_id=task.task_id,
                benchmark=self.name,
                correct=correct,
                predicted_answer=predicted,
                expected_answer=task.expected_answer,
                score=score,
            ))
        return results

    def check_answer(self, task: BenchmarkTask, predicted: Any) -> bool:
        """Default answer checking - exact match."""
        return predicted == task.expected_answer


class BenchmarkRegistry:
    """Central registry for all benchmarks."""

    _benchmarks: Dict[str, Callable[[], BaseBenchmark]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[], BaseBenchmark]) -> None:
        cls._benchmarks[name] = factory

    @classmethod
    def load(cls, name: str) -> BaseBenchmark:
        if name not in cls._benchmarks:
            raise KeyError(f"Unknown benchmark: {name}. Available: {list(cls._benchmarks.keys())}")
        return cls._benchmarks[name]()

    @classmethod
    def load_all(cls) -> Dict[str, BaseBenchmark]:
        return {name: factory() for name, factory in cls._benchmarks.items()}

    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls._benchmarks.keys())

    @classmethod
    def evaluate(cls, name: str, agent: Any, tasks: Optional[List[BenchmarkTask]] = None) -> List[BenchmarkResult]:
        benchmark = cls.load(name)
        return benchmark.evaluate(agent, tasks)

    @classmethod
    def clear(cls) -> None:
        cls._benchmarks.clear()


def register_all_benchmarks() -> None:
    """Register all built-in benchmarks."""
    from src.benchmarks.math500 import MATH500Benchmark
    from src.benchmarks.arc_agi import ARCAGIBenchmark
    from src.benchmarks.oolong import OOLONGBenchmark
    from src.benchmarks.humaneval import HumanEvalBenchmark
    from src.benchmarks.swebench import SWEBenchBenchmark
    from src.benchmarks.financial import FinancialBenchmark

    BenchmarkRegistry.register("math500", MATH500Benchmark)
    BenchmarkRegistry.register("arc_agi", ARCAGIBenchmark)
    BenchmarkRegistry.register("oolong", OOLONGBenchmark)
    BenchmarkRegistry.register("humaneval", HumanEvalBenchmark)
    BenchmarkRegistry.register("swebench", SWEBenchBenchmark)
    BenchmarkRegistry.register("financial", FinancialBenchmark)
