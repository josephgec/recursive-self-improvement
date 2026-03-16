"""Test fixtures: MockAgent, MockPipeline, sample benchmarks, collapse baselines."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import pytest

from src.benchmarks.registry import (
    BaseBenchmark,
    BenchmarkRegistry,
    BenchmarkTask,
    register_all_benchmarks,
)
from src.collapse.baseline_loader import CollapseBaselineLoader, CollapseTrajectory


class MockAgent:
    """Mock agent whose accuracy varies by iteration and condition.

    Behavior per condition:
    - full_pipeline: accuracy = 0.60 + 0.015*iteration (improving)
    - no_soar: accuracy = 0.60 + 0.008*iteration (slower)
    - no_ctm: accuracy = 0.60 + 0.010*iteration
    - no_godel: accuracy = 0.60 + 0.005*iteration (minimal)
    - no_rlm: accuracy = 0.60 + 0.012*iteration
    - soar_only: accuracy = 0.60 + 0.006*iteration
    - naive_self_train: accuracy = 0.65 - 0.01*iteration (DECLINING)
    """

    RATES: Dict[str, float] = {
        "full_pipeline": 0.015,
        "no_soar": 0.008,
        "no_ctm": 0.010,
        "no_godel": 0.005,
        "no_rlm": 0.012,
        "soar_only": 0.006,
        "naive_self_train": -0.01,
    }

    BASES: Dict[str, float] = {
        "full_pipeline": 0.60,
        "no_soar": 0.60,
        "no_ctm": 0.60,
        "no_godel": 0.60,
        "no_rlm": 0.60,
        "soar_only": 0.60,
        "naive_self_train": 0.65,
    }

    def __init__(self, condition: str = "full_pipeline") -> None:
        self._condition = condition
        self._iteration = 0

    @property
    def condition(self) -> str:
        return self._condition

    @property
    def iteration(self) -> int:
        return self._iteration

    def set_iteration(self, iteration: int) -> None:
        self._iteration = iteration

    def solve(self, task: BenchmarkTask) -> Any:
        """Solve a task. Returns expected answer with probability based on accuracy.

        Uses a stable hash of just the task_id to assign each task a fixed
        difficulty. Whether the agent solves it depends on whether that
        difficulty falls below the current accuracy threshold. This guarantees
        that higher accuracy => strictly more tasks solved, and the ordering
        between conditions is preserved.
        """
        rate = self.RATES.get(self._condition, 0.015)
        base = self.BASES.get(self._condition, 0.60)
        accuracy = base + rate * self._iteration
        accuracy = max(0.0, min(1.0, accuracy))

        # Fixed difficulty per task (independent of iteration/condition)
        h = int(hashlib.md5(task.task_id.encode()).hexdigest(), 16) % 10000
        difficulty = h / 10000.0  # 0.0 to 1.0

        if difficulty < accuracy:
            return task.expected_answer
        return None


class MockPipeline:
    """Mock RSI pipeline for testing."""

    def __init__(self, condition: str = "full_pipeline") -> None:
        self._agent = MockAgent(condition)
        self._condition = condition

    @property
    def agent(self) -> MockAgent:
        return self._agent

    def run_iteration(self, iteration: int) -> MockAgent:
        self._agent.set_iteration(iteration)
        return self._agent


@pytest.fixture
def mock_agent():
    """Create a mock agent with full_pipeline condition."""
    return MockAgent("full_pipeline")


@pytest.fixture
def mock_agent_factory():
    """Factory for creating mock agents with different conditions."""
    def factory(condition: str = "full_pipeline") -> MockAgent:
        return MockAgent(condition)
    return factory


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline."""
    return MockPipeline("full_pipeline")


@pytest.fixture
def registered_benchmarks():
    """Register and return all benchmarks."""
    BenchmarkRegistry.clear()
    register_all_benchmarks()
    return BenchmarkRegistry.load_all()


@pytest.fixture
def sample_benchmarks():
    """Load a subset of benchmarks for faster testing."""
    BenchmarkRegistry.clear()
    register_all_benchmarks()
    return {
        name: BenchmarkRegistry.load(name)
        for name in ["math500", "humaneval"]
    }


@pytest.fixture
def collapse_baselines():
    """Create collapse baseline trajectories."""
    loader = CollapseBaselineLoader(
        num_generations=20,
        initial_accuracy=0.78,
        decay_rate=0.02,
        entropy_decay=0.05,
    )
    return loader


@pytest.fixture
def improving_curve():
    """Create an improving accuracy curve."""
    return [(i, 0.60 + 0.015 * i) for i in range(15)]


@pytest.fixture
def declining_curve():
    """Create a declining accuracy curve (collapse)."""
    return [(i, 0.65 - 0.01 * i) for i in range(15)]


@pytest.fixture
def plateaued_curve():
    """Create a plateaued accuracy curve."""
    return [(i, 0.75 + 0.001 * ((-1) ** i)) for i in range(15)]
