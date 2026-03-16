"""Shared test fixtures for soar-hindsight tests."""

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collection.trajectory import (
    IndividualRecord,
    ImprovementStep,
    SearchTrajectory,
    TaskSpec,
)
from src.synthesis.synthesizer import TrainingPair

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def task_spec():
    """A simple task specification."""
    return TaskSpec(
        task_id="test-task-1",
        description="Implement a function that adds two numbers",
        test_cases=[{"input": [1, 2], "expected": 3}],
        difficulty="easy",
        tags=["math", "basic"],
    )


@pytest.fixture
def individual_records():
    """A list of individual records spanning generations."""
    return [
        IndividualRecord(
            individual_id="ind-a",
            generation=0,
            code="def add(a, b):\n    return 0",
            fitness=0.1,
            operator="init",
            error="AssertionError: expected 3 got 0",
        ),
        IndividualRecord(
            individual_id="ind-b",
            generation=0,
            code="def add(a, b):\n    return a",
            fitness=0.3,
            operator="init",
        ),
        IndividualRecord(
            individual_id="ind-c",
            generation=1,
            code="def add(a, b):\n    return a + b",
            fitness=1.0,
            parent_ids=["ind-a"],
            operator="mutation",
        ),
        IndividualRecord(
            individual_id="ind-d",
            generation=1,
            code="def add(a, b):\n    result = a + b\n    return result",
            fitness=1.0,
            parent_ids=["ind-a", "ind-b"],
            operator="crossover",
        ),
    ]


@pytest.fixture
def solved_trajectory(task_spec, individual_records):
    """A trajectory that solved its task."""
    traj = SearchTrajectory(
        trajectory_id="test-solved",
        task=task_spec,
        best_fitness=1.0,
        total_generations=2,
        solved=True,
    )
    for ind in individual_records:
        traj.add_individual(ind)
    return traj


@pytest.fixture
def partial_trajectory():
    """A trajectory with partial success."""
    task = TaskSpec(
        task_id="test-task-2",
        description="Implement a function that sorts a list using merge sort",
        difficulty="medium",
        tags=["sorting"],
    )
    traj = SearchTrajectory(
        trajectory_id="test-partial",
        task=task,
        best_fitness=0.6,
        total_generations=3,
        solved=False,
    )
    traj.add_individual(IndividualRecord(
        individual_id="p-a",
        generation=0,
        code="def merge_sort(lst):\n    return lst",
        fitness=0.1,
        operator="init",
    ))
    traj.add_individual(IndividualRecord(
        individual_id="p-b",
        generation=1,
        code="def merge_sort(lst):\n    if len(lst) <= 1:\n        return lst\n    return sorted(lst[:len(lst)//2]) + lst[len(lst)//2:]",
        fitness=0.4,
        parent_ids=["p-a"],
        operator="mutation",
    ))
    traj.add_individual(IndividualRecord(
        individual_id="p-c",
        generation=2,
        code="def merge_sort(lst):\n    if len(lst) <= 1:\n        return lst\n    mid = len(lst) // 2\n    left = merge_sort(lst[:mid])\n    right = merge_sort(lst[mid:])\n    return sorted(left + right)",
        fitness=0.6,
        parent_ids=["p-b"],
        operator="mutation",
    ))
    return traj


@pytest.fixture
def failed_trajectory():
    """A trajectory with errors."""
    task = TaskSpec(
        task_id="test-task-3",
        description="Implement BFS on a graph",
        difficulty="hard",
        tags=["graph", "search"],
    )
    traj = SearchTrajectory(
        trajectory_id="test-failed",
        task=task,
        best_fitness=0.3,
        total_generations=2,
        solved=False,
    )
    traj.add_individual(IndividualRecord(
        individual_id="f-a",
        generation=0,
        code="def bfs(graph, start):\n    return []",
        fitness=0.0,
        operator="init",
        error="TypeError: wrong return type",
    ))
    traj.add_individual(IndividualRecord(
        individual_id="f-b",
        generation=1,
        code="def bfs(graph, start):\n    visited = [start]\n    return visited",
        fitness=0.3,
        parent_ids=["f-a"],
        operator="mutation",
    ))
    return traj


@pytest.fixture
def all_trajectories(solved_trajectory, partial_trajectory, failed_trajectory):
    """All test trajectories combined."""
    return [solved_trajectory, partial_trajectory, failed_trajectory]


@pytest.fixture
def sample_training_pairs():
    """Sample training pairs for testing."""
    return [
        TrainingPair(
            pair_id="pair-1",
            strategy="direct_solution",
            task_id="task-1",
            prompt="Solve: add two numbers",
            completion="def add(a, b):\n    return a + b",
            quality_score=0.9,
            prompt_tokens=20,
            completion_tokens=15,
        ),
        TrainingPair(
            pair_id="pair-2",
            strategy="error_correction",
            task_id="task-2",
            prompt="Fix: def sort(lst): return lst\nError: wrong order",
            completion="def sort(lst):\n    return sorted(lst)",
            quality_score=0.7,
            prompt_tokens=30,
            completion_tokens=15,
        ),
        TrainingPair(
            pair_id="pair-3",
            strategy="improvement_chain",
            task_id="task-3",
            prompt="Improve: def fib(n): return n",
            completion="def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)",
            quality_score=0.8,
            prompt_tokens=25,
            completion_tokens=30,
        ),
        TrainingPair(
            pair_id="pair-4",
            strategy="direct_solution",
            task_id="task-1",
            prompt="Solve: multiply two numbers",
            completion="def multiply(a, b):\n    return a * b",
            quality_score=0.85,
            prompt_tokens=22,
            completion_tokens=16,
        ),
    ]


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def solved_trajectory_file(fixtures_dir):
    """Path to solved trajectory fixture."""
    return os.path.join(fixtures_dir, "solved_trajectory.json")


@pytest.fixture
def partial_trajectory_file(fixtures_dir):
    """Path to partial trajectory fixture."""
    return os.path.join(fixtures_dir, "partial_trajectory.json")


@pytest.fixture
def failed_trajectory_file(fixtures_dir):
    """Path to failed trajectory fixture."""
    return os.path.join(fixtures_dir, "failed_trajectory.json")


@pytest.fixture
def trajectory_with_crossover():
    """Trajectory with crossover events for testing crossover strategy."""
    task = TaskSpec(
        task_id="crossover-task",
        description="Combine solutions for maximum efficiency",
        difficulty="medium",
        tags=["optimization"],
    )
    parent_a = IndividualRecord(
        individual_id="cx-parent-a",
        generation=0,
        code="def solve(x):\n    return x + 1",
        fitness=0.5,
        operator="init",
    )
    parent_b = IndividualRecord(
        individual_id="cx-parent-b",
        generation=0,
        code="def solve(x):\n    return x * 2",
        fitness=0.6,
        operator="init",
    )
    child = IndividualRecord(
        individual_id="cx-child",
        generation=1,
        code="def solve(x):\n    return (x + 1) * 2",
        fitness=0.9,
        parent_ids=["cx-parent-a", "cx-parent-b"],
        operator="crossover",
    )
    traj = SearchTrajectory(
        trajectory_id="test-crossover",
        task=task,
        best_fitness=0.9,
        total_generations=2,
        solved=False,
    )
    for ind in [parent_a, parent_b, child]:
        traj.add_individual(ind)
    return traj
