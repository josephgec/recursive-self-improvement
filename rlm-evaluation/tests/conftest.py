"""Test fixtures and mock objects for evaluation tests."""

import sys
import os
import random

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.task import EvalTask, EvalResult
from src.execution.rlm_executor import RLMExecutor
from src.execution.standard_executor import StandardExecutor


class MockRLM:
    """Mock RLM that returns correct answers with realistic trajectories."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        return "mock_rlm_response"


class MockStandardLLM:
    """Mock standard LLM for testing."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        return "mock_standard_response"


@pytest.fixture
def mock_rlm():
    """Create a mock RLM."""
    return MockRLM()


@pytest.fixture
def mock_standard_llm():
    """Create a mock standard LLM."""
    return MockStandardLLM()


@pytest.fixture
def rlm_executor():
    """Create an RLM executor with mock LLM."""
    mock = MockRLM()
    return RLMExecutor(llm_fn=mock, seed=42)


@pytest.fixture
def standard_executor():
    """Create a standard executor with mock LLM."""
    mock = MockStandardLLM()
    return StandardExecutor(llm_fn=mock, context_window=8192, seed=42)


@pytest.fixture
def small_context_task():
    """A task with small context (fits in standard window)."""
    return EvalTask(
        task_id="test_small",
        benchmark="test",
        query="What is the secret code?",
        context="The secret code is ALPHA-7749. " * 100,
        expected_answer="ALPHA-7749",
        category="retrieval",
        context_tokens=500,
        difficulty="easy",
    )


@pytest.fixture
def medium_context_task():
    """A task with medium context."""
    return EvalTask(
        task_id="test_medium",
        benchmark="test",
        query="How many items are there?",
        context="Item found. " * 2000,
        expected_answer="2000",
        category="counting",
        context_tokens=4000,
        difficulty="medium",
    )


@pytest.fixture
def large_context_task():
    """A task with large context (exceeds standard window)."""
    return EvalTask(
        task_id="test_large",
        benchmark="test",
        query="What is the hidden value?",
        context=("Filler text for context padding. " * 5000 +
                 "The hidden value is 42." +
                 "More filler text. " * 5000),
        expected_answer="42",
        category="needle_in_haystack",
        context_tokens=20000,
        difficulty="hard",
    )


@pytest.fixture
def sample_tasks():
    """A set of sample tasks at various context sizes."""
    return [
        EvalTask(
            task_id=f"sample_{i}",
            benchmark="test",
            query=f"Question {i}?",
            context=f"Answer is {i}. " * (100 * (i + 1)),
            expected_answer=str(i),
            category=["retrieval", "counting", "reasoning", "aggregation"][i % 4],
            context_tokens=500 * (i + 1),
            difficulty=["easy", "medium", "hard"][i % 3],
        )
        for i in range(8)
    ]


@pytest.fixture
def sample_rlm_results():
    """Sample RLM evaluation results."""
    strategies = ["PEEK_THEN_GREP", "MAP_REDUCE", "ITERATIVE_SEARCH",
                  "HIERARCHICAL", "DIRECT", "HYBRID"]
    results = []
    rng = random.Random(42)

    for i in range(10):
        strategy = strategies[i % len(strategies)]
        correct = rng.random() < 0.8

        trajectory = []
        if strategy == "PEEK_THEN_GREP":
            trajectory = [
                "# Peek at structure\nhead -100 context.txt",
                "# Search for info\ngrep -n 'target' context.txt",
                "# Read section\nsed -n '50,60p' context.txt",
                f"# Output\necho 'answer_{i}'",
            ]
        elif strategy == "MAP_REDUCE":
            trajectory = [
                "# Chunk context\nsplit -l 100 context.txt chunk_",
                "# Process chunks\nfor f in chunk_*; do process $f; done",
                "# Loop through results\nfor r in results: aggregate(r)",
                "# Aggregate\nresult = aggregate(partial_results)",
                f"# Output\necho 'answer_{i}'",
            ]
        elif strategy == "ITERATIVE_SEARCH":
            trajectory = [
                "# Search\ngrep -c 'target' context.txt",
                "# Loop and count\nfor line in context: if target in line: count += 1",
                f"# Output\necho 'answer_{i}'",
            ]
        elif strategy == "HIERARCHICAL":
            trajectory = [
                "# Peek\nhead -50 context.txt",
                "# Sub-query 1\npython analyze_part1.py",
                "# Sub-query 2\npython analyze_part2.py",
                "# Synthesize\nresult = aggregate(sub_results)",
                f"# Output\necho 'answer_{i}'",
            ]
        elif strategy == "DIRECT":
            trajectory = [
                "# Direct read\ncat context.txt",
                f"# Output\necho 'answer_{i}'",
            ]
        else:  # HYBRID
            trajectory = [
                "# Peek\nhead -100 context.txt",
                "# Search\ngrep 'key' context.txt",
                "# Chunk\nsplit -l 50 context.txt chunk_",
                "# Sub-query\npython analyze.py",
                "# Loop\nfor item in items: process(item)",
                f"# Output\necho 'answer_{i}'",
            ]

        results.append(EvalResult(
            task_id=f"sample_{i}",
            benchmark="test",
            answer=f"answer_{i}" if correct else "wrong",
            correct=correct,
            trajectory=trajectory,
            strategy_detected=strategy,
            cost=rng.uniform(0.001, 0.05),
            input_tokens=rng.randint(500, 20000),
            output_tokens=rng.randint(50, 500),
            num_calls=len(trajectory),
            latency_ms=rng.uniform(200, 3000),
        ))

    return results


@pytest.fixture
def sample_std_results(sample_rlm_results):
    """Sample standard LLM results matching the RLM results."""
    rng = random.Random(99)
    results = []
    for rlm_r in sample_rlm_results:
        correct = rng.random() < 0.5  # Lower accuracy for standard
        results.append(EvalResult(
            task_id=rlm_r.task_id,
            benchmark="test",
            answer=rlm_r.answer if correct else "wrong_std",
            correct=correct,
            trajectory=["# Single-shot query"],
            strategy_detected="DIRECT",
            cost=rng.uniform(0.0005, 0.01),
            input_tokens=rng.randint(500, 8192),
            output_tokens=rng.randint(50, 200),
            num_calls=1,
            latency_ms=rng.uniform(100, 1000),
        ))
    return results


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoints."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    return str(d)
