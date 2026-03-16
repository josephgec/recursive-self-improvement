"""Shared fixtures for tests: mock LLMs, pipelines, sample tasks, temp dirs."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from src.hybrid.pipeline import HybridPipeline, HybridResult, ReasoningStep
from src.integrative.pipeline import IntegrativePipeline, IntegrativeResult
from src.utils.task_domains import MultiDomainTaskLoader, Task


# ── Mock LLM ──

class MockLLM:
    """Deterministic mock LLM for testing."""

    def __init__(self, responses: Optional[Dict[str, str]] = None) -> None:
        self._responses = responses or {}
        self.call_count = 0
        self.last_prompt = ""

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt

        # Check for exact matches first
        for key, value in self._responses.items():
            if key in prompt:
                return value

        # Default arithmetic handling
        arith = re.search(r"(\d+)\s*([\+\-\*])\s*(\d+)", prompt)
        if arith:
            a, op, b = int(arith.group(1)), arith.group(2), int(arith.group(3))
            ops = {"+": a + b, "-": a - b, "*": a * b}
            result = ops.get(op, 0)
            if "tool result" in prompt.lower():
                return f"Based on the result, the answer is {result}.\nANSWER: {result}"
            return f"TOOL_CALL: sympy_solve({a} {op} {b})\nANSWER: {result}"

        return "ANSWER: unknown"


# ── Mock Prose Pipeline ──

@dataclass
class ProseResult:
    """Result from prose baseline."""
    answer: str = ""
    correct: bool = False
    metadata: dict = field(default_factory=dict)


class ProsePipeline:
    """Simple prose baseline that does pattern matching."""

    def solve(self, problem: str) -> ProseResult:
        arith = re.search(r"(\d+)\s*([\+\-\*])\s*(\d+)", problem)
        if arith:
            a, op, b = int(arith.group(1)), arith.group(2), int(arith.group(3))
            ops = {"+": a + b, "-": a - b, "*": a * b}
            result = ops.get(op, 0)
            return ProseResult(answer=str(result))
        if "true" in problem.lower():
            return ProseResult(answer="true")
        if "false" in problem.lower():
            return ProseResult(answer="false")
        return ProseResult(answer="unknown")


# ── Fixtures ──

@pytest.fixture
def mock_llm():
    """Provide a MockLLM instance."""
    return MockLLM()


@pytest.fixture
def mock_llm_with_tools():
    """Provide a MockLLM that generates tool calls."""
    return MockLLM({
        "What is 3 + 4": "I need to compute 3 + 4.\nTOOL_CALL: sympy_solve(3 + 4)\nANSWER: computing",
        "tool result": "The answer is 7.\nANSWER: 7",
    })


@pytest.fixture
def hybrid_pipeline(mock_llm):
    """Provide a HybridPipeline with mock LLM."""
    return HybridPipeline(llm=mock_llm, max_tool_calls=3)


@pytest.fixture
def integrative_pipeline():
    """Provide an IntegrativePipeline."""
    return IntegrativePipeline()


@pytest.fixture
def prose_pipeline():
    """Provide a ProsePipeline."""
    return ProsePipeline()


@pytest.fixture
def task_loader():
    """Provide a MultiDomainTaskLoader."""
    return MultiDomainTaskLoader()


@pytest.fixture
def sample_arithmetic_tasks():
    """Provide sample arithmetic tasks."""
    return [
        Task(task_id="t1", domain="arithmetic", problem="What is 3 + 4?", expected_answer="7"),
        Task(task_id="t2", domain="arithmetic", problem="Compute 5 * 6.", expected_answer="30"),
        Task(task_id="t3", domain="arithmetic", problem="What is 10 - 3?", expected_answer="7"),
        Task(task_id="t4", domain="arithmetic", problem="Calculate 8 + 2.", expected_answer="10"),
    ]


@pytest.fixture
def sample_algebra_tasks():
    """Provide sample algebra tasks."""
    return [
        Task(task_id="a1", domain="algebra", problem="Solve for x: x + 3 = 7", expected_answer="4"),
        Task(task_id="a2", domain="algebra", problem="Solve for x: 2 * x = 10", expected_answer="5"),
    ]


@pytest.fixture
def sample_logic_tasks():
    """Provide sample logic tasks."""
    return [
        Task(task_id="l1", domain="logic", problem="If P is true, what is P?", expected_answer="true"),
        Task(task_id="l2", domain="logic", problem="If P is false, what is not P?", expected_answer="true"),
    ]


@pytest.fixture
def paired_tasks():
    """Provide paired original and perturbed tasks."""
    originals = [
        Task(task_id="p1", domain="arithmetic", problem="What is 3 + 4?", expected_answer="7"),
        Task(task_id="p2", domain="arithmetic", problem="Compute 5 * 6.", expected_answer="30"),
    ]
    perturbed = [
        Task(task_id="p1_pert", domain="arithmetic", problem="Calculate 3 + 4.", expected_answer="7"),
        Task(task_id="p2_pert", domain="arithmetic", problem="Find the value of 5 * 6.", expected_answer="30"),
    ]
    return originals, perturbed


@pytest.fixture
def temp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
