"""Shared test fixtures."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.core.executor import Task, TaskResult, MockLLMClient
from src.core.state import AgentState, StateManager
from src.meta.prompt_strategy import DefaultPromptStrategy
from src.meta.registry import ComponentRegistry
from src.modification.modifier import CodeModifier, ModificationProposal


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """A mock LLM client."""
    return MockLLMClient(default_response="Let me think step by step.\n\nThe answer is: 42")


@pytest.fixture
def mock_llm_with_proposal() -> MockLLMClient:
    """A mock LLM client that returns modification proposals."""
    proposal = json.dumps({
        "action": "modify",
        "target": "prompt_strategy",
        "method_name": "prepare_prompt",
        "description": "Improve prompt clarity",
        "code": "def prepare_prompt(self, task, examples):\n    return f'Solve carefully: {task.question}'\n",
        "risk": "low",
        "rationale": "Simpler prompt may improve accuracy",
    })
    return MockLLMClient(default_response=proposal)


@pytest.fixture
def mock_llm_defer() -> MockLLMClient:
    """A mock LLM that defers modification."""
    return MockLLMClient(
        default_response=json.dumps({"action": "defer", "rationale": "Performance is acceptable"})
    )


@pytest.fixture
def sample_tasks() -> list[Task]:
    """A small set of sample tasks."""
    return [
        Task(task_id="t1", question="What is 2 + 2?", expected_answer="4", domain="math", category="arithmetic"),
        Task(task_id="t2", question="What is 3 * 5?", expected_answer="15", domain="math", category="arithmetic"),
        Task(task_id="t3", question="What is 10 / 2?", expected_answer="5", domain="math", category="arithmetic"),
        Task(task_id="t4", question="Solve: x + 3 = 7", expected_answer="4", domain="math", category="algebra"),
        Task(task_id="t5", question="What is 6!?", expected_answer="720", domain="math", category="number_theory"),
    ]


@pytest.fixture
def sample_task() -> Task:
    """A single sample task."""
    return Task(task_id="t1", question="What is 2 + 2?", expected_answer="4", domain="math", category="arithmetic")


@pytest.fixture
def sample_results(sample_tasks: list[Task]) -> list[TaskResult]:
    """Sample task results."""
    return [
        TaskResult(task=sample_tasks[0], response="The answer is: 4", extracted_answer="4", correct=True),
        TaskResult(task=sample_tasks[1], response="The answer is: 15", extracted_answer="15", correct=True),
        TaskResult(task=sample_tasks[2], response="The answer is: 5", extracted_answer="5", correct=True),
        TaskResult(task=sample_tasks[3], response="The answer is: 3", extracted_answer="3", correct=False),
        TaskResult(task=sample_tasks[4], response="The answer is: 120", extracted_answer="120", correct=False),
    ]


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def state_manager(tmp_dir: Path) -> StateManager:
    """A StateManager using a temp directory."""
    return StateManager(checkpoint_dir=str(tmp_dir / "checkpoints"))


@pytest.fixture
def sample_state() -> AgentState:
    """A sample agent state."""
    return AgentState(
        iteration=5,
        system_prompt="You are a helpful problem solver.",
        accuracy_history=[0.6, 0.65, 0.7, 0.68, 0.72],
    )


@pytest.fixture
def registry() -> ComponentRegistry:
    """A component registry with a default strategy."""
    reg = ComponentRegistry()
    reg.register("prompt_strategy", DefaultPromptStrategy())
    return reg


@pytest.fixture
def modifier() -> CodeModifier:
    """A code modifier with default settings."""
    return CodeModifier(
        allowed_targets=["prompt_strategy", "few_shot_selector", "reasoning_strategy"],
        forbidden_targets=["validation.suite", "audit.logger"],
    )


@pytest.fixture
def valid_proposal() -> ModificationProposal:
    """A valid modification proposal."""
    return ModificationProposal(
        target="prompt_strategy",
        method_name="prepare_prompt",
        description="Improve prompt",
        code="def prepare_prompt(self, task, examples):\n    return f'Solve: {task.question}'\n",
        risk="low",
        rationale="Simplify prompt",
    )


@pytest.fixture
def debug_config() -> dict[str, Any]:
    """Debug configuration dict."""
    return {
        "project": {"name": "test", "seed": 42, "output_dir": "data"},
        "agent": {
            "llm_provider": "mock",
            "llm_model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 4096,
            "system_prompt": "You are a helpful problem solver.",
        },
        "meta_learning": {
            "max_iterations": 5,
            "warmup_iterations": 2,
            "tasks_per_iteration": 5,
            "modification_cooldown": 1,
            "num_examples": 3,
        },
        "modification": {
            "max_concurrent_changes": 1,
            "require_deliberation": True,
            "deliberation_depth": 2,
            "allowed_targets": ["prompt_strategy", "few_shot_selector", "reasoning_strategy"],
            "forbidden_targets": ["validation.suite", "validation.runner", "audit.logger", "rollback.mechanism"],
            "max_complexity_ratio": 5.0,
        },
        "validation": {
            "suite": "core",
            "min_pass_rate": 0.60,
            "performance_threshold": -0.05,
            "auto_rollback": True,
        },
        "audit": {
            "log_dir": "data/audit_logs",
            "log_diffs": True,
            "log_reasoning": True,
            "log_runtime_state": True,
        },
    }
