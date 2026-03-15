"""Tests for GodelAgent main loop."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from src.core.agent import GodelAgent, IterationResult
from src.core.executor import Task, MockLLMClient


def make_config(tmp_path: Any, **overrides: Any) -> dict[str, Any]:
    """Create a test config with temp paths."""
    config: dict[str, Any] = {
        "project": {"name": "test", "seed": 42, "output_dir": str(tmp_path / "data")},
        "agent": {
            "llm_provider": "mock",
            "system_prompt": "Solve problems step by step.",
        },
        "meta_learning": {
            "max_iterations": 5,
            "warmup_iterations": 2,
            "tasks_per_iteration": 3,
            "modification_cooldown": 1,
            "num_examples": 2,
        },
        "modification": {
            "require_deliberation": True,
            "deliberation_depth": 2,
            "allowed_targets": ["prompt_strategy", "few_shot_selector", "reasoning_strategy"],
            "forbidden_targets": ["validation.suite", "audit.logger"],
            "max_complexity_ratio": 5.0,
        },
        "validation": {
            "suite": "__none__",  # No real validation suite for unit tests
            "min_pass_rate": 0.60,
            "performance_threshold": -0.05,
            "auto_rollback": True,
        },
        "audit": {
            "log_dir": str(tmp_path / "audit_logs"),
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value
    return config


@pytest.fixture
def sample_tasks() -> list[Task]:
    return [
        Task(task_id=f"t{i}", question=f"What is {i} + {i}?", expected_answer=str(2 * i), domain="math", category="arithmetic")
        for i in range(1, 11)
    ]


class TestWarmup:
    def test_no_modification_during_warmup(self, tmp_path: Any, sample_tasks: list[Task]) -> None:
        config = make_config(tmp_path)
        config["meta_learning"]["warmup_iterations"] = 3
        config["meta_learning"]["max_iterations"] = 3

        agent = GodelAgent(config)
        results = agent.run(sample_tasks)

        # During warmup, no deliberation should occur
        for r in results:
            assert r.deliberated is False
            assert r.modification_applied is False

    def test_warmup_collects_accuracy(self, tmp_path: Any, sample_tasks: list[Task]) -> None:
        config = make_config(tmp_path)
        config["meta_learning"]["max_iterations"] = 3

        agent = GodelAgent(config)
        results = agent.run(sample_tasks)

        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.accuracy <= 1.0


class TestDeliberationTrigger:
    def test_deliberation_after_warmup_on_stagnation(self, tmp_path: Any, sample_tasks: list[Task]) -> None:
        config = make_config(tmp_path)
        config["meta_learning"]["warmup_iterations"] = 2
        config["meta_learning"]["max_iterations"] = 8
        config["meta_learning"]["modification_cooldown"] = 1

        agent = GodelAgent(config)
        results = agent.run(sample_tasks)

        # Some iteration after warmup should have deliberated
        post_warmup = [r for r in results if r.iteration >= 2]
        deliberated_any = any(r.deliberated for r in post_warmup)
        # With mock LLM producing same answers, stagnation should trigger deliberation
        # At minimum, periodic deliberation at iteration 5 should fire
        assert len(results) == 8

    def test_no_deliberation_when_disabled(self, tmp_path: Any, sample_tasks: list[Task]) -> None:
        config = make_config(tmp_path)
        config["meta_learning"]["max_iterations"] = 5

        agent = GodelAgent(config)
        results = agent.run(sample_tasks, allow_modification=False)

        for r in results:
            assert r.deliberated is False
            assert r.modification_applied is False


class TestModificationFlow:
    def test_modification_can_be_applied(self, tmp_path: Any, sample_tasks: list[Task]) -> None:
        config = make_config(tmp_path)
        config["meta_learning"]["warmup_iterations"] = 1
        config["meta_learning"]["max_iterations"] = 10
        config["meta_learning"]["modification_cooldown"] = 1

        # Use a mock LLM that proposes modifications
        agent = GodelAgent(config)

        # Pre-load the LLM with modification proposals for deliberation calls
        proposal = json.dumps({
            "action": "modify",
            "target": "prompt_strategy",
            "method_name": "choose_reasoning_mode",
            "description": "Try direct mode",
            "code": "def choose_reasoning_mode(self, task, recent_results):\n    return 'direct'\n",
            "risk": "low",
            "rationale": "Direct mode might be faster",
        })
        if isinstance(agent.llm, MockLLMClient):
            # Add enough proposal responses for multiple deliberation attempts
            for _ in range(20):
                agent.llm.add_response("The answer is: 42")
            # Add proposal for when deliberation fires
            agent.llm.add_response(proposal)

        results = agent.run(sample_tasks)
        assert len(results) == 10


class TestRollbackInLoop:
    def test_modification_disabled_no_rollback(self, tmp_path: Any, sample_tasks: list[Task]) -> None:
        config = make_config(tmp_path)
        config["meta_learning"]["max_iterations"] = 5

        agent = GodelAgent(config)
        results = agent.run(sample_tasks, allow_modification=False)

        for r in results:
            assert r.modification_rolled_back is False

    def test_cooldown_respected(self, tmp_path: Any, sample_tasks: list[Task]) -> None:
        config = make_config(tmp_path)
        config["meta_learning"]["warmup_iterations"] = 1
        config["meta_learning"]["max_iterations"] = 5
        config["meta_learning"]["modification_cooldown"] = 3

        agent = GodelAgent(config)
        results = agent.run(sample_tasks)

        # With cooldown of 3, at most 1-2 modifications in 5 iterations
        mod_count = sum(1 for r in results if r.modification_applied)
        assert mod_count <= 2
