"""Tests for validation: suite, runner, checkpointing re-export."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.core.executor import Task, TaskResult, TaskExecutor, MockLLMClient
from src.validation.suite import ValidationSuite
from src.validation.runner import ValidationRunner, ValidationResult


# ===================================================================
# ValidationSuite
# ===================================================================

class TestValidationSuite:
    def test_load_nonexistent_suite(self) -> None:
        suite = ValidationSuite(suite_name="nonexistent", config_dir="/tmp/no_such_dir")
        assert suite.get_tasks() == []
        assert suite.size == 0
        assert suite.suite_name == "nonexistent"

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "suites"
        config_dir.mkdir()
        suite_file = config_dir / "test_suite.yaml"
        data = {
            "tasks": [
                {"id": "v1", "question": "What is 1+1?", "answer": 2, "domain": "math", "category": "arithmetic", "difficulty": "easy"},
                {"id": "v2", "question": "What is 3*3?", "answer": 9, "domain": "math", "category": "arithmetic"},
            ]
        }
        with open(suite_file, "w") as f:
            yaml.dump(data, f)

        suite = ValidationSuite(suite_name="test_suite", config_dir=str(config_dir))
        tasks = suite.get_tasks()
        assert len(tasks) == 2
        assert tasks[0].task_id == "v1"
        assert tasks[0].expected_answer == "2"
        assert tasks[0].domain == "math"
        assert tasks[0].difficulty == "easy"
        assert tasks[1].task_id == "v2"
        assert tasks[1].difficulty == "medium"  # default

    def test_suite_name_property(self) -> None:
        suite = ValidationSuite(suite_name="my_suite", config_dir="/dev/null/no")
        assert suite.suite_name == "my_suite"

    def test_get_tasks_returns_new_list(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "suites"
        config_dir.mkdir()
        suite_file = config_dir / "test2.yaml"
        data = {"tasks": [{"id": "v1", "question": "q", "answer": "a"}]}
        with open(suite_file, "w") as f:
            yaml.dump(data, f)

        suite = ValidationSuite(suite_name="test2", config_dir=str(config_dir))
        t1 = suite.get_tasks()
        t2 = suite.get_tasks()
        assert t1 is not t2
        assert len(t1) == len(t2) == 1

    def test_size_property(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "suites"
        config_dir.mkdir()
        suite_file = config_dir / "sized.yaml"
        data = {"tasks": [{"id": f"v{i}", "question": f"q{i}", "answer": str(i)} for i in range(5)]}
        with open(suite_file, "w") as f:
            yaml.dump(data, f)

        suite = ValidationSuite(suite_name="sized", config_dir=str(config_dir))
        assert suite.size == 5

    def test_empty_yaml(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "suites"
        config_dir.mkdir()
        suite_file = config_dir / "empty.yaml"
        data = {"tasks": []}
        with open(suite_file, "w") as f:
            yaml.dump(data, f)

        suite = ValidationSuite(suite_name="empty", config_dir=str(config_dir))
        assert suite.size == 0
        assert suite.get_tasks() == []


# ===================================================================
# ValidationResult
# ===================================================================

class TestValidationResult:
    def test_default_values(self) -> None:
        vr = ValidationResult()
        assert vr.passed is False
        assert vr.pass_rate == 0.0
        assert vr.total == 0
        assert vr.correct == 0

    def test_to_dict(self) -> None:
        vr = ValidationResult(
            passed=True,
            pass_rate=0.9,
            baseline_rate=0.85,
            delta=0.05,
            total=10,
            correct=9,
        )
        d = vr.to_dict()
        assert d["passed"] is True
        assert d["pass_rate"] == 0.9
        assert d["baseline_rate"] == 0.85
        assert d["delta"] == 0.05
        assert d["total"] == 10
        assert d["correct"] == 9


# ===================================================================
# ValidationRunner
# ===================================================================

class TestValidationRunner:
    def _make_suite(self, tmp_path: Path, n_tasks: int = 5) -> ValidationSuite:
        config_dir = tmp_path / "suites"
        config_dir.mkdir(exist_ok=True)
        suite_file = config_dir / "runner_test.yaml"
        data = {
            "tasks": [
                {"id": f"vt{i}", "question": f"What is {i}+{i}?", "answer": str(2 * i)}
                for i in range(1, n_tasks + 1)
            ]
        }
        with open(suite_file, "w") as f:
            yaml.dump(data, f)
        return ValidationSuite(suite_name="runner_test", config_dir=str(config_dir))

    def test_set_baseline(self, tmp_path: Path) -> None:
        mock_llm = MockLLMClient(default_response="The answer is: 42")
        executor = TaskExecutor(mock_llm)
        suite = self._make_suite(tmp_path)
        runner = ValidationRunner(executor, suite)
        baseline = runner.set_baseline()
        assert isinstance(baseline, float)
        assert 0.0 <= baseline <= 1.0

    def test_set_baseline_empty_suite(self, tmp_path: Path) -> None:
        mock_llm = MockLLMClient()
        executor = TaskExecutor(mock_llm)
        config_dir = tmp_path / "suites"
        config_dir.mkdir(exist_ok=True)
        suite_file = config_dir / "empty.yaml"
        with open(suite_file, "w") as f:
            yaml.dump({"tasks": []}, f)
        suite = ValidationSuite(suite_name="empty", config_dir=str(config_dir))
        runner = ValidationRunner(executor, suite)
        baseline = runner.set_baseline()
        assert baseline == 1.0

    def test_run_full(self, tmp_path: Path) -> None:
        mock_llm = MockLLMClient(default_response="The answer is: 42")
        executor = TaskExecutor(mock_llm)
        suite = self._make_suite(tmp_path)
        runner = ValidationRunner(executor, suite, min_pass_rate=0.0)
        result = runner.run()
        assert isinstance(result, ValidationResult)
        assert result.total == 5
        assert result.pass_rate >= 0.0

    def test_run_empty_suite(self, tmp_path: Path) -> None:
        mock_llm = MockLLMClient()
        executor = TaskExecutor(mock_llm)
        config_dir = tmp_path / "suites"
        config_dir.mkdir(exist_ok=True)
        suite_file = config_dir / "empty2.yaml"
        with open(suite_file, "w") as f:
            yaml.dump({"tasks": []}, f)
        suite = ValidationSuite(suite_name="empty2", config_dir=str(config_dir))
        runner = ValidationRunner(executor, suite)
        result = runner.run()
        assert result.passed is True
        assert result.pass_rate == 1.0

    def test_run_quick(self, tmp_path: Path) -> None:
        mock_llm = MockLLMClient(default_response="The answer is: 42")
        executor = TaskExecutor(mock_llm)
        suite = self._make_suite(tmp_path, n_tasks=10)
        runner = ValidationRunner(executor, suite, min_pass_rate=0.0)
        result = runner.run_quick(sample_fraction=0.3)
        assert isinstance(result, ValidationResult)
        # 30% of 10 = 3
        assert result.total <= 10
        assert result.total >= 1

    def test_run_quick_empty_suite(self, tmp_path: Path) -> None:
        mock_llm = MockLLMClient()
        executor = TaskExecutor(mock_llm)
        config_dir = tmp_path / "suites"
        config_dir.mkdir(exist_ok=True)
        suite_file = config_dir / "empty3.yaml"
        with open(suite_file, "w") as f:
            yaml.dump({"tasks": []}, f)
        suite = ValidationSuite(suite_name="empty3", config_dir=str(config_dir))
        runner = ValidationRunner(executor, suite)
        result = runner.run_quick()
        assert result.passed is True
        assert result.pass_rate == 1.0

    def test_evaluate_with_baseline(self, tmp_path: Path) -> None:
        mock_llm = MockLLMClient(default_response="The answer is: 2")
        executor = TaskExecutor(mock_llm)
        suite = self._make_suite(tmp_path, n_tasks=5)
        runner = ValidationRunner(executor, suite, min_pass_rate=0.0, performance_threshold=-1.0)

        # Set baseline first
        runner.set_baseline()
        result = runner.run()
        assert result.baseline_rate >= 0.0
        assert isinstance(result.delta, float)

    def test_passed_logic(self, tmp_path: Path) -> None:
        """Test that passed is True when pass_rate >= min_pass_rate AND delta >= threshold."""
        mock_llm = MockLLMClient(default_response="The answer is: 2")
        executor = TaskExecutor(mock_llm)
        suite = self._make_suite(tmp_path, n_tasks=5)
        runner = ValidationRunner(executor, suite, min_pass_rate=0.0, performance_threshold=-1.0)
        result = runner.run()
        # With min_pass_rate=0 and threshold=-1.0, should always pass
        assert result.passed is True

    def test_evaluate_empty_results(self, tmp_path: Path) -> None:
        mock_llm = MockLLMClient()
        executor = TaskExecutor(mock_llm)
        suite = self._make_suite(tmp_path)
        runner = ValidationRunner(executor, suite)
        result = runner._evaluate([])
        assert result.passed is True
        assert result.pass_rate == 1.0


# ===================================================================
# Checkpointing re-export
# ===================================================================

class TestCheckpointingReExport:
    def test_imports(self) -> None:
        from src.validation.checkpointing import StateManager, AgentState
        assert StateManager is not None
        assert AgentState is not None
