"""Tests for task loaders: math_tasks, code_tasks, loader."""

from __future__ import annotations

import pytest

from src.core.executor import Task
from src.tasks.math_tasks import MathTaskLoader
from src.tasks.code_tasks import CodeTaskLoader
from src.tasks.loader import TaskSuiteLoader


# ===================================================================
# MathTaskLoader
# ===================================================================

class TestMathTaskLoader:
    def test_load_returns_tasks(self) -> None:
        loader = MathTaskLoader()
        tasks = loader.load()
        assert isinstance(tasks, list)
        assert len(tasks) >= 50

    def test_all_tasks_are_task_instances(self) -> None:
        loader = MathTaskLoader()
        for t in loader.load():
            assert isinstance(t, Task)

    def test_task_ids_are_unique(self) -> None:
        loader = MathTaskLoader()
        ids = [t.task_id for t in loader.load()]
        assert len(ids) == len(set(ids))

    def test_all_tasks_have_domain_math(self) -> None:
        loader = MathTaskLoader()
        for t in loader.load():
            assert t.domain == "math"

    def test_all_tasks_have_expected_answer(self) -> None:
        loader = MathTaskLoader()
        for t in loader.load():
            assert t.expected_answer != ""

    def test_all_tasks_have_category(self) -> None:
        loader = MathTaskLoader()
        for t in loader.load():
            assert t.category != ""

    def test_categories_present(self) -> None:
        loader = MathTaskLoader()
        categories = {t.category for t in loader.load()}
        assert "arithmetic" in categories
        assert "algebra" in categories
        assert "geometry" in categories


# ===================================================================
# CodeTaskLoader
# ===================================================================

class TestCodeTaskLoader:
    def test_load_returns_tasks(self) -> None:
        loader = CodeTaskLoader()
        tasks = loader.load()
        assert isinstance(tasks, list)
        assert len(tasks) >= 20

    def test_all_tasks_are_task_instances(self) -> None:
        loader = CodeTaskLoader()
        for t in loader.load():
            assert isinstance(t, Task)

    def test_task_ids_are_unique(self) -> None:
        loader = CodeTaskLoader()
        ids = [t.task_id for t in loader.load()]
        assert len(ids) == len(set(ids))

    def test_all_tasks_have_domain_code(self) -> None:
        loader = CodeTaskLoader()
        for t in loader.load():
            assert t.domain == "code"

    def test_all_tasks_have_expected_answer(self) -> None:
        loader = CodeTaskLoader()
        for t in loader.load():
            assert t.expected_answer != ""

    def test_categories_present(self) -> None:
        loader = CodeTaskLoader()
        categories = {t.category for t in loader.load()}
        assert "strings" in categories
        assert "lists" in categories


# ===================================================================
# TaskSuiteLoader
# ===================================================================

class TestTaskSuiteLoader:
    def test_load_math_domain(self) -> None:
        tasks = TaskSuiteLoader.load("math")
        assert len(tasks) >= 50
        assert all(t.domain == "math" for t in tasks)

    def test_load_code_domain(self) -> None:
        tasks = TaskSuiteLoader.load("code")
        assert len(tasks) >= 20
        assert all(t.domain == "code" for t in tasks)

    def test_load_all_domain(self) -> None:
        tasks = TaskSuiteLoader.load("all")
        domains = {t.domain for t in tasks}
        assert "math" in domains
        assert "code" in domains
        assert len(tasks) >= 70  # 50+ math + 20+ code

    def test_load_unknown_domain_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown domain"):
            TaskSuiteLoader.load("imaginary")

    def test_available_domains(self) -> None:
        domains = TaskSuiteLoader.available_domains()
        assert "math" in domains
        assert "code" in domains
        assert "all" in domains
