"""Immutable validation suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.core.executor import Task


class ValidationSuite:
    """Immutable validation task suite loaded from config."""

    def __init__(self, suite_name: str = "core", config_dir: str = "configs/validation_suites") -> None:
        self._suite_name = suite_name
        self._tasks: tuple[Task, ...] = ()
        self._load(suite_name, config_dir)

    def _load(self, suite_name: str, config_dir: str) -> None:
        """Load validation tasks from YAML config."""
        config_path = Path(config_dir) / f"{suite_name}.yaml"
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
            tasks = []
            for item in data.get("tasks", []):
                tasks.append(Task(
                    task_id=item.get("id", ""),
                    question=item.get("question", ""),
                    expected_answer=str(item.get("answer", "")),
                    domain=item.get("domain", "validation"),
                    category=item.get("category", ""),
                    difficulty=item.get("difficulty", "medium"),
                ))
            self._tasks = tuple(tasks)

    def get_tasks(self) -> list[Task]:
        """Return all validation tasks (as a new list)."""
        return list(self._tasks)

    @property
    def size(self) -> int:
        """Number of validation tasks."""
        return len(self._tasks)

    @property
    def suite_name(self) -> str:
        return self._suite_name
