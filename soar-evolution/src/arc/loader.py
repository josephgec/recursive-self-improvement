"""ARC task loader with built-in fixture tasks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from src.arc.grid import ARCTask


# Built-in fixture tasks for testing without external data
BUILTIN_TASKS: Dict[str, dict] = {
    "simple_color_swap": {
        "train": [
            {
                "input": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "output": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            },
            {
                "input": [[1, 1, 0], [0, 0, 0], [0, 1, 1]],
                "output": [[2, 2, 0], [0, 0, 0], [0, 2, 2]],
            },
        ],
        "test": [
            {
                "input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                "output": [[0, 2, 0], [2, 0, 2], [0, 2, 0]],
            },
        ],
    },
    "pattern_fill": {
        "train": [
            {
                "input": [[0, 0, 0], [0, 5, 0], [0, 0, 0]],
                "output": [[5, 5, 5], [5, 5, 5], [5, 5, 5]],
            },
            {
                "input": [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "output": [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]],
            },
        ],
        "test": [
            {
                "input": [[0, 0], [7, 0]],
                "output": [[7, 7], [7, 7]],
            },
        ],
    },
    "grid_transform": {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[3, 1], [4, 2]],
            },
            {
                "input": [[5, 6], [7, 8]],
                "output": [[7, 5], [8, 6]],
            },
        ],
        "test": [
            {
                "input": [[9, 1], [2, 3]],
                "output": [[2, 9], [3, 1]],
            },
        ],
    },
}


class ARCLoader:
    """Loads ARC tasks from files or built-in fixtures."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self._cache: Dict[str, ARCTask] = {}

    def load_task(self, task_id: str) -> ARCTask:
        """Load a single task by ID."""
        if task_id in self._cache:
            return self._cache[task_id]

        # Try built-in tasks first
        if task_id in BUILTIN_TASKS:
            task = ARCTask.from_dict(task_id, BUILTIN_TASKS[task_id])
            self._cache[task_id] = task
            return task

        # Try loading from file
        if self.data_dir:
            file_path = self.data_dir / f"{task_id}.json"
            if file_path.exists():
                with open(file_path, "r") as f:
                    data = json.load(f)
                task = ARCTask.from_dict(task_id, data)
                self._cache[task_id] = task
                return task

        raise FileNotFoundError(
            f"Task '{task_id}' not found in built-in tasks or data directory"
        )

    def load_all(self) -> List[ARCTask]:
        """Load all available tasks."""
        tasks = []

        # Load built-in tasks
        for task_id in BUILTIN_TASKS:
            tasks.append(self.load_task(task_id))

        # Load from data directory if specified
        if self.data_dir and self.data_dir.exists():
            for file_path in sorted(self.data_dir.glob("*.json")):
                task_id = file_path.stem
                if task_id not in self._cache:
                    try:
                        tasks.append(self.load_task(task_id))
                    except Exception:
                        pass

        return tasks

    def list_task_ids(self) -> List[str]:
        """List all available task IDs."""
        ids = list(BUILTIN_TASKS.keys())

        if self.data_dir and self.data_dir.exists():
            for file_path in sorted(self.data_dir.glob("*.json")):
                task_id = file_path.stem
                if task_id not in ids:
                    ids.append(task_id)

        return ids

    def clear_cache(self):
        """Clear the task cache."""
        self._cache.clear()
