"""Checkpoint management for ablation runs."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from src.suites.base import AblationSuiteResult, ConditionRun


class CheckpointManager:
    """Save and load ablation run checkpoints for resume capability."""

    def __init__(self, checkpoint_dir: str = "data/results"):
        self.checkpoint_dir = checkpoint_dir

    def _get_path(self, suite_name: str) -> str:
        """Get checkpoint file path for a suite."""
        safe_name = suite_name.replace(" ", "_").lower()
        return os.path.join(self.checkpoint_dir, f"checkpoint_{safe_name}.json")

    def save(self, result: AblationSuiteResult) -> str:
        """Save a checkpoint of the current results."""
        path = self._get_path(result.suite_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "suite_name": result.suite_name,
            "condition_runs": {},
        }
        for cond_name, runs in result.condition_runs.items():
            data["condition_runs"][cond_name] = [
                {
                    "condition_name": run.condition_name,
                    "repetition": run.repetition,
                    "accuracy": run.accuracy,
                    "metrics": run.metrics,
                    "seed": run.seed,
                }
                for run in runs
            ]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path

    def load(self, suite_name: str) -> Optional[AblationSuiteResult]:
        """Load a checkpoint if it exists."""
        path = self._get_path(suite_name)
        if not os.path.exists(path):
            return None

        with open(path, "r") as f:
            data = json.load(f)

        result = AblationSuiteResult(suite_name=data["suite_name"])
        for cond_name, runs_data in data["condition_runs"].items():
            runs = []
            for rd in runs_data:
                runs.append(ConditionRun(
                    condition_name=rd["condition_name"],
                    repetition=rd["repetition"],
                    accuracy=rd["accuracy"],
                    metrics=rd.get("metrics", {}),
                    seed=rd.get("seed", 0),
                ))
            result.condition_runs[cond_name] = runs

        return result

    def resume(self, suite_name: str) -> Optional[AblationSuiteResult]:
        """Resume from a checkpoint (alias for load)."""
        return self.load(suite_name)

    def exists(self, suite_name: str) -> bool:
        """Check if a checkpoint exists."""
        path = self._get_path(suite_name)
        return os.path.exists(path)

    def delete(self, suite_name: str) -> bool:
        """Delete a checkpoint file."""
        path = self._get_path(suite_name)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        if not os.path.exists(self.checkpoint_dir):
            return []
        return [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_") and f.endswith(".json")
        ]
