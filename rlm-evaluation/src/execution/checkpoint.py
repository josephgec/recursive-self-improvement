"""Checkpoint manager for saving and resuming benchmark runs."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from src.benchmarks.task import EvalResult


class CheckpointManager:
    """Save, load, and resume benchmark evaluation state."""

    def __init__(self, checkpoint_dir: str = "data/results") -> None:
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _checkpoint_path(self, run_id: str) -> str:
        """Get the path for a checkpoint file."""
        return os.path.join(self.checkpoint_dir, f"checkpoint_{run_id}.json")

    def save(self, run_id: str, results: List[EvalResult], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save checkpoint to disk."""
        path = self._checkpoint_path(run_id)
        data = {
            "run_id": run_id,
            "num_results": len(results),
            "results": [r.to_dict() for r in results],
            "metadata": metadata or {},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def load(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint from disk."""
        path = self._checkpoint_path(run_id)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        # Reconstruct EvalResult objects
        data["results"] = [EvalResult.from_dict(r) for r in data["results"]]
        return data

    def resume(self, run_id: str) -> List[EvalResult]:
        """Resume from checkpoint, returning completed results."""
        checkpoint = self.load(run_id)
        if checkpoint is None:
            return []
        return checkpoint["results"]

    def get_completed_task_ids(self, run_id: str) -> set:
        """Get the set of task IDs already completed."""
        results = self.resume(run_id)
        return {r.task_id for r in results}

    def exists(self, run_id: str) -> bool:
        """Check if a checkpoint exists."""
        return os.path.exists(self._checkpoint_path(run_id))

    def delete(self, run_id: str) -> bool:
        """Delete a checkpoint."""
        path = self._checkpoint_path(run_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoint run IDs."""
        checkpoints = []
        if os.path.exists(self.checkpoint_dir):
            for fname in os.listdir(self.checkpoint_dir):
                if fname.startswith("checkpoint_") and fname.endswith(".json"):
                    run_id = fname[len("checkpoint_"):-len(".json")]
                    checkpoints.append(run_id)
        return checkpoints
