"""Parquet-like JSON storage for trajectories with querying support."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.collection.trajectory import SearchTrajectory


class TrajectoryDatabase:
    """JSON-based storage for trajectories with query capabilities.

    Stores each trajectory as a separate JSON file in a flat directory,
    with a manifest index for fast lookups.
    """

    def __init__(self, db_dir: str = "data/trajectories"):
        self.db_dir = db_dir
        self._manifest: Dict[str, Dict[str, Any]] = {}
        self._cache: Dict[str, SearchTrajectory] = {}

    def initialize(self) -> None:
        """Create database directory and load existing manifest."""
        os.makedirs(self.db_dir, exist_ok=True)
        manifest_path = os.path.join(self.db_dir, "_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                self._manifest = json.load(f)

    def store(self, trajectory: SearchTrajectory) -> str:
        """Store a trajectory and return its ID."""
        os.makedirs(self.db_dir, exist_ok=True)
        tid = trajectory.trajectory_id
        filepath = os.path.join(self.db_dir, f"{tid}.json")

        with open(filepath, "w") as f:
            json.dump(trajectory.to_dict(), f, indent=2)

        self._manifest[tid] = {
            "trajectory_id": tid,
            "task_id": trajectory.task.task_id if trajectory.task else None,
            "best_fitness": trajectory.best_fitness,
            "solved": trajectory.solved,
            "total_generations": trajectory.total_generations,
            "n_individuals": len(trajectory.individuals),
        }
        self._cache[tid] = trajectory
        self._save_manifest()
        return tid

    def load(self, trajectory_id: str) -> Optional[SearchTrajectory]:
        """Load a trajectory by ID."""
        if trajectory_id in self._cache:
            return self._cache[trajectory_id]

        filepath = os.path.join(self.db_dir, f"{trajectory_id}.json")
        if not os.path.exists(filepath):
            return None

        with open(filepath, "r") as f:
            data = json.load(f)
        traj = SearchTrajectory.from_dict(data)
        self._cache[trajectory_id] = traj
        return traj

    def delete(self, trajectory_id: str) -> bool:
        """Delete a trajectory by ID."""
        filepath = os.path.join(self.db_dir, f"{trajectory_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        self._manifest.pop(trajectory_id, None)
        self._cache.pop(trajectory_id, None)
        self._save_manifest()
        return True

    def list_ids(self) -> List[str]:
        """List all trajectory IDs."""
        return list(self._manifest.keys())

    def query_by_fitness(
        self, min_fitness: float = 0.0, max_fitness: float = 1.0
    ) -> List[str]:
        """Query trajectory IDs by fitness range."""
        return [
            tid
            for tid, meta in self._manifest.items()
            if min_fitness <= meta.get("best_fitness", 0.0) <= max_fitness
        ]

    def query_by_solved(self, solved: bool = True) -> List[str]:
        """Query trajectory IDs by solved status."""
        return [
            tid
            for tid, meta in self._manifest.items()
            if meta.get("solved", False) == solved
        ]

    def query_by_task(self, task_id: str) -> List[str]:
        """Query trajectory IDs by task ID."""
        return [
            tid
            for tid, meta in self._manifest.items()
            if meta.get("task_id") == task_id
        ]

    def query(self, predicate: Callable[[Dict[str, Any]], bool]) -> List[str]:
        """Query trajectory IDs by a custom predicate on manifest entries."""
        return [
            tid
            for tid, meta in self._manifest.items()
            if predicate(meta)
        ]

    def count(self) -> int:
        """Return total number of stored trajectories."""
        return len(self._manifest)

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics of the database."""
        if not self._manifest:
            return {"total": 0, "solved": 0, "avg_fitness": 0.0}
        total = len(self._manifest)
        solved = sum(1 for m in self._manifest.values() if m.get("solved", False))
        avg_fitness = sum(m.get("best_fitness", 0.0) for m in self._manifest.values()) / total
        return {
            "total": total,
            "solved": solved,
            "avg_fitness": round(avg_fitness, 4),
        }

    def _save_manifest(self) -> None:
        """Persist the manifest to disk."""
        os.makedirs(self.db_dir, exist_ok=True)
        manifest_path = os.path.join(self.db_dir, "_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)
