"""Collector for harvesting search results and extracting improvement chains."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.collection.trajectory import (
    IndividualRecord,
    ImprovementStep,
    SearchTrajectory,
    TaskSpec,
)


class TrajectoryCollector:
    """Harvests search results from trajectory files and extracts improvement chains."""

    def __init__(self, trajectory_dir: str = "data/trajectories", min_fitness: float = 0.0):
        self.trajectory_dir = trajectory_dir
        self.min_fitness = min_fitness
        self._trajectories: List[SearchTrajectory] = []

    @property
    def trajectories(self) -> List[SearchTrajectory]:
        return list(self._trajectories)

    @property
    def count(self) -> int:
        return len(self._trajectories)

    def collect_from_file(self, filepath: str) -> Optional[SearchTrajectory]:
        """Load a single trajectory from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        traj = SearchTrajectory.from_dict(data)
        if traj.best_fitness >= self.min_fitness:
            self._trajectories.append(traj)
            return traj
        return None

    def collect_from_directory(self, directory: Optional[str] = None) -> List[SearchTrajectory]:
        """Load all trajectories from a directory."""
        target_dir = directory or self.trajectory_dir
        collected = []
        if not os.path.isdir(target_dir):
            return collected

        for fname in sorted(os.listdir(target_dir)):
            if fname.endswith(".json"):
                filepath = os.path.join(target_dir, fname)
                traj = self.collect_from_file(filepath)
                if traj is not None:
                    collected.append(traj)
        return collected

    def collect_from_dicts(self, data_list: List[Dict[str, Any]]) -> List[SearchTrajectory]:
        """Load trajectories from a list of dictionaries (for testing)."""
        collected = []
        for data in data_list:
            traj = SearchTrajectory.from_dict(data)
            if traj.best_fitness >= self.min_fitness:
                self._trajectories.append(traj)
                collected.append(traj)
        return collected

    def extract_improvement_chains(self) -> List[List[ImprovementStep]]:
        """Extract improvement chains from all collected trajectories."""
        chains = []
        for traj in self._trajectories:
            chain = traj.extract_improvement_chain()
            if chain:
                chains.append(chain)
        return chains

    def get_solved_trajectories(self) -> List[SearchTrajectory]:
        """Return only trajectories that solved their task."""
        return [t for t in self._trajectories if t.solved]

    def get_partial_trajectories(self, min_fitness: float = 0.0, max_fitness: float = 1.0) -> List[SearchTrajectory]:
        """Return trajectories with best fitness in the given range."""
        return [
            t for t in self._trajectories
            if min_fitness <= t.best_fitness < max_fitness
        ]

    def get_failed_trajectories(self) -> List[SearchTrajectory]:
        """Return trajectories that have error individuals."""
        return [
            t for t in self._trajectories
            if any(ind.error is not None for ind in t.individuals)
        ]

    def get_crossover_candidates(self) -> List[SearchTrajectory]:
        """Return trajectories that have individuals with multiple parents."""
        return [
            t for t in self._trajectories
            if any(len(ind.parent_ids) >= 2 for ind in t.individuals)
        ]

    def clear(self) -> None:
        """Clear all collected trajectories."""
        self._trajectories.clear()

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics of collected trajectories."""
        if not self._trajectories:
            return {
                "total": 0,
                "solved": 0,
                "partial": 0,
                "avg_fitness": 0.0,
                "avg_generations": 0.0,
            }
        total = len(self._trajectories)
        solved = len(self.get_solved_trajectories())
        avg_fitness = sum(t.best_fitness for t in self._trajectories) / total
        avg_gens = sum(t.total_generations for t in self._trajectories) / total
        return {
            "total": total,
            "solved": solved,
            "partial": total - solved,
            "avg_fitness": round(avg_fitness, 4),
            "avg_generations": round(avg_gens, 2),
        }
