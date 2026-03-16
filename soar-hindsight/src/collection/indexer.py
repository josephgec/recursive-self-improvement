"""Index trajectories by task, fitness, and operator type."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from src.collection.trajectory import SearchTrajectory


class TrajectoryIndexer:
    """Indexes trajectories for fast lookup by various attributes."""

    def __init__(self) -> None:
        self._by_task: Dict[str, List[str]] = defaultdict(list)
        self._by_fitness_bucket: Dict[str, List[str]] = defaultdict(list)
        self._by_operator: Dict[str, List[str]] = defaultdict(list)
        self._by_solved: Dict[bool, List[str]] = defaultdict(list)
        self._all_ids: Set[str] = set()

    def index(self, trajectory: SearchTrajectory) -> None:
        """Index a single trajectory."""
        tid = trajectory.trajectory_id

        if tid in self._all_ids:
            return
        self._all_ids.add(tid)

        # Index by task
        if trajectory.task:
            self._by_task[trajectory.task.task_id].append(tid)

        # Index by fitness bucket (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
        bucket = self._fitness_bucket(trajectory.best_fitness)
        self._by_fitness_bucket[bucket].append(tid)

        # Index by operator types used
        operators = set()
        for ind in trajectory.individuals:
            if ind.operator:
                operators.add(ind.operator)
        for op in operators:
            self._by_operator[op].append(tid)

        # Index by solved status
        self._by_solved[trajectory.solved].append(tid)

    def index_many(self, trajectories: List[SearchTrajectory]) -> int:
        """Index multiple trajectories, return count indexed."""
        count = 0
        for traj in trajectories:
            if traj.trajectory_id not in self._all_ids:
                self.index(traj)
                count += 1
        return count

    def lookup_by_task(self, task_id: str) -> List[str]:
        """Look up trajectory IDs by task ID."""
        return list(self._by_task.get(task_id, []))

    def lookup_by_fitness(self, min_fitness: float = 0.0, max_fitness: float = 1.0) -> List[str]:
        """Look up trajectory IDs by fitness range."""
        result = []
        for bucket_key, tids in self._by_fitness_bucket.items():
            low = float(bucket_key.split("-")[0])
            high = float(bucket_key.split("-")[1])
            if high >= min_fitness and low <= max_fitness:
                result.extend(tids)
        return result

    def lookup_by_operator(self, operator: str) -> List[str]:
        """Look up trajectory IDs by operator type."""
        return list(self._by_operator.get(operator, []))

    def lookup_solved(self, solved: bool = True) -> List[str]:
        """Look up trajectory IDs by solved status."""
        return list(self._by_solved.get(solved, []))

    def all_tasks(self) -> List[str]:
        """Return all indexed task IDs."""
        return list(self._by_task.keys())

    def all_operators(self) -> List[str]:
        """Return all indexed operator types."""
        return list(self._by_operator.keys())

    def total_indexed(self) -> int:
        """Return total number of indexed trajectories."""
        return len(self._all_ids)

    def summary(self) -> Dict[str, Any]:
        """Return summary of the index."""
        return {
            "total_indexed": self.total_indexed(),
            "unique_tasks": len(self._by_task),
            "unique_operators": len(self._by_operator),
            "solved_count": len(self._by_solved.get(True, [])),
            "unsolved_count": len(self._by_solved.get(False, [])),
            "fitness_distribution": {
                k: len(v) for k, v in sorted(self._by_fitness_bucket.items())
            },
        }

    def clear(self) -> None:
        """Clear all indexes."""
        self._by_task.clear()
        self._by_fitness_bucket.clear()
        self._by_operator.clear()
        self._by_solved.clear()
        self._all_ids.clear()

    @staticmethod
    def _fitness_bucket(fitness: float) -> str:
        """Map a fitness value to a bucket key."""
        clamped = max(0.0, min(1.0, fitness))
        low = int(clamped * 10) / 10.0
        high = min(low + 0.1, 1.0)
        return f"{low:.1f}-{high:.1f}"
