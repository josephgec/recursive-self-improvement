"""Collapse baseline loader: load collapse trajectory data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CollapseTrajectory:
    """A model collapse trajectory."""
    schedule: str  # e.g., "standard_decay", "rapid_collapse"
    generations: List[int]
    accuracy: List[float]
    entropy: List[float]
    kl_divergence: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CollapseBaselineLoader:
    """Load and manage collapse baseline trajectories."""

    def __init__(
        self,
        num_generations: int = 20,
        initial_accuracy: float = 0.78,
        decay_rate: float = 0.02,
        entropy_decay: float = 0.05,
    ) -> None:
        self._num_generations = num_generations
        self._initial_accuracy = initial_accuracy
        self._decay_rate = decay_rate
        self._entropy_decay = entropy_decay
        self._trajectories: Dict[str, CollapseTrajectory] = {}
        self._build_defaults()

    def _build_defaults(self) -> None:
        """Build default collapse trajectories."""
        # Standard decay
        self._trajectories["standard_decay"] = self._build_trajectory(
            "standard_decay",
            self._initial_accuracy,
            self._decay_rate,
            self._entropy_decay,
        )

        # Rapid collapse
        self._trajectories["rapid_collapse"] = self._build_trajectory(
            "rapid_collapse",
            self._initial_accuracy,
            self._decay_rate * 2,
            self._entropy_decay * 2,
        )

        # Slow collapse
        self._trajectories["slow_collapse"] = self._build_trajectory(
            "slow_collapse",
            self._initial_accuracy,
            self._decay_rate * 0.5,
            self._entropy_decay * 0.5,
        )

    def _build_trajectory(
        self,
        schedule: str,
        initial_acc: float,
        decay: float,
        entropy_decay: float,
    ) -> CollapseTrajectory:
        """Build a single collapse trajectory."""
        generations = list(range(self._num_generations))
        accuracy = [max(0.0, initial_acc - decay * g) for g in generations]
        initial_entropy = 3.0
        entropy = [max(0.0, initial_entropy - entropy_decay * g) for g in generations]

        # KL divergence increases with generation
        kl_divergence = [decay * g * 0.5 for g in generations]

        return CollapseTrajectory(
            schedule=schedule,
            generations=generations,
            accuracy=accuracy,
            entropy=entropy,
            kl_divergence=kl_divergence,
        )

    def load(self, schedule: str = "standard_decay") -> CollapseTrajectory:
        """Load a collapse trajectory by schedule name."""
        if schedule not in self._trajectories:
            raise KeyError(
                f"Unknown schedule: {schedule}. "
                f"Available: {list(self._trajectories.keys())}"
            )
        return self._trajectories[schedule]

    def load_all(self) -> Dict[str, CollapseTrajectory]:
        """Load all collapse trajectories."""
        return dict(self._trajectories)

    def available_schedules(self) -> List[str]:
        """List available collapse schedules."""
        return sorted(self._trajectories.keys())

    def add_trajectory(self, trajectory: CollapseTrajectory) -> None:
        """Add a custom collapse trajectory."""
        self._trajectories[trajectory.schedule] = trajectory
