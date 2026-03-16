"""Direct solution strategy: task -> solution pairs from solved tasks."""

from __future__ import annotations

from typing import List

from src.collection.trajectory import SearchTrajectory
from src.synthesis.synthesizer import TrainingPair
from src.utils.tokenization import count_tokens


class DirectSolutionStrategy:
    """Generate training pairs from tasks that were successfully solved.

    For each solved trajectory, creates a (task_description -> best_solution) pair.
    """

    name = "direct_solution"

    def __init__(self, min_fitness: float = 0.5):
        self.min_fitness = min_fitness

    def generate(self, trajectories: List[SearchTrajectory]) -> List[TrainingPair]:
        pairs: List[TrainingPair] = []
        for traj in trajectories:
            if not traj.task:
                continue
            best = traj.best_individual
            if best is None or best.fitness < self.min_fitness:
                continue

            prompt = f"Solve the following programming task:\n\n{traj.task.description}"
            completion = best.code

            if not prompt.strip() or not completion.strip():
                continue

            pair = TrainingPair(
                strategy=self.name,
                task_id=traj.task.task_id,
                prompt=prompt,
                completion=completion,
                quality_score=best.fitness,
                metadata={
                    "trajectory_id": traj.trajectory_id,
                    "generation": best.generation,
                    "fitness": best.fitness,
                },
                prompt_tokens=count_tokens(prompt),
                completion_tokens=count_tokens(completion),
            )
            pairs.append(pair)
        return pairs
