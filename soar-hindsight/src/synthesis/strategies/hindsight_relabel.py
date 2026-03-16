"""Hindsight relabel strategy: partial solutions as targets for simpler tasks."""

from __future__ import annotations

from typing import List

from src.collection.trajectory import SearchTrajectory
from src.synthesis.synthesizer import TrainingPair
from src.utils.tokenization import count_tokens


class HindsightRelabelStrategy:
    """Relabel partial solutions as correct solutions for relaxed tasks.

    When a search achieves partial success (fitness < 1.0), we relabel the
    best individual as the target for a simpler version of the task.
    """

    name = "hindsight_relabel"

    def __init__(self, fitness_threshold: float = 0.3):
        self.fitness_threshold = fitness_threshold

    def generate(self, trajectories: List[SearchTrajectory]) -> List[TrainingPair]:
        pairs: List[TrainingPair] = []
        for traj in trajectories:
            if not traj.task:
                continue

            best = traj.best_individual
            if best is None:
                continue

            # Only use partial solutions (not fully solved, but above threshold)
            if best.fitness >= 1.0 or best.fitness < self.fitness_threshold:
                continue

            if not best.code.strip():
                continue

            # Create a relaxed task description
            relaxed_desc = self._relax_task(traj.task.description, best.fitness)

            prompt = f"Solve the following programming task:\n\n{relaxed_desc}"
            completion = best.code

            pair = TrainingPair(
                strategy=self.name,
                task_id=traj.task.task_id,
                prompt=prompt,
                completion=completion,
                quality_score=best.fitness * 0.7,  # Discount for relabeled
                metadata={
                    "trajectory_id": traj.trajectory_id,
                    "original_task": traj.task.description,
                    "original_fitness": best.fitness,
                    "relabeled": True,
                },
                prompt_tokens=count_tokens(prompt),
                completion_tokens=count_tokens(completion),
            )
            pairs.append(pair)

        return pairs

    @staticmethod
    def _relax_task(description: str, fitness: float) -> str:
        """Create a relaxed version of the task description.

        The relaxation level depends on how close the partial solution is.
        """
        if fitness >= 0.7:
            prefix = "Provide a partial solution that handles the main cases for"
        elif fitness >= 0.4:
            prefix = "Write a basic implementation that addresses the core logic of"
        else:
            prefix = "Write a simple sketch or outline for"

        # Strip leading "Write"/"Implement"/"Create" if present
        desc_lower = description.lower()
        for verb in ["write ", "implement ", "create ", "design ", "build "]:
            if desc_lower.startswith(verb):
                description = description[len(verb):]
                break

        return f"{prefix} the following task: {description}"
