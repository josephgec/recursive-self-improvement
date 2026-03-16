"""Pattern description strategy: task -> description -> code pairs."""

from __future__ import annotations

from typing import List

from src.collection.trajectory import SearchTrajectory
from src.synthesis.synthesizer import TrainingPair
from src.utils.tokenization import count_tokens


class PatternDescriptionStrategy:
    """Generate training pairs that describe the solution pattern before coding.

    Creates (task -> pattern_description + code) pairs to teach the model
    to reason about the approach before writing code.
    """

    name = "pattern_description"

    def generate(self, trajectories: List[SearchTrajectory]) -> List[TrainingPair]:
        pairs: List[TrainingPair] = []
        for traj in trajectories:
            if not traj.task:
                continue

            best = traj.best_individual
            if best is None or best.fitness < 0.5:
                continue

            if not best.code.strip():
                continue

            # Generate a pattern description from the trajectory
            pattern_desc = self._describe_pattern(traj)

            prompt = (
                f"Describe the solution approach and then implement it for:\n\n"
                f"{traj.task.description}"
            )

            completion = (
                f"## Approach\n{pattern_desc}\n\n"
                f"## Implementation\n```\n{best.code}\n```"
            )

            pair = TrainingPair(
                strategy=self.name,
                task_id=traj.task.task_id,
                prompt=prompt,
                completion=completion,
                quality_score=best.fitness * 0.9,
                metadata={
                    "trajectory_id": traj.trajectory_id,
                    "pattern": pattern_desc,
                    "fitness": best.fitness,
                },
                prompt_tokens=count_tokens(prompt),
                completion_tokens=count_tokens(completion),
            )
            pairs.append(pair)

        return pairs

    def _describe_pattern(self, traj: SearchTrajectory) -> str:
        """Generate a natural language description of the solution pattern."""
        parts = []

        # Describe the operators that led to improvements
        if not traj.improvement_steps:
            traj.extract_improvement_chain()

        if traj.improvement_steps:
            operators_used = [s.operator for s in traj.improvement_steps if s.operator]
            if operators_used:
                unique_ops = list(dict.fromkeys(operators_used))  # preserve order
                parts.append(
                    f"Key transformations used: {', '.join(unique_ops)}."
                )

            total_gain = sum(s.fitness_delta for s in traj.improvement_steps)
            parts.append(
                f"Total fitness improvement: {total_gain:.2f} over "
                f"{len(traj.improvement_steps)} steps."
            )

        # Describe the task characteristics
        if traj.task and traj.task.tags:
            parts.append(f"Task categories: {', '.join(traj.task.tags)}.")

        if traj.task and traj.task.difficulty:
            parts.append(f"Difficulty level: {traj.task.difficulty}.")

        if not parts:
            parts.append("Direct solution approach.")

        return " ".join(parts)
