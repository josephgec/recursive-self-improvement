"""Error correction strategy: buggy_code + error -> fixed_code pairs."""

from __future__ import annotations

from typing import List

from src.collection.trajectory import SearchTrajectory
from src.synthesis.synthesizer import TrainingPair
from src.utils.tokenization import count_tokens


class ErrorCorrectionStrategy:
    """Generate training pairs from error correction events.

    Finds individuals that had errors and were later fixed, creating
    (buggy_code + error_message -> fixed_code) pairs.
    """

    name = "error_correction"

    def __init__(self, min_attempts: int = 1):
        self.min_attempts = min_attempts

    def generate(self, trajectories: List[SearchTrajectory]) -> List[TrainingPair]:
        pairs: List[TrainingPair] = []
        for traj in trajectories:
            if not traj.task:
                continue

            # Extract improvement steps if not done
            if not traj.improvement_steps:
                traj.extract_improvement_chain()

            # Look for steps where error was fixed
            for step in traj.improvement_steps:
                if step.error_before and not step.error_after and step.is_improvement:
                    prompt = (
                        f"Fix the following buggy code for the task: {traj.task.description}\n\n"
                        f"Buggy code:\n```\n{step.code_before}\n```\n\n"
                        f"Error:\n{step.error_before}\n\n"
                        f"Provide the corrected code."
                    )
                    completion = step.code_after

                    if not completion.strip():
                        continue

                    pair = TrainingPair(
                        strategy=self.name,
                        task_id=traj.task.task_id,
                        prompt=prompt,
                        completion=completion,
                        quality_score=step.fitness_after,
                        metadata={
                            "trajectory_id": traj.trajectory_id,
                            "operator": step.operator,
                            "fitness_delta": step.fitness_delta,
                            "error": step.error_before,
                        },
                        prompt_tokens=count_tokens(prompt),
                        completion_tokens=count_tokens(completion),
                    )
                    pairs.append(pair)

            # Also look at individual-level error patterns
            error_inds = [i for i in traj.individuals if i.error is not None]
            fixed_inds = [i for i in traj.individuals if i.error is None and i.fitness > 0]

            if len(error_inds) >= self.min_attempts and fixed_inds:
                best_fixed = max(fixed_inds, key=lambda x: x.fitness)
                for err_ind in error_inds:
                    if err_ind.code == best_fixed.code:
                        continue
                    prompt = (
                        f"Fix the following buggy code for the task: {traj.task.description}\n\n"
                        f"Buggy code:\n```\n{err_ind.code}\n```\n\n"
                        f"Error:\n{err_ind.error}\n\n"
                        f"Provide the corrected code."
                    )
                    completion = best_fixed.code

                    if not completion.strip():
                        continue

                    pair = TrainingPair(
                        strategy=self.name,
                        task_id=traj.task.task_id,
                        prompt=prompt,
                        completion=completion,
                        quality_score=best_fixed.fitness * 0.8,
                        metadata={
                            "trajectory_id": traj.trajectory_id,
                            "error_individual": err_ind.individual_id,
                            "fixed_individual": best_fixed.individual_id,
                        },
                        prompt_tokens=count_tokens(prompt),
                        completion_tokens=count_tokens(completion),
                    )
                    pairs.append(pair)

        return pairs
