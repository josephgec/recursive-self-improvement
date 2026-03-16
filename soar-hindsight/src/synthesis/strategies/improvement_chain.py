"""Improvement chain strategy: multi-step refinement sequences."""

from __future__ import annotations

from typing import List

from src.collection.trajectory import SearchTrajectory
from src.synthesis.synthesizer import TrainingPair
from src.utils.tokenization import count_tokens


class ImprovementChainStrategy:
    """Generate training pairs from multi-step improvement chains.

    Creates (current_code + instruction -> improved_code) pairs from
    sequential improvements in the trajectory.
    """

    name = "improvement_chain"

    def __init__(self, min_steps: int = 2, max_steps: int = 10):
        self.min_steps = min_steps
        self.max_steps = max_steps

    def generate(self, trajectories: List[SearchTrajectory]) -> List[TrainingPair]:
        pairs: List[TrainingPair] = []
        for traj in trajectories:
            if not traj.task:
                continue

            # Extract improvement steps if not done
            if not traj.improvement_steps:
                traj.extract_improvement_chain()

            steps = traj.improvement_steps
            if len(steps) < self.min_steps:
                continue

            # Limit to max_steps
            steps = steps[: self.max_steps]

            # Generate a pair for each step
            for i, step in enumerate(steps):
                if not step.code_before.strip() or not step.code_after.strip():
                    continue

                prompt = (
                    f"Improve the following code for the task: {traj.task.description}\n\n"
                    f"Current code (fitness {step.fitness_before:.2f}):\n"
                    f"```\n{step.code_before}\n```\n\n"
                    f"Apply the '{step.operator}' operator to improve this code."
                )
                completion = step.code_after

                quality = step.fitness_after * (step.fitness_delta + 0.1)
                quality = min(1.0, max(0.0, quality))

                pair = TrainingPair(
                    strategy=self.name,
                    task_id=traj.task.task_id,
                    prompt=prompt,
                    completion=completion,
                    quality_score=quality,
                    metadata={
                        "trajectory_id": traj.trajectory_id,
                        "step_index": i,
                        "total_steps": len(steps),
                        "operator": step.operator,
                        "fitness_before": step.fitness_before,
                        "fitness_after": step.fitness_after,
                    },
                    prompt_tokens=count_tokens(prompt),
                    completion_tokens=count_tokens(completion),
                )
                pairs.append(pair)

            # Also generate a cumulative chain pair (first -> last)
            if len(steps) >= 2:
                first_code = steps[0].code_before
                last_code = steps[-1].code_after
                if first_code.strip() and last_code.strip():
                    prompt = (
                        f"Significantly improve the following code for the task: "
                        f"{traj.task.description}\n\n"
                        f"Initial code (fitness {steps[0].fitness_before:.2f}):\n"
                        f"```\n{first_code}\n```\n\n"
                        f"Provide a much improved version."
                    )
                    pair = TrainingPair(
                        strategy=self.name,
                        task_id=traj.task.task_id,
                        prompt=prompt,
                        completion=last_code,
                        quality_score=steps[-1].fitness_after,
                        metadata={
                            "trajectory_id": traj.trajectory_id,
                            "type": "cumulative",
                            "chain_length": len(steps),
                        },
                        prompt_tokens=count_tokens(prompt),
                        completion_tokens=count_tokens(last_code),
                    )
                    pairs.append(pair)

        return pairs
