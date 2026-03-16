"""Crossover pairs strategy: parent_a + parent_b -> child pairs."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.collection.trajectory import IndividualRecord, SearchTrajectory
from src.synthesis.synthesizer import TrainingPair
from src.utils.tokenization import count_tokens


class CrossoverPairsStrategy:
    """Generate training pairs from crossover events.

    When two parents are combined to produce a child, create
    (parent_a_code + parent_b_code -> child_code) pairs.
    """

    name = "crossover_pairs"

    def generate(self, trajectories: List[SearchTrajectory]) -> List[TrainingPair]:
        pairs: List[TrainingPair] = []
        for traj in trajectories:
            if not traj.task:
                continue

            # Build lookup of individuals by ID
            ind_map: Dict[str, IndividualRecord] = {
                ind.individual_id: ind for ind in traj.individuals
            }

            for ind in traj.individuals:
                if len(ind.parent_ids) < 2:
                    continue

                parent_a = ind_map.get(ind.parent_ids[0])
                parent_b = ind_map.get(ind.parent_ids[1])

                if parent_a is None or parent_b is None:
                    continue

                if not parent_a.code.strip() or not parent_b.code.strip() or not ind.code.strip():
                    continue

                # Only include if child is at least as good as best parent
                best_parent_fitness = max(parent_a.fitness, parent_b.fitness)
                if ind.fitness < best_parent_fitness * 0.8:
                    continue

                prompt = (
                    f"Combine the following two solutions for the task: {traj.task.description}\n\n"
                    f"Solution A (fitness {parent_a.fitness:.2f}):\n"
                    f"```\n{parent_a.code}\n```\n\n"
                    f"Solution B (fitness {parent_b.fitness:.2f}):\n"
                    f"```\n{parent_b.code}\n```\n\n"
                    f"Produce a combined solution that takes the best parts of both."
                )
                completion = ind.code

                quality = ind.fitness * (
                    1.0 if ind.fitness >= best_parent_fitness else 0.7
                )

                pair = TrainingPair(
                    strategy=self.name,
                    task_id=traj.task.task_id,
                    prompt=prompt,
                    completion=completion,
                    quality_score=min(1.0, quality),
                    metadata={
                        "trajectory_id": traj.trajectory_id,
                        "child_id": ind.individual_id,
                        "parent_a_id": parent_a.individual_id,
                        "parent_b_id": parent_b.individual_id,
                        "parent_a_fitness": parent_a.fitness,
                        "parent_b_fitness": parent_b.fitness,
                        "child_fitness": ind.fitness,
                    },
                    prompt_tokens=count_tokens(prompt),
                    completion_tokens=count_tokens(completion),
                )
                pairs.append(pair)

        return pairs
