"""Fitness computation helpers."""

from __future__ import annotations

from typing import Optional

from src.operators.thinking_evaluator import FitnessDetails


def compute_composite_fitness(
    accuracy: float,
    reasoning_score: float,
    consistency_score: float,
    accuracy_weight: float = 0.7,
    reasoning_weight: float = 0.15,
    consistency_weight: float = 0.15,
) -> float:
    """Compute weighted composite fitness from component scores.

    All inputs should be in [0, 1].
    Returns composite fitness in [0, 1].
    """
    composite = (
        accuracy_weight * accuracy
        + reasoning_weight * reasoning_score
        + consistency_weight * consistency_score
    )
    return max(0.0, min(1.0, composite))


def fitness_details_to_dict(details: FitnessDetails) -> dict:
    """Convert FitnessDetails to a plain dictionary."""
    return {
        "accuracy": details.accuracy,
        "reasoning_score": details.reasoning_score,
        "consistency_score": details.consistency_score,
        "composite_fitness": details.composite_fitness,
        "section_scores": dict(details.section_scores),
        "num_tasks": len(details.task_evaluations),
        "num_correct": sum(
            1 for te in details.task_evaluations if te.is_correct
        ),
    }
