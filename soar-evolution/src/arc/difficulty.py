"""Estimate ARC task difficulty based on structural features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from src.arc.grid import ARCTask, diff_grids


@dataclass
class DifficultyEstimate:
    """Estimated difficulty of an ARC task."""

    score: float  # 0.0 (easy) to 1.0 (hard)
    factors: Dict[str, float]
    level: str  # "easy", "medium", "hard"

    @classmethod
    def from_score(cls, score: float, factors: Dict[str, float]) -> DifficultyEstimate:
        score = max(0.0, min(1.0, score))
        if score < 0.33:
            level = "easy"
        elif score < 0.66:
            level = "medium"
        else:
            level = "hard"
        return cls(score=score, factors=factors, level=level)


def estimate_difficulty(task: ARCTask) -> DifficultyEstimate:
    """Estimate the difficulty of an ARC task based on structural features."""
    factors: Dict[str, float] = {}

    # Factor 1: Grid size (larger grids = harder)
    max_pixels = 0
    for ex in task.all_examples():
        pixels = max(ex.input_grid.pixel_count(), ex.output_grid.pixel_count())
        max_pixels = max(max_pixels, pixels)
    factors["grid_size"] = min(max_pixels / 900.0, 1.0)  # 30x30 = max

    # Factor 2: Number of colors used
    all_colors = set()
    for ex in task.all_examples():
        all_colors |= ex.input_grid.colors_used()
        all_colors |= ex.output_grid.colors_used()
    factors["color_variety"] = min(len(all_colors) / 10.0, 1.0)

    # Factor 3: Shape changes between input and output
    shape_changes = 0
    for ex in task.train:
        if ex.input_grid.shape != ex.output_grid.shape:
            shape_changes += 1
    factors["shape_change"] = (
        shape_changes / max(len(task.train), 1)
    )

    # Factor 4: Transformation complexity (average change ratio)
    total_change_ratio = 0.0
    for ex in task.train:
        d = diff_grids(ex.input_grid, ex.output_grid)
        total_change_ratio += d.change_ratio
    factors["change_ratio"] = total_change_ratio / max(len(task.train), 1)

    # Factor 5: Number of training examples (fewer = harder)
    factors["few_examples"] = max(0.0, 1.0 - task.num_train / 5.0)

    # Factor 6: Consistency of grid sizes across examples
    sizes = set()
    for ex in task.train:
        sizes.add(ex.input_grid.shape)
    factors["size_variety"] = min((len(sizes) - 1) / 3.0, 1.0) if len(sizes) > 1 else 0.0

    # Weighted combination
    weights = {
        "grid_size": 0.15,
        "color_variety": 0.15,
        "shape_change": 0.20,
        "change_ratio": 0.20,
        "few_examples": 0.15,
        "size_variety": 0.15,
    }

    score = sum(factors[k] * weights[k] for k in factors)
    return DifficultyEstimate.from_score(score, factors)
