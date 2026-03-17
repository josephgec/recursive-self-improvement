"""Measure behavioral change magnitude."""

from typing import Dict, List, Optional
import numpy as np


def measure_behavioral_change(before_outputs: Dict[str, str],
                               after_outputs: Dict[str, str]) -> float:
    """Measure the magnitude of behavioral change between model outputs.

    Uses simple text similarity: fraction of outputs that changed.

    Args:
        before_outputs: probe_id -> model output text (before modification)
        after_outputs: probe_id -> model output text (after modification)

    Returns:
        Float in [0, 1] representing fraction of changed outputs.
    """
    common_probes = set(before_outputs.keys()) & set(after_outputs.keys())
    if not common_probes:
        return 0.0

    changed = 0
    for probe_id in common_probes:
        if before_outputs[probe_id] != after_outputs[probe_id]:
            changed += 1

    return changed / len(common_probes)


def measure_behavioral_change_numeric(before_scores: Dict[str, float],
                                       after_scores: Dict[str, float]) -> float:
    """Measure behavioral change from numeric scores.

    Args:
        before_scores: probe_id -> score (before modification)
        after_scores: probe_id -> score (after modification)

    Returns:
        Mean absolute change in scores.
    """
    common = set(before_scores.keys()) & set(after_scores.keys())
    if not common:
        return 0.0

    changes = [abs(after_scores[k] - before_scores[k]) for k in common]
    return float(np.mean(changes))


def measure_behavioral_change_from_magnitude(magnitude: float) -> float:
    """Convert a pre-computed behavioral change magnitude to [0,1] range.

    Useful when behavioral change is already computed externally.
    """
    return min(1.0, max(0.0, magnitude))
