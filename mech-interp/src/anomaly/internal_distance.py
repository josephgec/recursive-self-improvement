"""Measure internal representation change magnitude."""

from typing import Optional
import numpy as np

from src.probing.diff import ActivationDiffResult


def measure_internal_change(diff_result: ActivationDiffResult) -> float:
    """Measure the magnitude of internal representation change.

    Combines overall change magnitude with direction similarity shifts.

    Args:
        diff_result: Result from ActivationDiff.compute()

    Returns:
        Float representing internal change magnitude.
    """
    if not diff_result.layer_diffs:
        return 0.0

    # Use overall change magnitude
    magnitude = diff_result.overall_change_magnitude

    # Adjust by direction similarity (lower similarity = more change)
    avg_direction_sim = np.mean([
        ld.direction_similarity for ld in diff_result.layer_diffs.values()
    ])
    direction_change = 1.0 - avg_direction_sim

    # Combined metric
    combined = magnitude * (1.0 + direction_change)
    return float(combined)


def measure_internal_change_per_layer(diff_result: ActivationDiffResult) -> dict:
    """Return per-layer internal change magnitudes."""
    result = {}
    for layer_name, layer_diff in diff_result.layer_diffs.items():
        probe_changes = list(layer_diff.per_probe_changes.values())
        if probe_changes:
            result[layer_name] = {
                "mean_change": float(np.mean(probe_changes)),
                "max_change": float(np.max(probe_changes)),
                "direction_similarity": layer_diff.direction_similarity,
            }
    return result


def measure_safety_internal_change(diff_result: ActivationDiffResult) -> float:
    """Measure internal change specifically for safety-related probes."""
    if not diff_result.layer_diffs:
        return 0.0

    safety_changes = []
    for layer_diff in diff_result.layer_diffs.values():
        for probe_id, change in layer_diff.per_probe_changes.items():
            if "safety" in probe_id:
                safety_changes.append(change)

    if not safety_changes:
        return 0.0
    return float(np.mean(safety_changes))
