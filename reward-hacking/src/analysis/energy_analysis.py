from __future__ import annotations

"""Analysis utilities for energy tracking results."""

import numpy as np

from ..energy.energy_tracker import EnergyMeasurement


def analyze_energy(measurements: list[EnergyMeasurement]) -> dict:
    """Analyze energy tracking results.

    Args:
        measurements: List of energy measurements.

    Returns:
        Dictionary with analysis metrics.
    """
    if not measurements:
        return {"status": "no_data"}

    energies = [m.total_energy for m in measurements]
    energy_arr = np.array(energies)

    # Trend
    if len(energies) >= 2:
        slope = float(np.polyfit(range(len(energies)), energies, 1)[0])
    else:
        slope = 0.0

    # Per-layer analysis
    num_layers = measurements[0].num_layers
    layer_stats = {}
    for layer in range(num_layers):
        layer_energies = [
            m.per_layer_energy[layer]
            for m in measurements
            if layer < len(m.per_layer_energy)
        ]
        if layer_energies:
            layer_stats[f"layer_{layer}"] = {
                "mean": float(np.mean(layer_energies)),
                "std": float(np.std(layer_energies)),
                "trend": float(np.polyfit(range(len(layer_energies)), layer_energies, 1)[0])
                if len(layer_energies) >= 2
                else 0.0,
            }

    # Stability
    cv = float(np.std(energy_arr) / np.mean(energy_arr)) if np.mean(energy_arr) > 0 else 0.0

    return {
        "num_measurements": len(measurements),
        "total_energy": {
            "initial": energies[0],
            "final": energies[-1],
            "mean": float(np.mean(energy_arr)),
            "std": float(np.std(energy_arr)),
            "slope": slope,
            "cv": cv,
        },
        "per_layer": layer_stats,
        "stable": abs(slope) < 0.05 and cv < 0.3,
        "declining": slope < -0.05,
    }
