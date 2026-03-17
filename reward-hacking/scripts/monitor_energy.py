#!/usr/bin/env python3
"""Monitor activation energy demo."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.energy.energy_tracker import EnergyTracker
from src.energy.homogenization import HomogenizationDetector
from src.energy.early_warning import EnergyEarlyWarning


def main():
    rng = np.random.RandomState(42)
    tracker = EnergyTracker(num_layers=6)
    detector = HomogenizationDetector()
    warning = EnergyEarlyWarning()

    print("Energy Monitoring Demo")
    print("=" * 50)

    # Simulate declining energy
    for step in range(50):
        scale = max(0.1, 1.0 - 0.015 * step)
        activations = [rng.randn(64) * scale for _ in range(6)]
        measurement = tracker.measure(activations)

        if step == 10:
            tracker.set_baseline()

        if step % 10 == 0:
            print(
                f"Step {step}: energy={measurement.total_energy:.4f}, "
                f"relative={measurement.relative_energy or 'N/A'}"
            )

    # Check homogenization
    result = detector.detect(tracker.measurements)
    print(f"\nHomogenization: {result.is_homogenizing}")
    print(f"Patterns: {result.patterns_detected}")
    print(f"Severity: {result.severity:.4f}")

    # Early warning
    pred = warning.predict(tracker.measurements)
    print(f"\nPredicted energy: {pred.predicted_energy:.4f}")
    print(f"Will decline: {pred.will_decline}")
    print(f"Steps to critical: {pred.estimated_steps_to_critical}")


if __name__ == "__main__":
    main()
