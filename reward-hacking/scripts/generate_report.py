#!/usr/bin/env python3
"""Generate full analysis report demo."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.eppo.config import EPPOConfig
from src.eppo.trainer import EPPOTrainer
from src.bounding.process_reward import ProcessRewardShaper
from src.energy.energy_tracker import EnergyTracker
from src.analysis.report import generate_full_report


def main():
    rng = np.random.RandomState(42)

    print("Generating Full Report")
    print("=" * 50)

    # Run EPPO training
    trainer = EPPOTrainer(EPPOConfig(epochs=3))
    for _ in range(3):
        trainer.train_epoch(num_steps=5)

    # Run bounding
    shaper = ProcessRewardShaper()
    for _ in range(20):
        shaper.shape(rng.randn() * 3)

    # Run energy tracking
    tracker = EnergyTracker(num_layers=4)
    for step in range(20):
        scale = max(0.1, 1.0 - 0.01 * step)
        activations = [rng.randn(32) * scale for _ in range(4)]
        tracker.measure(activations)

    # Generate report
    report = generate_full_report(
        epoch_results=trainer.epoch_results,
        shaped_rewards=shaper.history,
        energy_measurements=tracker.measurements,
    )

    print(report)

    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "reports", "analysis_report.md")
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
