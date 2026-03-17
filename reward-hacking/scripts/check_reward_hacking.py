#!/usr/bin/env python3
"""Check for reward hacking demo."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.detection.composite_detector import (
    CompositeRewardHackingDetector,
    TrainingState,
)


def main():
    rng = np.random.RandomState(42)
    detector = CompositeRewardHackingDetector()

    print("Reward Hacking Detection Demo")
    print("=" * 50)

    # Simulate a healthy state
    healthy_state = TrainingState(
        rewards=[0.5 + rng.randn() * 0.1 for _ in range(30)],
        accuracies=[0.7 + rng.randn() * 0.05 for _ in range(30)],
        output_lengths=[50 + int(rng.randn() * 5) for _ in range(10)],
        baseline_lengths=[45 + int(rng.randn() * 5) for _ in range(10)],
        outputs=[list(rng.randint(0, 100, 50)) for _ in range(10)],
        output_strings=["Normal output text here"] * 10,
    )

    report = detector.check(healthy_state)
    print(f"\nHealthy State:")
    print(f"  Hacking Detected: {report.is_hacking_detected}")
    print(f"  Severity: {report.severity:.4f}")
    print(f"  Recommendation: {report.recommendation}")


if __name__ == "__main__":
    main()
