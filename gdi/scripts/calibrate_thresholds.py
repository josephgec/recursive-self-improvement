#!/usr/bin/env python3
"""Calibrate GDI thresholds from collapse data."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.composite.gdi import GoalDriftIndex
from src.calibration.collapse_calibrator import CollapseCalibrator


def main():
    """Calibrate thresholds using fixture data."""
    fixtures_dir = os.path.join(
        os.path.dirname(__file__), "..", "tests", "fixtures"
    )

    with open(os.path.join(fixtures_dir, "reference_outputs.json")) as f:
        ref_data = json.load(f)
    with open(os.path.join(fixtures_dir, "collapsed_outputs.json")) as f:
        col_data = json.load(f)

    reference = ref_data["outputs"]

    # Build collapse trajectory
    collapse_data = [
        {"outputs": reference, "reference": reference, "health": "healthy"},
        {"outputs": col_data["outputs"], "reference": reference, "health": "collapsed"},
    ]

    gdi = GoalDriftIndex()
    calibrator = CollapseCalibrator()
    thresholds = calibrator.calibrate(gdi, collapse_data)

    print(f"Calibrated Thresholds:")
    print(f"  Green max:  {thresholds.green_max:.3f}")
    print(f"  Yellow max: {thresholds.yellow_max:.3f}")
    print(f"  Orange max: {thresholds.orange_max:.3f}")
    print(f"  Red min:    {thresholds.red_min:.3f}")
    print(f"  AUC:        {thresholds.auc:.3f}")


if __name__ == "__main__":
    main()
