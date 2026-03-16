#!/usr/bin/env python3
"""Compute GDI score for given outputs."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.composite.gdi import GoalDriftIndex


def main():
    """Compute GDI between reference and drifted outputs."""
    fixtures_dir = os.path.join(
        os.path.dirname(__file__), "..", "tests", "fixtures"
    )

    with open(os.path.join(fixtures_dir, "reference_outputs.json")) as f:
        ref_data = json.load(f)
    with open(os.path.join(fixtures_dir, "drifted_outputs.json")) as f:
        drift_data = json.load(f)
    with open(os.path.join(fixtures_dir, "collapsed_outputs.json")) as f:
        collapse_data = json.load(f)

    reference = ref_data["outputs"]

    gdi = GoalDriftIndex()

    # Same outputs
    result = gdi.compute(reference, reference)
    print(f"Same outputs:      GDI={result.composite_score:.3f} [{result.alert_level}]")

    # Drifted outputs
    result = gdi.compute(drift_data["outputs"], reference)
    print(f"Drifted outputs:   GDI={result.composite_score:.3f} [{result.alert_level}]")

    # Collapsed outputs
    result = gdi.compute(collapse_data["outputs"], reference)
    print(f"Collapsed outputs: GDI={result.composite_score:.3f} [{result.alert_level}]")

    print(f"\nSignal breakdown (collapsed):")
    print(f"  Semantic:       {result.semantic_score:.3f}")
    print(f"  Lexical:        {result.lexical_score:.3f}")
    print(f"  Structural:     {result.structural_score:.3f}")
    print(f"  Distributional: {result.distributional_score:.3f}")


if __name__ == "__main__":
    main()
