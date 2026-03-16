#!/usr/bin/env python3
"""Generate a GDI analysis report."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.composite.gdi import GoalDriftIndex
from src.analysis.report import generate_report


def main():
    """Generate a report from fixture data."""
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

    # Build history
    history = []
    history.append(gdi.compute(reference, reference))
    history.append(gdi.compute(drift_data["outputs"], reference))
    history.append(gdi.compute(collapse_data["outputs"], reference))

    report = generate_report(
        gdi_history=history,
        accuracy_history=[0.95, 0.85, 0.40],
    )

    report_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "reports", "gdi_report.json"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report generated: {report_path}")
    print(f"Latest GDI: {report['latest']['composite_score']:.3f}")
    print(f"Alert level: {report['latest']['alert_level']}")
    print(f"Drift type: {report['characterization']['drift_type']}")


if __name__ == "__main__":
    main()
