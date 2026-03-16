#!/usr/bin/env python3
"""Run GDI monitoring loop (demonstration)."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.composite.gdi import GoalDriftIndex
from src.monitoring.time_series import GDITimeSeries
from src.monitoring.anomaly_detector import AnomalyDetector
from src.alerting.alert_manager import AlertManager
from src.alerting.channels import LogChannel


def main():
    """Run a monitoring demonstration."""
    fixtures_dir = os.path.join(
        os.path.dirname(__file__), "..", "tests", "fixtures"
    )

    with open(os.path.join(fixtures_dir, "reference_outputs.json")) as f:
        ref_data = json.load(f)
    with open(os.path.join(fixtures_dir, "drifted_outputs.json")) as f:
        drift_data = json.load(f)

    reference = ref_data["outputs"]
    drifted = drift_data["outputs"]

    gdi = GoalDriftIndex()
    ts = GDITimeSeries()
    detector = AnomalyDetector()
    channel = LogChannel()
    alert_mgr = AlertManager(channels=[channel])

    # Simulate monitoring iterations
    for i in range(5):
        if i < 3:
            result = gdi.compute(reference, reference)
        else:
            result = gdi.compute(drifted, reference)

        ts.record(result, iteration=i)
        alert = alert_mgr.process(result, i)

        level = result.alert_level
        score = result.composite_score
        print(f"Iteration {i}: score={score:.3f}, level={level}")
        if alert:
            print(f"  ALERT: {alert.message}")

    # Check for anomalies
    scores = ts.get_scores()
    anomalies = detector.detect(scores)
    if anomalies:
        print(f"\nDetected {len(anomalies)} anomalies:")
        for a in anomalies:
            print(f"  Index {a.index}: z={a.z_score:.2f}, direction={a.direction}")


if __name__ == "__main__":
    main()
