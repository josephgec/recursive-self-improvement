#!/usr/bin/env python3
"""Generate a full interpretability report."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.probing.probe_set import ProbeSet
from src.probing.extractor import MockModel, MockModifiedModel, ActivationExtractor
from src.probing.diff import ActivationDiff
from src.attention.head_extractor import HeadExtractor
from src.attention.specialization import HeadSpecializationTracker
from src.anomaly.divergence_detector import BehavioralInternalDivergenceDetector
from src.anomaly.deceptive_alignment import DeceptiveAlignmentProber
from src.monitoring.alert_rules import InterpretabilityAlertRules
from src.analysis.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="Generate interpretability report")
    parser.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")
    parser.add_argument("--perturbation", type=float, default=0.5, help="Perturbation scale")
    args = parser.parse_args()

    probe_set = ProbeSet()
    probes = probe_set.get_all()

    model = MockModel(num_layers=4, num_heads=4, activation_dim=32)
    modified = MockModifiedModel(
        num_layers=4, num_heads=4, activation_dim=32,
        perturbed_layers=[2],
        perturbation_scale=args.perturbation,
    )

    # Extract & diff
    ext_before = ActivationExtractor(model)
    ext_after = ActivationExtractor(modified)
    before = ext_before.extract(probes)
    after = ext_after.extract(probes)
    diff = ActivationDiff().compute(before, after)

    # Divergence
    detector = BehavioralInternalDivergenceDetector()
    div_result = detector.check(diff, behavioral_change=0.1)

    # Heads
    head_ext = HeadExtractor(model)
    head_stats = head_ext.extract_aggregate_stats(probes[:5])
    tracker = HeadSpecializationTracker()
    head_result = tracker.track(head_stats)

    # Deceptive alignment
    da = DeceptiveAlignmentProber(model)
    da_report = da.run_all_probes()

    # Alerts
    alert_rules = InterpretabilityAlertRules()
    data = {
        "divergence": div_result.to_dict(),
        "diff_summary": diff.to_dict(),
        "head_tracking": head_result.to_dict(),
        "deceptive_alignment": da_report.to_dict(),
    }
    alerts = alert_rules.evaluate(data)

    # Report
    report = generate_report(
        iteration=1,
        divergence_data=div_result.to_dict(),
        diff_data=diff.to_dict(),
        head_tracking_data=head_result.to_dict(),
        deceptive_alignment_data=da_report.to_dict(),
        alerts=[a.to_dict() for a in alerts],
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
