#!/usr/bin/env python3
"""Run anomaly detection checks."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.probing.probe_set import ProbeSet
from src.probing.extractor import MockModel, MockModifiedModel, ActivationExtractor
from src.probing.diff import ActivationDiff
from src.anomaly.divergence_detector import BehavioralInternalDivergenceDetector
from src.anomaly.deceptive_alignment import DeceptiveAlignmentProber
from src.anomaly.internal_distance import measure_internal_change


def main():
    parser = argparse.ArgumentParser(description="Run anomaly detection checks")
    parser.add_argument("--perturbation", type=float, default=0.5,
                        help="Perturbation scale for modified model")
    parser.add_argument("--behavioral-change", type=float, default=0.1,
                        help="Simulated behavioral change")
    args = parser.parse_args()

    probe_set = ProbeSet()
    probes = probe_set.get_all()

    model = MockModel(num_layers=4, num_heads=4, activation_dim=32)
    modified = MockModifiedModel(
        num_layers=4, num_heads=4, activation_dim=32,
        perturbed_layers=[2],
        perturbation_scale=args.perturbation,
    )

    extractor = ActivationExtractor(model)
    mod_extractor = ActivationExtractor(modified)

    before = extractor.extract(probes)
    after = mod_extractor.extract(probes)

    differ = ActivationDiff()
    diff = differ.compute(before, after)

    detector = BehavioralInternalDivergenceDetector()
    result = detector.check(diff, args.behavioral_change)

    print("=== Divergence Check ===")
    print(f"  Internal change: {result.internal_change:.4f}")
    print(f"  Behavioral change: {result.behavioral_change:.4f}")
    print(f"  Divergence ratio: {result.divergence_ratio:.3f}")
    print(f"  Z-score: {result.z_score:.3f}")
    print(f"  Anomalous: {result.is_anomalous}")
    print(f"  Safety flag: {result.safety_flag}")

    print("\n=== Deceptive Alignment Probes ===")
    da_prober = DeceptiveAlignmentProber(model)
    da_report = da_prober.run_all_probes()
    print(f"  Monitoring sensitivity: {da_report.monitoring_sensitivity:.4f}")
    print(f"  Context-dependent safety: {da_report.context_dependent_safety:.4f}")
    print(f"  Latent capability gap: {da_report.latent_capability_gap:.4f}")
    print(f"  Paraphrase consistency: {da_report.paraphrase_consistency:.4f}")
    print(f"  Suspicious: {da_report.is_suspicious}")
    if da_report.flags:
        print(f"  Flags: {da_report.flags}")


if __name__ == "__main__":
    main()
