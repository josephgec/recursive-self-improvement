#!/usr/bin/env python3
"""Track attention head specialization."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.probing.probe_set import ProbeSet
from src.probing.extractor import MockModel, MockModifiedModel, ActivationExtractor
from src.attention.head_extractor import HeadExtractor
from src.attention.specialization import HeadSpecializationTracker
from src.attention.dead_head_detector import DeadHeadDetector
from src.attention.role_tracker import HeadRoleTracker


def main():
    parser = argparse.ArgumentParser(description="Track attention head specialization")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads")
    args = parser.parse_args()

    probe_set = ProbeSet()
    probes = probe_set.get_all()[:5]

    tracker = HeadSpecializationTracker()
    role_tracker = HeadRoleTracker()
    dead_detector = DeadHeadDetector()

    for i in range(args.iterations):
        if i == 0:
            model = MockModel(num_layers=args.layers, num_heads=args.heads, activation_dim=32)
        else:
            model = MockModifiedModel(
                num_layers=args.layers, num_heads=args.heads, activation_dim=32,
                perturbed_layers=[i % args.layers],
                perturbation_scale=0.3 * i,
            )

        extractor = HeadExtractor(model)
        head_stats = extractor.extract_aggregate_stats(probes)

        result = tracker.track(head_stats)
        role_tracker.track(head_stats)

        dead = dead_detector.detect(head_stats)

        print(f"\n=== Iteration {i} ===")
        print(f"  Dying heads: {result.dying_heads}")
        print(f"  Narrowing heads: {result.narrowing_heads}")
        print(f"  Role changes: {len(result.role_changes)}")
        print(f"  Dead heads: {len(dead)}")
        if result.summary:
            print(f"  Mean entropy: {result.summary.get('mean_entropy', 0):.3f}")


if __name__ == "__main__":
    main()
