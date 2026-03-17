#!/usr/bin/env python3
"""Compare two activation snapshots."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.probing.snapshot import load_snapshot
from src.probing.diff import ActivationDiff


def main():
    parser = argparse.ArgumentParser(description="Compare two activation snapshots")
    parser.add_argument("before", help="Path to before snapshot")
    parser.add_argument("after", help="Path to after snapshot")
    parser.add_argument("--safety-factor", type=float, default=2.0,
                        help="Safety disproportionate factor")
    args = parser.parse_args()

    before = load_snapshot(args.before)
    after = load_snapshot(args.after)

    differ = ActivationDiff(safety_disproportionate_factor=args.safety_factor)
    result = differ.compute(before, after)

    print("=== Activation Diff ===")
    print(f"Overall change magnitude: {result.overall_change_magnitude:.4f}")
    print(f"Most changed layers: {result.most_changed_layers}")
    print(f"Most changed probes: {result.most_changed_probes}")
    print(f"Safety disproportionate: {result.safety_disproportionate}")
    print(f"Safety change ratio: {result.safety_change_ratio:.3f}")

    print("\nPer-layer diffs:")
    for name, ld in sorted(result.layer_diffs.items()):
        print(f"  {name}: mean_shift={ld.mean_shift:.4f}, "
              f"direction_sim={ld.direction_similarity:.4f}")


if __name__ == "__main__":
    main()
