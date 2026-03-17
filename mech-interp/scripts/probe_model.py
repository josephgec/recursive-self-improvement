#!/usr/bin/env python3
"""Probe a model and save activation snapshot."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.probing.probe_set import ProbeSet
from src.probing.extractor import MockModel, ActivationExtractor
from src.probing.snapshot import save_snapshot


def main():
    parser = argparse.ArgumentParser(description="Probe model and save activations")
    parser.add_argument("--output", "-o", default="data/snapshots/snapshot.json",
                        help="Output snapshot path")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--dim", type=int, default=32, help="Activation dimension")
    args = parser.parse_args()

    model = MockModel(
        num_layers=args.layers,
        num_heads=args.heads,
        activation_dim=args.dim,
    )

    probe_set = ProbeSet()
    extractor = ActivationExtractor(model)
    snapshot = extractor.extract(probe_set.get_all())

    save_snapshot(snapshot, args.output)
    print(f"Saved snapshot to {args.output}")
    print(f"  Probes: {len(snapshot.get_probe_ids())}")
    print(f"  Layers: {len(snapshot.get_all_layer_names())}")


if __name__ == "__main__":
    main()
