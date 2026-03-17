#!/usr/bin/env python3
"""Package safety deliverables demo."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.deliverables.phase_gate import PhaseGateSafetyPackage


def main():
    print("Safety Package Demo")
    print("=" * 50)

    gate = PhaseGateSafetyPackage()

    # All green
    histories = {
        "gdi": {"governance_review": True, "deployment_checks_passed": True, "impact_assessed": True},
        "constraint": {"reward_bounded": True, "entropy_above_min": True, "energy_stable": True},
        "interp": {"energy_interpretable": True, "homogenization_checked": True, "activations_tracked": True},
        "reward": {"no_divergence": True, "no_shortcuts": True, "no_gaming": True},
    }

    pkg = gate.package("phase_1", (0, 100), histories)
    validation = gate.validate(pkg)

    print(f"\nPhase 1 Package:")
    print(f"  All Green: {pkg.all_green}")
    print(f"  Valid: {validation.is_valid}")
    print(f"  Summary: {pkg.summary}")


if __name__ == "__main__":
    main()
