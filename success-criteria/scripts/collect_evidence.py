#!/usr/bin/env python3
"""Collect all evidence from phases and safety subsystems."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evidence.phase_collector import PhaseEvidenceCollector
from src.evidence.safety_collector import SafetyEvidenceCollector


def main() -> None:
    print("Collecting evidence...")

    # Phase evidence
    phase_collector = PhaseEvidenceCollector()
    evidence = phase_collector.collect_all()

    print(f"  Improvement curve: {evidence.get_improvement_curve()}")
    print(f"  Collapse curve: {evidence.get_collapse_curve()}")
    print(f"  Publications: {len(evidence.publications)}")
    print(f"  GDI readings: {len(evidence.get_gdi_readings())}")

    # Safety evidence
    safety_collector = SafetyEvidenceCollector()
    safety_data = safety_collector.collect()
    for sid, sdata in safety_data.items():
        print(f"  {sid}: {sdata['subsystem_name']} - {sdata['status']}")

    # Save evidence snapshot
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "evidence"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "collected_evidence.json")
    with open(output_path, "w") as f:
        json.dump({
            "improvement_curve": evidence.get_improvement_curve(),
            "collapse_curve": evidence.get_collapse_curve(),
            "publications": evidence.publications,
            "gdi_readings": evidence.get_gdi_readings(),
        }, f, indent=2)

    print(f"\nEvidence saved to {output_path}")


if __name__ == "__main__":
    main()
