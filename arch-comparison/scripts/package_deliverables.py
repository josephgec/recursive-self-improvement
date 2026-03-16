#!/usr/bin/env python3
"""Package deliverables for Phase 1a."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deliverables.symcode_summary import package_symcode_results
from src.deliverables.bdm_summary import package_bdm_results
from src.deliverables.final_report import generate_phase1a_report


def main() -> None:
    results_dir = "data/output"
    os.makedirs(results_dir, exist_ok=True)

    print("Packaging SymCode results...")
    symcode = package_symcode_results(results_dir)
    with open(os.path.join(results_dir, "symcode_summary.json"), "w") as f:
        json.dump(symcode, f, indent=2)
    print(f"  Saved to {results_dir}/symcode_summary.json")

    print("Packaging BDM results...")
    bdm = package_bdm_results(results_dir)
    with open(os.path.join(results_dir, "bdm_summary.json"), "w") as f:
        json.dump(bdm, f, indent=2)
    print(f"  Saved to {results_dir}/bdm_summary.json")

    print("Generating Phase 1a report...")
    report = generate_phase1a_report(results_dir)
    os.makedirs("reports", exist_ok=True)
    with open("reports/phase1a_report.md", "w") as f:
        f.write(report)
    print("  Saved to reports/phase1a_report.md")

    print("\nDone!")


if __name__ == "__main__":
    main()
