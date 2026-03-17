#!/usr/bin/env python3
"""Verify data integrity of collected evidence."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evidence.phase_collector import PhaseEvidenceCollector
from src.evidence.data_integrity import DataIntegrityVerifier


def main() -> None:
    print("Verifying data integrity...")

    collector = PhaseEvidenceCollector()
    evidence = collector.collect_all()

    verifier = DataIntegrityVerifier()
    report = verifier.verify(evidence)

    print(f"\n{report.summary()}")
    print(f"  Hash chain valid: {report.hash_chain_valid}")
    print(f"  Preregistration valid: {report.preregistration_valid}")
    print(f"  Timestamps valid: {report.timestamp_valid}")

    if report.issues:
        print("\nIssues:")
        for issue in report.issues:
            print(f"  - {issue}")
    else:
        print("\nNo integrity issues found.")


if __name__ == "__main__":
    main()
