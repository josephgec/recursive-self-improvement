#!/usr/bin/env python3
"""Package all artifacts for reproducibility."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evidence.phase_collector import PhaseEvidenceCollector
from src.evaluation.evaluator import CriteriaEvaluator
from src.verdict.verdict import SuccessVerdict
from src.reporting.reproducibility import ReproducibilityPackager


def main() -> None:
    print("Packaging reproducibility bundle...")

    # Collect evidence
    collector = PhaseEvidenceCollector()
    evidence = collector.collect_all()

    # Evaluate
    evaluator = CriteriaEvaluator()
    results = evaluator.evaluate_all(evidence)

    # Verdict
    verdict_engine = SuccessVerdict()
    verdict = verdict_engine.evaluate(results)

    # Package
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "reports"
    )
    packager = ReproducibilityPackager()
    artifacts = packager.package(evidence, verdict, output_dir)

    print("\nPackaged artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")

    print(f"\nReproducibility package complete in {output_dir}")


if __name__ == "__main__":
    main()
