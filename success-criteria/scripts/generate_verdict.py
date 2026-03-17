#!/usr/bin/env python3
"""Generate the final verdict and save to disk."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evidence.phase_collector import PhaseEvidenceCollector
from src.evaluation.evaluator import CriteriaEvaluator
from src.verdict.verdict import SuccessVerdict


def main() -> None:
    print("Generating verdict...")

    # Collect and evaluate
    collector = PhaseEvidenceCollector()
    evidence = collector.collect_all()

    evaluator = CriteriaEvaluator()
    results = evaluator.evaluate_all(evidence)

    verdict_engine = SuccessVerdict()
    verdict = verdict_engine.evaluate(results)

    print(f"\n{verdict.summary()}")

    # Save verdict
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "verdicts"
    )
    os.makedirs(output_dir, exist_ok=True)

    verdict_data = {
        "category": verdict.category.value,
        "n_passed": verdict.n_passed,
        "n_total": verdict.n_total,
        "overall_confidence": verdict.overall_confidence,
        "rationale": verdict.rationale,
        "criteria": [
            {
                "name": r.criterion_name,
                "passed": r.passed,
                "confidence": r.confidence,
                "margin": r.margin,
            }
            for r in verdict.criteria_results
        ],
    }

    output_path = os.path.join(output_dir, "verdict.json")
    with open(output_path, "w") as f:
        json.dump(verdict_data, f, indent=2)

    print(f"Verdict saved to {output_path}")


if __name__ == "__main__":
    main()
