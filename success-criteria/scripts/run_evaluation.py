#!/usr/bin/env python3
"""Run the full evaluation pipeline."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evidence.phase_collector import PhaseEvidenceCollector
from src.evaluation.evaluator import CriteriaEvaluator
from src.verdict.verdict import SuccessVerdict
from src.verdict.recommendations import RecommendationGenerator


def main() -> None:
    print("=" * 60)
    print("Month-18 GO/NO-GO Evaluation")
    print("=" * 60)

    # Collect evidence
    print("\n[1/4] Collecting evidence...")
    collector = PhaseEvidenceCollector()
    evidence = collector.collect_all()

    # Evaluate criteria
    print("[2/4] Evaluating criteria...")
    evaluator = CriteriaEvaluator()
    results = evaluator.evaluate_all(evidence)

    # Determine verdict
    print("[3/4] Determining verdict...")
    verdict_engine = SuccessVerdict()
    verdict = verdict_engine.evaluate(results)

    # Print results
    print("\n" + "-" * 60)
    print(verdict.summary())
    print("-" * 60)

    for i, result in enumerate(verdict.criteria_results, 1):
        print(f"\n  {i}. {result.summary()}")

    # Recommendations
    print("\n[4/4] Generating recommendations...")
    recommender = RecommendationGenerator()
    recommendations = recommender.generate(verdict)
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  - {rec}")

    print("\n" + "=" * 60)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
