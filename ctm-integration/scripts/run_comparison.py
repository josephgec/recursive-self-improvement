#!/usr/bin/env python3
"""Compare augmented vs standard prompts."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.library.rule import VerifiedRule
from src.library.store import RuleStore
from src.integration.augmented_prompt import AugmentedPromptBuilder
from src.integration.comparison import AugmentationComparison


def main():
    print("=== Augmented vs Standard Prompt Comparison ===\n")

    # Build a sample library
    store = RuleStore()

    sample_rules = [
        VerifiedRule(
            rule_id="math_double",
            domain="math",
            description="Doubles a number",
            source_code="def rule(x):\n    return x * 2\n",
            accuracy=1.0,
            bdm_score=15.0,
            tags=["math", "arithmetic", "double"],
        ),
        VerifiedRule(
            rule_id="math_square",
            domain="math",
            description="Squares a number",
            source_code="def rule(x):\n    return x ** 2\n",
            accuracy=1.0,
            bdm_score=16.0,
            tags=["math", "arithmetic", "square"],
        ),
        VerifiedRule(
            rule_id="string_reverse",
            domain="string",
            description="Reverses a string",
            source_code="def rule(x):\n    return x[::-1]\n",
            accuracy=1.0,
            bdm_score=18.0,
            tags=["string", "reverse"],
        ),
    ]

    for rule in sample_rules:
        store.add(rule)

    # Run comparison
    tasks = [
        "Write a function that triples a number",
        "Write a function that computes the cube of a number",
        "Write a function that reverses a list",
        "Write a function that sorts an array",
    ]

    comparison = AugmentationComparison(store=store)
    results = comparison.run_comparison(tasks)
    analysis = comparison.analyze(results)

    print(f"Tasks analyzed: {analysis.total_tasks}")
    print(f"Tasks with rules: {analysis.tasks_with_rules}")
    print(f"Augmentation rate: {analysis.augmentation_rate:.1%}")
    print(f"Average rules included: {analysis.avg_rules_included:.1f}")
    print(f"Average length ratio: {analysis.avg_length_ratio:.1f}x")

    for result in results:
        print(f"\n--- Task: {result.task[:50]}... ---")
        print(f"  Rules included: {result.rules_included}")
        print(f"  Standard length: {result.standard_length}")
        print(f"  Augmented length: {result.augmented_length}")


if __name__ == "__main__":
    main()
