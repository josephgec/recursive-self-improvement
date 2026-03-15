#!/usr/bin/env python3
"""Run the symbolic synthesis loop on example problems."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synthesis.candidate_generator import IOExample
from src.synthesis.synthesis_loop import SymbolicSynthesisLoop
from src.library.rule import VerifiedRule
from src.library.store import RuleStore
from src.library.evolution import LibraryEvolver


def main():
    print("=== Symbolic Synthesis Loop ===\n")

    # Define example problems
    problems = {
        "double": [
            IOExample(input=1, output=2, domain="math"),
            IOExample(input=2, output=4, domain="math"),
            IOExample(input=3, output=6, domain="math"),
            IOExample(input=5, output=10, domain="math"),
            IOExample(input=10, output=20, domain="math"),
        ],
        "square": [
            IOExample(input=1, output=1, domain="math"),
            IOExample(input=2, output=4, domain="math"),
            IOExample(input=3, output=9, domain="math"),
            IOExample(input=4, output=16, domain="math"),
            IOExample(input=5, output=25, domain="math"),
        ],
        "sum_list": [
            IOExample(input=[1, 2, 3], output=6, domain="math"),
            IOExample(input=[10, 20], output=30, domain="math"),
            IOExample(input=[5], output=5, domain="math"),
            IOExample(input=[1, 1, 1, 1], output=4, domain="math"),
        ],
    }

    # Setup library
    store_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "libraries", "synthesis_rules.json"
    )
    store = RuleStore(store_path)
    evolver = LibraryEvolver(store=store)

    loop = SymbolicSynthesisLoop()

    for name, examples in problems.items():
        print(f"\n--- Problem: {name} ---")
        result = loop.run(examples, max_iterations=3, candidates_per_iteration=5)

        print(f"  Iterations: {result.total_iterations}")
        print(f"  Total candidates: {result.total_candidates}")
        print(f"  Best accuracy: {result.final_best_accuracy:.2%}")

        # Add best rules to library
        for sr in result.best_rules:
            if sr.accuracy >= 0.8:
                verified = VerifiedRule(
                    rule_id=f"{name}_{sr.rule.rule_id}",
                    domain="math",
                    description=f"Rule for {name}: {sr.rule.description}",
                    source_code=sr.rule.source_code,
                    accuracy=sr.accuracy,
                    bdm_score=sr.bdm_complexity,
                    mdl_score=sr.mdl_score,
                    tags=[name, "math", "synthesized"],
                )
                store.add(verified)

    # Evolve library
    metrics = evolver.measure_library_quality()
    print(f"\n=== Library Quality ===")
    print(f"  Rules: {metrics.total_rules}")
    print(f"  Domains: {metrics.unique_domains}")
    print(f"  Avg accuracy: {metrics.avg_accuracy:.2%}")
    print(f"  Quality score: {metrics.quality_score:.4f}")


if __name__ == "__main__":
    main()
