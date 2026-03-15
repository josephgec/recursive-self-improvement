#!/usr/bin/env python3
"""Evaluate the current rule library quality."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.library.store import RuleStore
from src.library.evolution import LibraryEvolver
from src.library.index import RuleIndex


def main():
    print("=== Library Evaluation ===\n")

    store_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "libraries", "synthesis_rules.json"
    )

    if not os.path.exists(store_path):
        print(f"No library found at {store_path}")
        print("Run 'make run-synthesis' first to create a library.")
        return

    store = RuleStore(store_path)
    evolver = LibraryEvolver(store=store)
    index = RuleIndex(store=store)

    # Metrics
    metrics = evolver.measure_library_quality()
    print(f"Total rules: {metrics.total_rules}")
    print(f"Unique domains: {metrics.unique_domains}")
    print(f"Average accuracy: {metrics.avg_accuracy:.2%}")
    print(f"Average BDM score: {metrics.avg_bdm_score:.2f}")
    print(f"Quality score: {metrics.quality_score:.4f}")

    # List rules by domain
    all_rules = store.list_all()
    domains = set(r.domain for r in all_rules)

    for domain in sorted(domains):
        domain_rules = store.list_by_domain(domain)
        print(f"\n--- Domain: {domain} ({len(domain_rules)} rules) ---")
        for rule in domain_rules[:5]:
            print(f"  {rule.rule_id}: {rule.description[:60]} (acc={rule.accuracy:.1%})")

    # Test retrieval
    queries = ["arithmetic multiplication", "string processing", "list operations"]
    print("\n--- Retrieval Test ---")
    for query in queries:
        results = index.retrieve(query, k=3)
        print(f"\n  Query: '{query}'")
        for r in results:
            print(f"    -> {r.rule_id}: {r.description[:50]}")


if __name__ == "__main__":
    main()
