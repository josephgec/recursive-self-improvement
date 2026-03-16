#!/usr/bin/env python3
"""Generate a comprehensive ablation study report."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.suites import NeurosymbolicAblation, GodelAgentAblation, SOARAblation, RLMAblation
from src.execution.runner import AblationRunner
from src.analysis.statistical_tests import PublicationStatistics
from src.analysis.interaction_tests import CrossSuiteInteractionAnalyzer
from src.publication.narrative import NarrativeGenerator


def main():
    repetitions = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    suites = [
        NeurosymbolicAblation(),
        GodelAgentAblation(),
        SOARAblation(),
        RLMAblation(),
    ]

    runner = AblationRunner()
    stats = PublicationStatistics()
    narr = NarrativeGenerator()
    interaction_analyzer = CrossSuiteInteractionAnalyzer()

    all_results = {}
    for suite in suites:
        result = runner.run_suite(suite, repetitions=repetitions, seed=seed)
        all_results[suite.get_paper_name()] = result

        print(f"\n=== {suite.get_paper_name()} ===")
        analyses = suite.analyze(result)
        for key, pw in analyses.items():
            print(f"  {pw}")

    # Cross-suite interactions
    print("\n=== Cross-Suite Interactions ===")
    reports = interaction_analyzer.test_cross_paradigm_interactions(all_results)
    for report in reports:
        print(f"  {report.summary}")

    # Summary
    print("\n=== Summary ===")
    summary = narr.generate_summary_paragraph(all_results)
    print(summary)


if __name__ == "__main__":
    main()
