#!/usr/bin/env python3
"""Generate the comprehensive analysis report."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bdm.calibration import BDMCalibrator
from src.bdm.scorer import BDMScorer
from src.library.store import RuleStore
from src.library.evolution import LibraryEvolver
from src.analysis.escape_analysis import CollapseEscapeAnalyzer
from src.analysis.report import generate_report


def main():
    print("=== Generating Report ===\n")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. BDM Calibration
    print("Running BDM calibration...")
    scorer = BDMScorer()
    calibrator = BDMCalibrator(scorer)
    calibration_report = calibrator.run_calibration()
    print(f"  Ordering correct: {calibration_report.ordering_correct}")

    # 2. Library metrics
    store_path = os.path.join(project_root, "data", "libraries", "synthesis_rules.json")
    library_metrics = None
    metrics_history = []

    if os.path.exists(store_path):
        store = RuleStore(store_path)
        evolver = LibraryEvolver(store=store)
        library_metrics = evolver.measure_library_quality()
        print(f"  Library size: {library_metrics.total_rules}")

    # 3. Escape analysis
    escape_results = None
    if metrics_history:
        analyzer = CollapseEscapeAnalyzer()
        escape_results = analyzer.analyze(metrics_history)

    # 4. Generate report
    report_path = os.path.join(project_root, "data", "reports", "report.md")
    report = generate_report(
        calibration_report=calibration_report,
        library_metrics=library_metrics,
        escape_results=escape_results,
        output_path=report_path,
    )

    print(f"\nReport saved to: {report_path}")
    print(f"Report length: {len(report)} characters")


if __name__ == "__main__":
    main()
