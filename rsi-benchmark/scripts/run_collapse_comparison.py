#!/usr/bin/env python3
"""Run collapse comparison analysis."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.collapse.baseline_loader import CollapseBaselineLoader
from src.collapse.divergence_analyzer import DivergenceAnalyzer
from src.collapse.sustainability import SustainabilityAnalyzer


def main():
    # Build RSI curve (simulated)
    rsi_curve = [(i, 0.60 + 0.015 * i) for i in range(15)]

    # Load collapse baseline
    loader = CollapseBaselineLoader()
    collapse = loader.load("standard_decay")

    # Divergence analysis
    analyzer = DivergenceAnalyzer()
    result = analyzer.compute_divergence(rsi_curve, collapse.accuracy)

    print("Collapse Comparison:")
    print(f"  Mean Divergence: {result.mean_divergence:.4f}")
    print(f"  Max Divergence: {result.max_divergence:.4f}")
    print(f"  Trend: {result.divergence_trend}")
    print(f"  Prevention Score: {result.collapse_prevention_score:.4f}")

    # Sustainability
    rsi_accuracies = [v for _, v in rsi_curve]
    sus_analyzer = SustainabilityAnalyzer()
    report = sus_analyzer.analyze(rsi_accuracies)

    print(f"\nSustainability:")
    print(f"  Longest Streak: {report.longest_improvement_streak}")
    print(f"  Monotonicity: {report.monotonicity_score:.4f}")
    print(f"  Overall Score: {report.overall_sustainability_score:.4f}")


if __name__ == "__main__":
    main()
