#!/usr/bin/env python3
"""Run head-to-head comparison between thinking and non-thinking operators."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import create_mock_thinking_llm, create_mock_output_llm
from src.evaluation.financial_math import FinancialMathBenchmark
from src.evaluation.answer_checker import FinancialAnswerChecker
from src.comparison.head_to_head import HeadToHeadAnalyzer
from src.comparison.ablation import AblationResult, ConditionResult
from src.comparison.statistical_tests import StatisticalComparator


def main():
    bench = FinancialMathBenchmark(seed=42)
    tasks = bench.generate_tasks(n_per_category=2)
    eval_tasks = bench.to_eval_tasks(tasks[:10])

    # Simulate two conditions with known scores
    result = AblationResult()

    result.conditions["thinking"] = ConditionResult(
        condition_name="thinking",
        fitness_scores=[0.65, 0.70, 0.68, 0.72, 0.67],
    )
    result.conditions["thinking"].compute_stats()

    result.conditions["non_thinking"] = ConditionResult(
        condition_name="non_thinking",
        fitness_scores=[0.50, 0.55, 0.48, 0.52, 0.51],
    )
    result.conditions["non_thinking"].compute_stats()

    result.generate_summary()

    analyzer = HeadToHeadAnalyzer()
    pairwise = analyzer.compare_conditions(result)

    print(analyzer.plot_comparison(result))
    print()
    print(analyzer.generate_ranking_table(result, pairwise))


if __name__ == "__main__":
    main()
