#!/usr/bin/env python3
"""Evaluate a single prompt against the benchmark."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import create_mock_output_llm
from src.genome.prompt_genome import PromptGenome
from src.genome.sections import DEFAULT_SECTIONS
from src.evaluation.evaluator import UnifiedEvaluator
from src.evaluation.financial_math import FinancialMathBenchmark
from src.evaluation.answer_checker import FinancialAnswerChecker


def main():
    output_llm = create_mock_output_llm()
    checker = FinancialAnswerChecker(tolerance=0.05)
    evaluator = UnifiedEvaluator(output_llm, checker, use_thinking=True)

    # Create default genome
    genome = PromptGenome(genome_id="baseline")
    for section_name, content in DEFAULT_SECTIONS.items():
        genome.set_section(section_name, content)

    bench = FinancialMathBenchmark(seed=42)
    tasks = bench.generate_tasks(n_per_category=2)
    eval_tasks = bench.to_eval_tasks(tasks[:10])

    print("Evaluating baseline prompt...")
    details = evaluator.evaluate(genome, eval_tasks)
    print(f"Accuracy: {details.accuracy:.4f}")
    print(f"Reasoning: {details.reasoning_score:.4f}")
    print(f"Consistency: {details.consistency_score:.4f}")
    print(f"Composite: {details.composite_fitness:.4f}")


if __name__ == "__main__":
    main()
