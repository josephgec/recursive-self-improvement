#!/usr/bin/env python3
"""Run prompt evolution with mock LLMs."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import create_mock_thinking_llm, create_mock_output_llm
from src.operators.thinking_initializer import ThinkingInitializer
from src.operators.thinking_mutator import ThinkingMutator
from src.operators.thinking_crossover import ThinkingCrossover
from src.evaluation.evaluator import UnifiedEvaluator
from src.evaluation.financial_math import FinancialMathBenchmark
from src.evaluation.answer_checker import FinancialAnswerChecker
from src.ga.engine import GAEngine


def main():
    thinking_llm = create_mock_thinking_llm()
    output_llm = create_mock_output_llm()
    checker = FinancialAnswerChecker(tolerance=0.05)

    initializer = ThinkingInitializer(thinking_llm)
    mutator = ThinkingMutator(thinking_llm)
    crossover = ThinkingCrossover(thinking_llm)
    evaluator = UnifiedEvaluator(output_llm, checker, use_thinking=True)

    engine = GAEngine(
        initializer=initializer,
        mutator=mutator,
        crossover_op=crossover,
        evaluator=evaluator,
        population_size=6,
        num_generations=5,
        mutation_rate=0.3,
        crossover_rate=0.4,
        elitism_count=1,
        tournament_size=2,
    )

    bench = FinancialMathBenchmark(seed=42)
    tasks = bench.generate_tasks(n_per_category=2)
    eval_tasks = bench.to_eval_tasks(tasks)

    print("Starting evolution...")
    result = engine.evolve(
        domain_desc="Financial mathematics",
        example_tasks=["Calculate compound interest", "Find present value"],
        eval_tasks=eval_tasks[:10],
    )

    print(f"Evolution complete!")
    print(f"Generations: {result.generations_run}")
    print(f"Best fitness: {result.best_fitness:.4f}")
    print(f"Stop reason: {result.stopped_reason}")

    if result.best_genome:
        print(f"\nBest prompt:\n{result.best_genome.to_system_prompt()[:500]}")


if __name__ == "__main__":
    main()
