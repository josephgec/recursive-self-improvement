#!/usr/bin/env python3
"""Run ablation study."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import create_mock_thinking_llm, create_mock_output_llm
from src.operators.thinking_initializer import ThinkingInitializer
from src.operators.thinking_mutator import ThinkingMutator
from src.operators.thinking_crossover import ThinkingCrossover
from src.operators.non_thinking import NonThinkingInitializer, NonThinkingMutator, SimpleCrossover
from src.evaluation.evaluator import UnifiedEvaluator
from src.evaluation.financial_math import FinancialMathBenchmark
from src.evaluation.answer_checker import FinancialAnswerChecker
from src.comparison.ablation import AblationStudy
from src.ga.engine import GAEngine


def main():
    thinking_llm = create_mock_thinking_llm()
    output_llm = create_mock_output_llm()
    checker = FinancialAnswerChecker(tolerance=0.05)

    def engine_factory(**kwargs):
        use_thinking = kwargs.get("use_thinking", True)
        pop_size = kwargs.get("population_size", 4)
        crossover_rate = kwargs.get("crossover_rate", 0.4)
        elitism_count = kwargs.get("elitism_count", 1)

        if use_thinking and not kwargs.get("random_mutation", False):
            init = ThinkingInitializer(thinking_llm)
            mut = ThinkingMutator(thinking_llm)
            cross = ThinkingCrossover(thinking_llm)
        else:
            init = NonThinkingInitializer()
            mut = NonThinkingMutator()
            cross = SimpleCrossover()

        evaluator = UnifiedEvaluator(output_llm, checker, use_thinking=use_thinking)

        return GAEngine(
            initializer=init,
            mutator=mut,
            crossover_op=cross if crossover_rate > 0 else None,
            evaluator=evaluator,
            population_size=pop_size,
            num_generations=3,
            mutation_rate=0.3,
            crossover_rate=crossover_rate,
            elitism_count=elitism_count,
            tournament_size=2,
        )

    study = AblationStudy(
        engine_factory=engine_factory,
        domain_desc="Financial mathematics",
        example_tasks=["Calculate compound interest"],
    )

    bench = FinancialMathBenchmark(seed=42)
    tasks = bench.generate_tasks(n_per_category=1)
    eval_tasks = bench.to_eval_tasks(tasks)

    print("Running ablation study...")
    result = study.run(
        eval_tasks=eval_tasks[:7],
        repetitions=2,
        conditions=["full_thinking", "no_thinking", "random_mutation"],
    )

    print(result.summary)


if __name__ == "__main__":
    main()
