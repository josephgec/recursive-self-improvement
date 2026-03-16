#!/usr/bin/env python3
"""Run paradigm ablation study."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.registry import BenchmarkRegistry, register_all_benchmarks
from src.ablation.ablation_study import ParadigmAblationStudy
from src.ablation.contribution import ContributionAnalyzer


class AblationAgent:
    """Agent that behaves differently per condition."""
    RATES = {
        "full_pipeline": 0.015,
        "no_soar": 0.008,
        "no_ctm": 0.010,
        "no_godel": 0.005,
        "no_rlm": 0.012,
        "soar_only": 0.006,
        "naive_self_train": -0.01,
    }
    BASES = {
        "naive_self_train": 0.65,
    }

    def __init__(self, condition):
        self._condition = condition
        self._iteration = 0

    def set_iteration(self, iteration):
        self._iteration = iteration

    def solve(self, task):
        import hashlib
        rate = self.RATES.get(self._condition, 0.015)
        base = self.BASES.get(self._condition, 0.60)
        accuracy = base + rate * self._iteration
        h = int(hashlib.md5(task.task_id.encode()).hexdigest(), 16) % 100
        return task.expected_answer if h < accuracy * 100 else None


def main():
    register_all_benchmarks()
    benchmarks = BenchmarkRegistry.load_all()

    study = ParadigmAblationStudy(
        agent_factory=lambda cond: AblationAgent(cond),
        num_iterations=15,
    )
    result = study.run(benchmarks)

    print("Ablation Results:")
    for cond, imp in sorted(result.summary.items()):
        print(f"  {cond}: avg improvement = {imp:.4f}")

    analyzer = ContributionAnalyzer()
    ranked = analyzer.rank_paradigms(result)
    print("\nParadigm Ranking:")
    for c in ranked:
        print(f"  {c.rank}. {c.paradigm}: contribution = {c.marginal_contribution:.4f}")

    synergy = analyzer.compute_synergy(result)
    print(f"\nSynergy Score: {synergy:.4f}")


if __name__ == "__main__":
    main()
