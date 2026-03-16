"""Ablation runner with mock pipeline support."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Callable, Dict, List, Optional

from src.suites.base import (
    AblationCondition,
    AblationSuite,
    AblationSuiteResult,
    ConditionRun,
)


# Global accuracy map used by MockPipeline
MOCK_ACCURACY_MAP: Dict[str, float] = {
    # Neurosymbolic
    "full": 0.85,
    "symcode_only": 0.78,
    "bdm_only": 0.74,
    "prose_only": 0.72,
    "code_no_verify": 0.79,
    "hybrid_no_bdm": 0.80,
    "integrative": 0.83,
    # Godel
    "no_rollback": 0.78,
    "no_validation": 0.76,
    "no_ceiling": 0.80,
    "no_cooldown": 0.81,
    "no_audit": 0.82,
    "no_self_mod": 0.75,
    "unrestricted": 0.70,
    # SOAR
    "no_hindsight": 0.77,
    "no_crossover": 0.79,
    "no_error_guidance": 0.78,
    "no_mutation": 0.73,
    "random_search": 0.65,
    "single_candidate": 0.71,
    "hindsight_library": 0.84,
    # RLM
    "no_recursion": 0.73,
    "no_helpers": 0.78,
    "depth_1": 0.76,
    "no_repl": 0.60,
    "repl_no_code": 0.68,
    "chunked_prompt": 0.70,
    "rag_baseline": 0.72,
}


class MockPipeline:
    """Deterministic mock pipeline that returns accuracy based on condition name.

    Uses a seeded random generator to add realistic noise (std ~0.02).
    """

    def __init__(self, accuracy_map: Optional[Dict[str, float]] = None,
                 noise_std: float = 0.02):
        self.accuracy_map = accuracy_map or MOCK_ACCURACY_MAP
        self.noise_std = noise_std

    def run(self, condition: AblationCondition, seed: int = 42) -> float:
        """Run the mock pipeline and return accuracy."""
        base_acc = self.accuracy_map.get(condition.name, 0.70)
        # Deterministic noise from seed + condition name
        rng = random.Random(seed + hash(condition.name))
        noise = rng.gauss(0, self.noise_std)
        return max(0.0, min(1.0, base_acc + noise))


class AblationRunner:
    """Runs ablation studies across all conditions."""

    def __init__(self, pipeline_runner: Optional[Any] = None,
                 noise_std: float = 0.02):
        if pipeline_runner is None:
            self.pipeline = MockPipeline(noise_std=noise_std)
        elif isinstance(pipeline_runner, MockPipeline):
            self.pipeline = pipeline_runner
        else:
            self.pipeline = pipeline_runner

    def run_suite(self, suite: AblationSuite, repetitions: int = 5,
                  seed: int = 42) -> AblationSuiteResult:
        """Run all conditions in a suite with the given number of repetitions."""
        result = AblationSuiteResult(suite_name=suite.get_paper_name())
        conditions = suite.get_conditions()

        for condition in conditions:
            runs: List[ConditionRun] = []
            for rep in range(repetitions):
                rep_seed = seed + rep
                accuracy = self.pipeline.run(condition, seed=rep_seed)
                run = ConditionRun(
                    condition_name=condition.name,
                    repetition=rep,
                    accuracy=accuracy,
                    seed=rep_seed,
                    metrics={"accuracy": accuracy},
                )
                runs.append(run)
            result.condition_runs[condition.name] = runs

        return result

    def run_all_suites(self, suites: List[AblationSuite],
                       repetitions: int = 5,
                       seed: int = 42) -> Dict[str, AblationSuiteResult]:
        """Run multiple suites."""
        results = {}
        for suite in suites:
            results[suite.get_paper_name()] = self.run_suite(
                suite, repetitions=repetitions, seed=seed
            )
        return results

    def estimate_cost(self, suite: AblationSuite, repetitions: int = 5,
                      cost_per_run: float = 0.10) -> Dict[str, float]:
        """Estimate cost of running a suite."""
        n_conditions = len(suite.get_conditions())
        total_runs = n_conditions * repetitions
        return {
            "n_conditions": n_conditions,
            "repetitions": repetitions,
            "total_runs": total_runs,
            "cost_per_run": cost_per_run,
            "total_cost": total_runs * cost_per_run,
        }

    @staticmethod
    def _verify_power(scores_a: List[float], scores_b: List[float],
                      alpha: float = 0.05) -> Dict[str, float]:
        """Verify statistical power of a comparison."""
        from src.analysis.power_analysis import achieved_power
        from src.analysis.effect_sizes import cohens_d

        d = cohens_d(scores_a, scores_b)
        n = min(len(scores_a), len(scores_b))
        power = achieved_power(d, n, alpha)
        return {"cohens_d": d, "n": n, "power": power, "adequate": power >= 0.80}
