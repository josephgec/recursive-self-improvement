#!/usr/bin/env python3
"""Run a full constraint check against a mock agent."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.checker.suite import ConstraintSuite
from src.checker.runner import ConstraintRunner
from src.constraints.base import CheckContext


class _MockAgent:
    """Simple mock agent for script usage."""

    def __init__(self):
        self.held_out_tasks = None

    def evaluate(self, tasks):
        return [{"correct": True, "category": t.get("category", "general")} for t in tasks]

    def generate_probe_outputs(self, probes):
        import random
        random.seed(42)
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                 "cat", "bird", "tree", "sky", "sun", "moon", "star", "river",
                 "mountain", "ocean", "forest", "cloud", "rain", "snow", "wind"]
        outputs = []
        for _ in probes:
            n = random.randint(10, 30)
            outputs.append(" ".join(random.choices(words, k=n)))
        return outputs

    def generate_outputs(self, prompts):
        return ["I cannot help with that request." for _ in prompts]

    def compute_drift(self):
        return 0.15

    def get_benchmark_scores(self):
        return {"mmlu": 82.0, "hellaswag": 79.0, "arc_challenge": 75.0}

    def get_baseline_scores(self):
        return {"mmlu": 83.0, "hellaswag": 80.0, "arc_challenge": 76.0}

    def evaluate_consistency(self, pairs):
        return [{"equivalent": True} for _ in pairs]

    def get_latency_samples(self):
        return [1000 + i * 100 for i in range(50)]


def main():
    suite = ConstraintSuite()
    runner = ConstraintRunner(suite, parallel=False)
    agent = _MockAgent()
    context = CheckContext(modification_type="script_check")

    verdict = runner.run(agent, context)
    print(verdict.summary())
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
