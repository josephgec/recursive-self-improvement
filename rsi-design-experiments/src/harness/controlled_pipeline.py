"""Controlled pipeline and mock implementation for experiments."""

import math
import random
from typing import Any, Dict, List, Optional


class MockPipeline:
    """Simulates an RSI pipeline with deterministic behavior based on config.

    Behavior:
    - Modification frequency: every_5 and threshold_90 achieve ~85% accuracy;
      every_task has lower stability (more rollbacks); never stays at 75%.
    - Hindsight: both achieves 89%, library_only 82%, none 75%.
      Library generalizes better. Weights cost more.
    - RLM depth: accuracy = 0.55 + 0.17*(1-exp(-depth/1.5)),
      cost = 0.005 * 2^depth. Concave accuracy, convex cost.
    """

    # Accuracy profiles for modification frequency conditions
    FREQUENCY_PROFILES = {
        "every_task": {"base_accuracy": 0.82, "noise": 0.04, "rollback_rate": 0.25},
        "every_5": {"base_accuracy": 0.85, "noise": 0.02, "rollback_rate": 0.08},
        "every_10": {"base_accuracy": 0.83, "noise": 0.02, "rollback_rate": 0.06},
        "every_20": {"base_accuracy": 0.80, "noise": 0.01, "rollback_rate": 0.04},
        "threshold_90": {"base_accuracy": 0.85, "noise": 0.02, "rollback_rate": 0.05},
        "plateau_5": {"base_accuracy": 0.81, "noise": 0.02, "rollback_rate": 0.07},
        "never": {"base_accuracy": 0.75, "noise": 0.01, "rollback_rate": 0.02},
    }

    # Accuracy profiles for hindsight target conditions
    HINDSIGHT_PROFILES = {
        "weights_only": {
            "base_accuracy": 0.84,
            "ood_ratio": 0.85,
            "cost_factor": 1.5,
        },
        "library_only": {
            "base_accuracy": 0.82,
            "ood_ratio": 0.95,
            "cost_factor": 1.0,
        },
        "both": {
            "base_accuracy": 0.89,
            "ood_ratio": 0.92,
            "cost_factor": 2.0,
        },
        "none": {
            "base_accuracy": 0.75,
            "ood_ratio": 0.80,
            "cost_factor": 0.5,
        },
        "weights_then_library": {
            "base_accuracy": 0.86,
            "ood_ratio": 0.90,
            "cost_factor": 1.3,
        },
        "library_then_weights": {
            "base_accuracy": 0.83,
            "ood_ratio": 0.88,
            "cost_factor": 1.2,
        },
    }

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = random.Random(seed)
        self._condition_name: str = ""
        self._modification_policy: Optional[Any] = None
        self._hindsight_policy: Optional[Any] = None
        self._depth: int = 0
        self._iteration: int = 0
        self._current_accuracy: float = 0.55

    def set_seed(self, seed: int):
        """Set the random seed for reproducibility."""
        self._seed = seed
        self._rng = random.Random(seed)
        self._iteration = 0
        self._current_accuracy = 0.55

    def set_condition_name(self, name: str):
        """Set the current condition name."""
        self._condition_name = name

    def set_modification_policy(self, policy: Any):
        """Set the modification frequency policy."""
        self._modification_policy = policy

    def set_hindsight_policy(self, policy: Any):
        """Set the hindsight target policy."""
        self._hindsight_policy = policy

    def set_depth(self, depth: int):
        """Set the RLM recursion depth."""
        self._depth = depth

    def step(self, iteration: int) -> Dict[str, Any]:
        """Run one step of the pipeline and return metrics.

        Returns a dict with: accuracy, accuracy_type, rollback, llm_cost, finetuning_cost.
        """
        self._iteration = iteration

        if self._modification_policy is not None:
            return self._step_frequency(iteration)
        elif self._hindsight_policy is not None:
            return self._step_hindsight(iteration)
        else:
            return self._step_depth(iteration)

    def _step_frequency(self, iteration: int) -> Dict[str, Any]:
        """Step for modification frequency experiment."""
        profile = self.FREQUENCY_PROFILES.get(
            self._condition_name,
            {"base_accuracy": 0.75, "noise": 0.02, "rollback_rate": 0.05},
        )

        # Accuracy ramps up over iterations
        progress = min(iteration / 15.0, 1.0)
        base = 0.55 + (profile["base_accuracy"] - 0.55) * progress
        noise = self._rng.gauss(0, profile["noise"])
        accuracy = max(0.0, min(1.0, base + noise))

        # Determine if rollback occurs
        rollback = self._rng.random() < profile["rollback_rate"]

        # Determine accuracy type (70% in-distribution, 30% OOD)
        is_ood = self._rng.random() < 0.3
        accuracy_type = "out_of_distribution" if is_ood else "in_distribution"
        if is_ood:
            accuracy *= 0.9  # OOD is slightly worse

        # Cost
        should_modify = self._modification_policy.should_modify(iteration, accuracy)
        llm_cost = 0.02 if should_modify else 0.01
        finetuning_cost = 0.05 if should_modify else 0.0

        return {
            "accuracy": accuracy,
            "accuracy_type": accuracy_type,
            "rollback": rollback,
            "llm_cost": llm_cost,
            "finetuning_cost": finetuning_cost,
        }

    def _step_hindsight(self, iteration: int) -> Dict[str, Any]:
        """Step for hindsight target experiment."""
        profile = self.HINDSIGHT_PROFILES.get(
            self._condition_name,
            {"base_accuracy": 0.75, "ood_ratio": 0.80, "cost_factor": 1.0},
        )

        target = self._hindsight_policy.get_target(iteration)

        # Accuracy ramps up
        progress = min(iteration / 15.0, 1.0)
        base = 0.55 + (profile["base_accuracy"] - 0.55) * progress
        noise = self._rng.gauss(0, 0.02)
        accuracy = max(0.0, min(1.0, base + noise))

        # OOD accuracy depends on the ood_ratio
        is_ood = self._rng.random() < 0.3
        accuracy_type = "out_of_distribution" if is_ood else "in_distribution"
        if is_ood:
            accuracy *= profile["ood_ratio"]

        rollback = self._rng.random() < 0.05

        # Cost depends on target
        cost_mult = profile["cost_factor"]
        llm_cost = 0.01 * cost_mult
        finetuning_cost = 0.03 * cost_mult if target in ("weights", "both") else 0.0

        return {
            "accuracy": accuracy,
            "accuracy_type": accuracy_type,
            "rollback": rollback,
            "llm_cost": llm_cost,
            "finetuning_cost": finetuning_cost,
        }

    def _step_depth(self, iteration: int) -> Dict[str, Any]:
        """Step for RLM depth experiment."""
        # accuracy = 0.55 + 0.17*(1-exp(-depth/1.5))
        theoretical_acc = 0.55 + 0.17 * (1.0 - math.exp(-self._depth / 1.5))

        # Ramp up over iterations toward theoretical accuracy
        progress = min(iteration / 15.0, 1.0)
        base = 0.55 + (theoretical_acc - 0.55) * progress
        noise = self._rng.gauss(0, 0.015)
        accuracy = max(0.0, min(1.0, base + noise))

        is_ood = self._rng.random() < 0.3
        accuracy_type = "out_of_distribution" if is_ood else "in_distribution"
        if is_ood:
            accuracy *= 0.92

        # Higher depth = more rollbacks (instability)
        rollback_rate = 0.02 + self._depth * 0.01
        rollback = self._rng.random() < rollback_rate

        # cost = 0.005 * 2^depth
        depth_cost = 0.005 * (2 ** self._depth)
        llm_cost = 0.01 + depth_cost
        finetuning_cost = depth_cost * 0.5

        return {
            "accuracy": accuracy,
            "accuracy_type": accuracy_type,
            "rollback": rollback,
            "llm_cost": llm_cost,
            "finetuning_cost": finetuning_cost,
        }


class ControlledPipeline:
    """Wraps a MockPipeline to ensure controlled experimental conditions."""

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._pipeline = MockPipeline(seed)

    def get_pipeline(self) -> MockPipeline:
        """Get the underlying pipeline, reset to initial state."""
        self._pipeline = MockPipeline(self._seed)
        return self._pipeline

    def run_with_config(
        self,
        config: Dict[str, Any],
        eval_tasks: int = 50,
        iterations: int = 20,
    ) -> List[Dict[str, Any]]:
        """Run the pipeline with a given config and return per-iteration results."""
        pipeline = self.get_pipeline()

        if "modification_policy" in config:
            pipeline.set_modification_policy(config["modification_policy"])
        if "hindsight_policy" in config:
            pipeline.set_hindsight_policy(config["hindsight_policy"])
        if "depth" in config:
            pipeline.set_depth(config["depth"])
        if "condition_name" in config:
            pipeline.set_condition_name(config["condition_name"])
        if "seed" in config:
            pipeline.set_seed(config["seed"])

        results = []
        for i in range(iterations):
            step_result = pipeline.step(i)
            results.append(step_result)

        return results
