"""Controlled environment for isolated scenario execution."""

from __future__ import annotations

import copy
import inspect
import random
import textwrap
import types
from typing import Any, Callable, Dict, List, Optional

from src.adversarial.scenario_registry import AdversarialScenario
from src.utils.metrics import compute_cyclomatic_complexity, count_ast_nodes


class MockAgent:
    """Lightweight mock of a self-modifying agent for testing.

    Simulates:
    - Modifiable functions (stored as source code + compiled callables)
    - Accuracy tracking
    - Modification and rollback
    - Health inspection
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self._functions: Dict[str, str] = {}
        self._compiled: Dict[str, Callable[..., Any]] = {}
        self._accuracy: float = 0.8
        self._true_accuracy: float = 0.8
        self._modification_count: int = 0
        self._rollback_count: int = 0
        self._validation_rejections: int = 0
        self._functional: bool = True
        self._can_modify: bool = True
        self._can_validate: bool = True
        self._history: List[Dict[str, Any]] = []
        self._checkpoints: List[Dict[str, Any]] = []
        self._task: str = ""
        self._target_function: Optional[str] = None
        self._scoring_mode: str = "normal"
        self._nested_mod: bool = False
        self._mod_depth: int = 0
        self._rollback_delay: float = 0.0
        self._nesting_depth: int = 0
        self._function_length: int = 0
        self._state_variables: int = 0
        self._complexity_schedule: Optional[Dict[str, Any]] = None
        self._circular_deps: List[tuple[str, str]] = []
        self._distribution_shift_at: Optional[int] = None
        self._task_difficulty: str = "normal"
        self._poisoning: Optional[Dict[str, float]] = None
        self._iteration: int = 0
        self._checkpoint_corrupted: bool = False
        self._baseline_corrupted: bool = False
        self._rollback_tracking: bool = False

        # Install a default modifiable function
        self.install_function(
            "solve",
            textwrap.dedent("""\
                def solve(x):
                    if x > 0:
                        return x * 2
                    else:
                        return 0
            """),
        )

    # ---- Core interface ---- #

    def install_function(self, name: str, source: str) -> None:
        """Install a function by source code."""
        self._functions[name] = source
        try:
            code_obj = compile(source, f"<{name}>", "exec")
            namespace: Dict[str, Any] = {}
            # Add references to other installed functions
            for fn_name, fn_callable in self._compiled.items():
                namespace[fn_name] = fn_callable
            exec(code_obj, namespace)
            if name in namespace:
                self._compiled[name] = namespace[name]
        except Exception:
            pass  # Store source even if it doesn't compile

    def get_function_source(self, name: str) -> Optional[str]:
        """Get the source code of a function."""
        return self._functions.get(name)

    def get_all_code(self) -> str:
        """Get all function source code concatenated."""
        return "\n\n".join(self._functions.values())

    def modify_function(self, name: str, new_source: str) -> bool:
        """Attempt to modify a function. Returns True if accepted."""
        if not self._can_modify:
            return False

        self._modification_count += 1
        self._mod_depth += 1

        # Validate: try to compile
        try:
            compile(new_source, f"<{name}>", "exec")
        except SyntaxError:
            self._validation_rejections += 1
            self._mod_depth -= 1
            return False

        # Save checkpoint before modification
        self._save_checkpoint()

        old_source = self._functions.get(name, "")
        self.install_function(name, new_source)

        # Simulate accuracy change
        old_complexity = count_ast_nodes(old_source)
        new_complexity = count_ast_nodes(new_source)

        # More complex code has higher chance of degrading accuracy
        if new_complexity > old_complexity * 1.5:
            self._accuracy = max(0.0, self._accuracy - self.rng.uniform(0.05, 0.15))
        elif new_complexity < old_complexity * 0.8:
            self._accuracy = min(1.0, self._accuracy + self.rng.uniform(0.0, 0.05))
        else:
            self._accuracy += self.rng.uniform(-0.05, 0.05)
            self._accuracy = max(0.0, min(1.0, self._accuracy))

        self._true_accuracy = self._accuracy
        self._mod_depth -= 1
        self._record_history()
        return True

    def rollback(self) -> bool:
        """Roll back to the last checkpoint."""
        if not self._checkpoints:
            return False

        self._rollback_count += 1
        checkpoint = self._checkpoints.pop()

        if self._checkpoint_corrupted:
            # Corrupted checkpoint: partial restore
            self._accuracy = checkpoint.get("accuracy", self._accuracy) * 0.5
            return True

        self._functions = checkpoint["functions"]
        self._compiled = checkpoint.get("compiled", {})
        self._accuracy = checkpoint["accuracy"]
        self._true_accuracy = checkpoint.get("true_accuracy", self._accuracy)
        return True

    def run_iteration(self) -> float:
        """Run one iteration of the agent loop. Returns accuracy."""
        self._iteration += 1

        # Apply distribution shift if configured
        if self._distribution_shift_at and self._iteration >= self._distribution_shift_at:
            self._accuracy = max(0.0, self._accuracy - 0.1)

        # Apply poisoning if configured
        if self._poisoning:
            frac = min(
                self._poisoning["end_fraction"],
                self._poisoning["start_fraction"]
                + (self._iteration / max(self._poisoning.get("ramp_iterations", 20), 1))
                * (self._poisoning["end_fraction"] - self._poisoning["start_fraction"]),
            )
            self._accuracy = max(0.0, self._accuracy - frac * 0.1)

        # Apply complexity schedule
        if self._complexity_schedule:
            sched = self._complexity_schedule
            target_complexity = sched["start"] + self._iteration * sched.get("step", 0)
            if target_complexity > 200:
                self._accuracy = max(0.0, self._accuracy - 0.02)

        self._record_history()

        if self._scoring_mode == "inverted":
            return 1.0 - self._accuracy
        return self._accuracy

    # ---- Setup helpers used by scenarios ---- #

    def set_task(self, task: str) -> None:
        self._task = task

    def mark_function_as_target(self, name: str) -> None:
        self._target_function = name

    def enable_nested_modification(self) -> None:
        self._nested_mod = True

    def create_circular_dependency(self, a: str, b: str) -> None:
        self._circular_deps.append((a, b))

    def enable_rollback_tracking(self) -> None:
        self._rollback_tracking = True

    def set_scoring_mode(self, mode: str) -> None:
        self._scoring_mode = mode

    def set_distribution_shift(self, shift_at_iteration: int) -> None:
        self._distribution_shift_at = shift_at_iteration

    def set_task_difficulty(self, difficulty: str) -> None:
        self._task_difficulty = difficulty

    def set_poisoning_schedule(
        self, start_fraction: float, end_fraction: float, ramp_iterations: int
    ) -> None:
        self._poisoning = {
            "start_fraction": start_fraction,
            "end_fraction": end_fraction,
            "ramp_iterations": ramp_iterations,
        }

    def set_complexity_schedule(
        self, mode: str, start: int = 30, step: int = 20
    ) -> None:
        self._complexity_schedule = {"mode": mode, "start": start, "step": step}

    def set_nesting_depth(self, depth: int) -> None:
        self._nesting_depth = depth

    def set_function_length(self, length: int) -> None:
        self._function_length = length

    def set_state_variables(self, count: int) -> None:
        self._state_variables = count

    def corrupt_checkpoint(self, mode: str = "format") -> None:
        self._checkpoint_corrupted = True

    def corrupt_baseline(self, new_score: float = 0.0) -> None:
        self._baseline_corrupted = True
        self._accuracy = new_score

    def set_rollback_delay(self, seconds: float) -> None:
        self._rollback_delay = seconds

    # ---- Inspection ---- #

    def is_functional(self) -> bool:
        return self._functional

    def can_modify(self) -> bool:
        return self._can_modify

    def can_validate(self) -> bool:
        return self._can_validate

    def modification_depth(self) -> int:
        return self._mod_depth

    def modification_count(self) -> int:
        return self._modification_count

    def get_accuracy(self) -> float:
        if self._scoring_mode == "inverted":
            return 1.0 - self._accuracy
        return self._accuracy

    def get_true_accuracy(self) -> float:
        return self._true_accuracy

    def get_complexity(self) -> int:
        code = self.get_all_code()
        return count_ast_nodes(code) if code else 0

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "code": self.get_all_code(),
            "accuracy": self._accuracy,
            "complexity": self.get_complexity(),
            "modification_count": self._modification_count,
            "functional": self._functional,
        }

    # ---- Internal ---- #

    def _save_checkpoint(self) -> None:
        self._checkpoints.append({
            "functions": dict(self._functions),
            "compiled": dict(self._compiled),
            "accuracy": self._accuracy,
            "true_accuracy": self._true_accuracy,
        })

    def _record_history(self) -> None:
        self._history.append({
            "iteration": self._iteration,
            "accuracy": self._accuracy,
            "complexity": self.get_complexity(),
            "modification_count": self._modification_count,
        })


class ControlledEnvironment:
    """Provides isolated environments for each scenario."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def create_fresh_agent(self) -> MockAgent:
        """Create a fresh mock agent for an isolated scenario run."""
        return MockAgent(seed=self._seed)

    def apply_scenario(
        self, agent: MockAgent, scenario: AdversarialScenario
    ) -> None:
        """Apply a scenario's setup to an agent."""
        scenario.setup(agent)

    def run_iterations(
        self,
        agent: MockAgent,
        scenario: AdversarialScenario,
        max_iterations: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run a scenario for the specified number of iterations.

        Returns:
            List of per-iteration results.
        """
        n = max_iterations or scenario.max_iterations
        results = []

        for i in range(n):
            accuracy = agent.run_iteration()
            complexity = agent.get_complexity()

            result = {
                "iteration": i,
                "accuracy": accuracy,
                "complexity": complexity,
                "functional": agent.is_functional(),
                "modification_count": agent.modification_count(),
            }
            results.append(result)

            # Check success criteria
            if scenario.success_criteria and scenario.success_criteria(agent):
                result["success_criteria_met"] = True

            # Check if agent is dead
            if not agent.is_functional():
                break

        return results
