"""Main Godel Agent orchestrator."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.core.executor import Task, TaskResult, TaskExecutor, create_llm_client
from src.core.runtime import RuntimeInspector
from src.core.state import AgentState, StateManager
from src.meta.prompt_strategy import DefaultPromptStrategy
from src.meta.registry import ComponentRegistry
from src.modification.deliberation import DeliberationEngine
from src.modification.modifier import CodeModifier
from src.validation.rollback import RollbackManager
from src.validation.runner import ValidationRunner
from src.validation.suite import ValidationSuite
from src.audit.logger import AuditLogger
from src.audit.safety_hooks import SafetyHooks

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    """Result of a single iteration."""

    iteration: int
    accuracy: float
    results: list[TaskResult]
    deliberated: bool = False
    modification_applied: bool = False
    modification_rolled_back: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class GodelAgent:
    """Self-modifying agent with rollback-guarded self-improvement."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        agent_cfg = config.get("agent", {})
        meta_cfg = config.get("meta_learning", {})
        mod_cfg = config.get("modification", {})
        val_cfg = config.get("validation", {})
        audit_cfg = config.get("audit", {})

        # Core components
        self.llm = create_llm_client(
            provider=agent_cfg.get("llm_provider", "mock"),
            model=agent_cfg.get("llm_model", "gpt-4o"),
        )
        self.executor = TaskExecutor(self.llm, config)
        self.state_manager = StateManager(
            checkpoint_dir=config.get("project", {}).get("output_dir", "data") + "/checkpoints"
        )
        self.runtime_inspector = RuntimeInspector()

        # Meta-learning
        self.strategy = DefaultPromptStrategy(
            system_prompt=agent_cfg.get("system_prompt", ""),
            num_examples=meta_cfg.get("num_examples", 3),
        )
        self.registry = ComponentRegistry()
        self.registry.register("prompt_strategy", self.strategy)

        # Modification
        self.deliberation = DeliberationEngine(self.llm, mod_cfg)
        self.modifier = CodeModifier(
            allowed_targets=mod_cfg.get("allowed_targets", []),
            forbidden_targets=mod_cfg.get("forbidden_targets", []),
        )

        # Validation
        self.validation_suite = ValidationSuite(val_cfg.get("suite", "core"))
        self.validation_runner = ValidationRunner(
            self.executor,
            self.validation_suite,
            min_pass_rate=val_cfg.get("min_pass_rate", 0.90),
            performance_threshold=val_cfg.get("performance_threshold", -0.05),
        )
        self.rollback = RollbackManager(self.state_manager)

        # Audit
        self.audit = AuditLogger(
            log_dir=audit_cfg.get("log_dir", "data/audit_logs"),
            log_diffs=audit_cfg.get("log_diffs", True),
            log_reasoning=audit_cfg.get("log_reasoning", True),
        )
        self.safety_hooks = SafetyHooks(
            max_complexity_ratio=mod_cfg.get("max_complexity_ratio", 5.0),
        )

        # Config values
        self.max_iterations = meta_cfg.get("max_iterations", 50)
        self.warmup_iterations = meta_cfg.get("warmup_iterations", 5)
        self.tasks_per_iteration = meta_cfg.get("tasks_per_iteration", 10)
        self.modification_cooldown = meta_cfg.get("modification_cooldown", 2)
        self.require_deliberation = mod_cfg.get("require_deliberation", True)
        self.auto_rollback = val_cfg.get("auto_rollback", True)

        # Tracking
        self._iteration_results: list[IterationResult] = []
        self._last_modification_iter = -self.modification_cooldown  # allow first mod
        self._initial_complexity: float | None = None

    def run(
        self,
        tasks: list[Task],
        max_iterations: int | None = None,
        allow_modification: bool = True,
    ) -> list[IterationResult]:
        """Run the agent for multiple iterations."""
        max_iters = max_iterations or self.max_iterations

        # Create initial state
        state = self.state_manager.create_initial_state(
            system_prompt=self.config.get("agent", {}).get("system_prompt", ""),
        )

        # Set validation baseline
        if self.validation_suite.size > 0:
            self.validation_runner.set_baseline(state.system_prompt)

        results: list[IterationResult] = []
        for i in range(max_iters):
            # Select tasks for this iteration
            batch_size = min(self.tasks_per_iteration, len(tasks))
            start_idx = (i * batch_size) % len(tasks)
            batch = tasks[start_idx : start_idx + batch_size]
            if len(batch) < batch_size:
                batch = batch + tasks[: batch_size - len(batch)]

            result = self._run_iteration(i, batch, state, allow_modification)
            results.append(result)
            self._iteration_results.append(result)

            # Update state
            state.iteration = i + 1
            state.accuracy_history.append(result.accuracy)
            self.state_manager.capture(state)

            logger.info(
                f"Iteration {i}: accuracy={result.accuracy:.3f}, "
                f"deliberated={result.deliberated}, "
                f"modified={result.modification_applied}"
            )

        return results

    def _run_iteration(
        self,
        iteration: int,
        tasks: list[Task],
        state: AgentState,
        allow_modification: bool,
    ) -> IterationResult:
        """Run a single iteration of the agent loop."""
        # Execute tasks
        task_results = self.executor.execute_batch(
            tasks,
            system_prompt=state.system_prompt,
            few_shot_examples=state.few_shot_examples,
            reasoning_mode="cot",
        )

        correct = sum(1 for r in task_results if r.correct)
        accuracy = correct / len(task_results) if task_results else 0.0

        result = IterationResult(
            iteration=iteration,
            accuracy=accuracy,
            results=task_results,
        )

        # Log iteration
        self.audit.log_iteration(iteration, accuracy, len(task_results), correct)

        # Check if we should deliberate (only after warmup, if modification allowed)
        if (
            allow_modification
            and iteration >= self.warmup_iterations
            and self._should_deliberate(iteration, state.accuracy_history + [accuracy])
        ):
            result.deliberated = True

            # Safety hooks
            safe = True
            if not self.safety_hooks.check_modification_rate(
                self._iteration_results, iteration
            ):
                safe = False
                logger.warning("Modification rate limit exceeded")

            if safe:
                mod_applied, rolled_back = self._apply_modification_guarded(
                    iteration, state, accuracy
                )
                result.modification_applied = mod_applied
                result.modification_rolled_back = rolled_back

        return result

    def _should_deliberate(self, iteration: int, accuracy_history: list[float]) -> bool:
        """Determine if the agent should deliberate on self-modification."""
        if iteration - self._last_modification_iter < self.modification_cooldown:
            return False

        if len(accuracy_history) < 2:
            return False

        # Performance drop
        perf = self.runtime_inspector.inspect_performance(accuracy_history)
        if perf.trend < -0.02:
            return True

        # Stagnation: last 3 iterations similar
        if len(accuracy_history) >= 3:
            recent = accuracy_history[-3:]
            if max(recent) - min(recent) < 0.01:
                return True

        # Periodic: every 5 iterations
        if iteration % 5 == 0 and iteration > 0:
            return True

        return False

    def _apply_modification_guarded(
        self, iteration: int, state: AgentState, current_accuracy: float
    ) -> tuple[bool, bool]:
        """Apply a modification with validation and rollback.

        Returns (modification_applied, was_rolled_back).
        """
        # Checkpoint
        checkpoint_path = self.rollback.checkpoint(state)

        # Deliberate
        report = self.runtime_inspector.generate_self_report(
            state.accuracy_history,
            state.modifications_applied,
        )
        delib_result = self.deliberation.deliberate(
            trigger="performance",
            self_report=report,
            state=state,
            registry=self.registry,
        )

        self.audit.log_deliberation(iteration, delib_result.to_dict())

        if not delib_result.should_proceed:
            return False, False

        # Validate and apply proposal
        proposal = delib_result.proposal
        if proposal is None:
            return False, False

        validation = self.modifier.validate_proposal(proposal)
        if not validation["valid"]:
            logger.warning(f"Invalid proposal: {validation.get('reason', 'unknown')}")
            return False, False

        # Apply
        try:
            mod_result = self.modifier.apply_modification(proposal, self.registry)
        except Exception as e:
            logger.error(f"Modification failed: {e}")
            self.audit.log_rollback(iteration, str(e))
            return False, False

        if not mod_result.success:
            self.audit.log_rollback(iteration, mod_result.error)
            return False, False

        # Validate
        if self.auto_rollback and self.validation_suite.size > 0:
            val_result = self.validation_runner.run_quick(state.system_prompt)
            if not val_result.passed:
                # Rollback
                self.modifier.revert(mod_result)
                self.audit.log_rollback(iteration, "Validation failed after modification")
                self._last_modification_iter = iteration
                return True, True

        # Success
        self.audit.log_modification(iteration, proposal.to_dict(), accepted=True)
        state.modifications_applied.append(proposal.to_dict())
        self._last_modification_iter = iteration
        return True, False
