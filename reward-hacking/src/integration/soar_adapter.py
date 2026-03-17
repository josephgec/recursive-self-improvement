from __future__ import annotations

"""SOAR (Safe Online Adaptive Reinforcement) adapter for reward hacking defense."""

import numpy as np
from dataclasses import dataclass, field

from ..eppo.trainer import EPPOTrainer, EPPOStepResult
from ..bounding.process_reward import ProcessRewardShaper
from ..energy.energy_tracker import EnergyTracker
from ..detection.composite_detector import (
    CompositeRewardHackingDetector,
    TrainingState,
    RewardHackingReport,
)


@dataclass
class SOARStepResult:
    """Result from a SOAR-wrapped training step."""

    eppo_result: EPPOStepResult
    shaped_reward: float
    energy: float | None
    hacking_detected: bool
    step: int


@dataclass
class SOARTrainingResult:
    """Result from a complete SOAR-wrapped training run."""

    steps: list[SOARStepResult] = field(default_factory=list)
    final_report: RewardHackingReport | None = None
    was_stopped_early: bool = False
    total_steps: int = 0


class SOARRewardHackingAdapter:
    """Wraps training with SOAR-based reward hacking defense.

    Integrates EPPO training, reward shaping, energy tracking,
    and hacking detection into a unified training loop.
    """

    def __init__(
        self,
        trainer: EPPOTrainer | None = None,
        shaper: ProcessRewardShaper | None = None,
        energy_tracker: EnergyTracker | None = None,
        detector: CompositeRewardHackingDetector | None = None,
    ):
        self._trainer = trainer or EPPOTrainer()
        self._shaper = shaper or ProcessRewardShaper()
        self._energy_tracker = energy_tracker or EnergyTracker()
        self._detector = detector or CompositeRewardHackingDetector()
        self._step_results: list[SOARStepResult] = []
        self._rng = np.random.RandomState(42)

    @property
    def trainer(self) -> EPPOTrainer:
        return self._trainer

    @property
    def step_results(self) -> list[SOARStepResult]:
        return list(self._step_results)

    def on_training_step(self, batch: dict, raw_rewards: np.ndarray) -> SOARStepResult:
        """Process a single training step with all defenses.

        Args:
            batch: Training batch for EPPO.
            raw_rewards: Raw reward values to shape.

        Returns:
            SOARStepResult with all defense outputs.
        """
        # Shape rewards
        shaped_rewards = []
        for r in raw_rewards.flatten():
            shaped = self._shaper.shape(float(r))
            shaped_rewards.append(shaped.final)

        # Update batch with shaped rewards
        batch = dict(batch)
        batch["rewards"] = np.array(shaped_rewards, dtype=np.float32)

        # Run EPPO step
        eppo_result = self._trainer.train_step(batch)

        # Track energy (mock activations)
        num_layers = self._energy_tracker.num_layers
        activations = [
            self._rng.randn(64) * (1.0 - 0.01 * len(self._step_results))
            for _ in range(num_layers)
        ]
        energy_measurement = self._energy_tracker.measure(activations)

        result = SOARStepResult(
            eppo_result=eppo_result,
            shaped_reward=float(np.mean(shaped_rewards)),
            energy=energy_measurement.total_energy,
            hacking_detected=False,
            step=len(self._step_results),
        )
        self._step_results.append(result)
        return result

    def on_epoch_end(
        self,
        output_lengths: list[int] | None = None,
        baseline_lengths: list[int] | None = None,
        outputs: list[list[int]] | None = None,
        output_strings: list[str] | None = None,
    ) -> RewardHackingReport:
        """Run end-of-epoch detection checks.

        Args:
            output_lengths: Current output lengths.
            baseline_lengths: Baseline output lengths.
            outputs: Token sequences.
            output_strings: String outputs for gaming tests.

        Returns:
            RewardHackingReport.
        """
        # Defaults
        if output_lengths is None:
            output_lengths = [50] * 10
        if baseline_lengths is None:
            baseline_lengths = [40] * 10
        if outputs is None:
            outputs = [list(self._rng.randint(0, 100, 50)) for _ in range(10)]
        if output_strings is None:
            output_strings = ["Sample output text"] * 10

        rewards = [r.shaped_reward for r in self._step_results[-20:]]
        accuracies = [0.5 + self._rng.randn() * 0.05 for _ in rewards]

        state = TrainingState(
            rewards=rewards,
            accuracies=accuracies,
            output_lengths=output_lengths,
            baseline_lengths=baseline_lengths,
            outputs=outputs,
            output_strings=output_strings,
        )

        return self._detector.check(state)

    def wrap_training(
        self,
        num_steps: int = 50,
        check_interval: int = 10,
    ) -> SOARTrainingResult:
        """Run a complete training loop with all defenses.

        Args:
            num_steps: Total training steps.
            check_interval: Steps between hacking checks.

        Returns:
            SOARTrainingResult.
        """
        result = SOARTrainingResult()

        for step in range(num_steps):
            batch = self._trainer._make_random_batch()
            raw_rewards = batch["rewards"].copy()

            step_result = self.on_training_step(batch, raw_rewards)
            result.steps.append(step_result)

            if (step + 1) % check_interval == 0:
                report = self.on_epoch_end()
                result.final_report = report

                if report.should_stop:
                    result.was_stopped_early = True
                    break

        result.total_steps = len(result.steps)
        return result
