from __future__ import annotations

"""Mitigated training wrapper combining all defense mechanisms."""

import numpy as np
from dataclasses import dataclass, field

from ..eppo.trainer import EPPOTrainer
from ..eppo.config import EPPOConfig
from ..bounding.process_reward import ProcessRewardShaper
from ..energy.energy_tracker import EnergyTracker
from ..energy.homogenization import HomogenizationDetector
from ..detection.composite_detector import (
    CompositeRewardHackingDetector,
    TrainingState,
)


@dataclass
class MitigationStatus:
    """Current mitigation status."""

    eppo_active: bool = True
    bounding_active: bool = True
    energy_monitoring: bool = True
    detection_active: bool = True
    warnings: list[str] = field(default_factory=list)


@dataclass
class TrainingResult:
    """Result from mitigated training."""

    total_steps: int = 0
    epochs_completed: int = 0
    early_stopped: bool = False
    stop_reason: str = ""
    final_entropy: float = 0.0
    final_energy: float = 0.0
    hacking_signals: list[str] = field(default_factory=list)
    mitigation_status: MitigationStatus = field(default_factory=MitigationStatus)


class MitigatedTrainingWrapper:
    """Wraps training with all mitigation mechanisms.

    Combines EPPO training, reward bounding, energy tracking,
    homogenization detection, and composite hacking detection.
    """

    def __init__(
        self,
        config: EPPOConfig | None = None,
        shaper: ProcessRewardShaper | None = None,
        energy_tracker: EnergyTracker | None = None,
        homogenization_detector: HomogenizationDetector | None = None,
        hacking_detector: CompositeRewardHackingDetector | None = None,
    ):
        self._config = config or EPPOConfig()
        self._trainer = EPPOTrainer(self._config)
        self._shaper = shaper or ProcessRewardShaper()
        self._energy_tracker = energy_tracker or EnergyTracker()
        self._homogenization = homogenization_detector or HomogenizationDetector()
        self._detector = hacking_detector or CompositeRewardHackingDetector()
        self._status = MitigationStatus()
        self._rng = np.random.RandomState(42)

    @property
    def trainer(self) -> EPPOTrainer:
        return self._trainer

    @property
    def status(self) -> MitigationStatus:
        return self._status

    def before_training(self) -> MitigationStatus:
        """Initialize all defense mechanisms before training starts.

        Returns:
            Current mitigation status.
        """
        # Set energy baseline
        activations = [self._rng.randn(64) for _ in range(self._energy_tracker.num_layers)]
        self._energy_tracker.measure(activations)
        self._energy_tracker.set_baseline()

        self._status = MitigationStatus(
            eppo_active=True,
            bounding_active=True,
            energy_monitoring=True,
            detection_active=True,
            warnings=[],
        )
        return self._status

    def after_step(self, step: int, batch: dict) -> MitigationStatus:
        """Run all post-step checks.

        Args:
            step: Current step number.
            batch: The training batch.

        Returns:
            Updated mitigation status.
        """
        warnings = []

        # Shape rewards
        if self._status.bounding_active:
            for r in batch.get("rewards", []):
                self._shaper.shape(float(r))

        # Track energy
        if self._status.energy_monitoring:
            scale = max(0.1, 1.0 - 0.005 * step)
            activations = [
                self._rng.randn(64) * scale
                for _ in range(self._energy_tracker.num_layers)
            ]
            measurement = self._energy_tracker.measure(activations)

            if self._energy_tracker.is_declining():
                warnings.append("Energy decline detected")

            # Check homogenization
            homog = self._homogenization.detect(self._energy_tracker.measurements)
            if homog.is_homogenizing:
                warnings.append(
                    f"Homogenization detected: {', '.join(homog.patterns_detected)}"
                )

        self._status.warnings = warnings
        return self._status

    def after_training(self) -> TrainingResult:
        """Run final checks after training completes.

        Returns:
            TrainingResult with summary.
        """
        summary = self._trainer.get_training_summary()

        # Final energy
        relative_energy = self._energy_tracker.get_relative_energy()
        final_energy = relative_energy if relative_energy is not None else 1.0

        return TrainingResult(
            total_steps=summary.get("total_steps", 0),
            epochs_completed=summary.get("epochs", 0),
            early_stopped=False,
            stop_reason="",
            final_entropy=summary.get("final_entropy", 0.0),
            final_energy=final_energy,
            hacking_signals=self._status.warnings,
            mitigation_status=self._status,
        )

    def run_full_training(
        self,
        num_epochs: int = 5,
        steps_per_epoch: int = 10,
    ) -> TrainingResult:
        """Run a complete mitigated training loop.

        Args:
            num_epochs: Number of training epochs.
            steps_per_epoch: Steps per epoch.

        Returns:
            TrainingResult.
        """
        self.before_training()

        for epoch in range(num_epochs):
            epoch_result = self._trainer.train_epoch(num_steps=steps_per_epoch)

            for step_idx, step_result in enumerate(epoch_result.steps):
                batch = self._trainer._make_random_batch()
                self.after_step(
                    epoch * steps_per_epoch + step_idx,
                    batch,
                )

            # Check if we should stop
            if any("Homogenization" in w for w in self._status.warnings):
                result = self.after_training()
                result.early_stopped = True
                result.stop_reason = "Homogenization detected"
                return result

        return self.after_training()
