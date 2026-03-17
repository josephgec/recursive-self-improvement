from __future__ import annotations

"""Energy tracking for activation norms."""

import numpy as np
from dataclasses import dataclass


@dataclass
class EnergyMeasurement:
    """Single energy measurement from activations."""

    total_energy: float
    per_layer_energy: list[float]
    num_layers: int
    step: int
    relative_energy: float | None = None


class EnergyTracker:
    """Tracks activation energy (L2 norms) across layers.

    Monitors for energy decline that may indicate model collapse
    or reward hacking via representation degeneration.
    """

    def __init__(self, num_layers: int = 6, baseline_window: int = 50):
        self._num_layers = num_layers
        self._baseline_window = baseline_window
        self._measurements: list[EnergyMeasurement] = []
        self._baseline_energy: float | None = None
        self._step = 0

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def measurements(self) -> list[EnergyMeasurement]:
        return list(self._measurements)

    @property
    def baseline_energy(self) -> float | None:
        return self._baseline_energy

    def measure(self, activations: list[np.ndarray]) -> EnergyMeasurement:
        """Measure energy from layer activations.

        Args:
            activations: List of numpy arrays, one per layer.
                         Each array represents activation values.

        Returns:
            EnergyMeasurement with total and per-layer energies.
        """
        self._step += 1

        per_layer = []
        for act in activations:
            energy = float(np.sqrt(np.sum(act ** 2)))
            per_layer.append(energy)

        total = float(np.mean(per_layer))

        relative = None
        if self._baseline_energy is not None and self._baseline_energy > 0:
            relative = total / self._baseline_energy

        measurement = EnergyMeasurement(
            total_energy=total,
            per_layer_energy=per_layer,
            num_layers=len(activations),
            step=self._step,
            relative_energy=relative,
        )
        self._measurements.append(measurement)
        return measurement

    def set_baseline(self, energy: float | None = None) -> float:
        """Set the baseline energy level.

        Args:
            energy: Explicit baseline value. If None, computed from
                    recent measurements.

        Returns:
            The baseline energy value.
        """
        if energy is not None:
            self._baseline_energy = energy
        elif self._measurements:
            recent = self._measurements[-self._baseline_window:]
            self._baseline_energy = float(np.mean([m.total_energy for m in recent]))
        else:
            self._baseline_energy = 1.0

        return self._baseline_energy

    def get_relative_energy(self) -> float | None:
        """Get the most recent relative energy (vs baseline).

        Returns:
            Ratio of current to baseline energy, or None if no baseline.
        """
        if not self._measurements or self._baseline_energy is None:
            return None

        current = self._measurements[-1].total_energy
        if self._baseline_energy <= 0:
            return None
        return current / self._baseline_energy

    def is_declining(self, threshold: float = 0.1, window: int = 10) -> bool:
        """Check if energy is declining over recent measurements.

        Args:
            threshold: Minimum decline fraction to flag.
            window: Number of recent measurements to consider.

        Returns:
            True if energy has declined by more than threshold.
        """
        if len(self._measurements) < window:
            return False

        recent = [m.total_energy for m in self._measurements[-window:]]
        first_half = np.mean(recent[: window // 2])
        second_half = np.mean(recent[window // 2 :])

        if first_half <= 0:
            return False

        decline = (first_half - second_half) / first_half
        return decline > threshold

    def get_energy_history(self) -> list[float]:
        """Get history of total energy values."""
        return [m.total_energy for m in self._measurements]

    def get_layer_history(self, layer: int) -> list[float]:
        """Get history of energy for a specific layer."""
        return [
            m.per_layer_energy[layer]
            for m in self._measurements
            if layer < len(m.per_layer_energy)
        ]
