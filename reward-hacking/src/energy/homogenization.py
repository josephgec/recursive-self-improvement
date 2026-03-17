from __future__ import annotations

"""Homogenization detection in activation energy patterns."""

import numpy as np
from dataclasses import dataclass

from .energy_tracker import EnergyMeasurement


@dataclass
class HomogenizationResult:
    """Result of homogenization detection."""

    is_homogenizing: bool
    patterns_detected: list[str]
    severity: float  # 0.0 to 1.0
    details: dict


class HomogenizationDetector:
    """Detects homogenization patterns in activation energy.

    Patterns detected:
    - uniform_decline: All layers declining uniformly
    - final_layer_collapse: Last layer energy drops sharply
    - sudden_drop: Abrupt energy decrease across layers
    - oscillation: Energy oscillating without convergence
    """

    def __init__(
        self,
        decline_threshold: float = 0.2,
        collapse_threshold: float = 0.5,
        drop_threshold: float = 0.3,
        oscillation_threshold: float = 0.15,
    ):
        self._decline_threshold = decline_threshold
        self._collapse_threshold = collapse_threshold
        self._drop_threshold = drop_threshold
        self._oscillation_threshold = oscillation_threshold

    def detect(self, history: list[EnergyMeasurement]) -> HomogenizationResult:
        """Run all homogenization detection patterns.

        Args:
            history: List of EnergyMeasurements over time.

        Returns:
            HomogenizationResult with detected patterns.
        """
        if len(history) < 5:
            return HomogenizationResult(
                is_homogenizing=False,
                patterns_detected=[],
                severity=0.0,
                details={"reason": "insufficient_data"},
            )

        patterns = []
        details = {}
        severities = []

        # Check each pattern
        uniform = self._check_uniform_decline(history)
        if uniform[0]:
            patterns.append("uniform_decline")
            details["uniform_decline"] = uniform[1]
            severities.append(uniform[2])

        collapse = self._check_final_layer_collapse(history)
        if collapse[0]:
            patterns.append("final_layer_collapse")
            details["final_layer_collapse"] = collapse[1]
            severities.append(collapse[2])

        drop = self._check_sudden_drop(history)
        if drop[0]:
            patterns.append("sudden_drop")
            details["sudden_drop"] = drop[1]
            severities.append(drop[2])

        oscillation = self._check_oscillation(history)
        if oscillation[0]:
            patterns.append("oscillation")
            details["oscillation"] = oscillation[1]
            severities.append(oscillation[2])

        severity = float(np.mean(severities)) if severities else 0.0

        return HomogenizationResult(
            is_homogenizing=len(patterns) > 0,
            patterns_detected=patterns,
            severity=severity,
            details=details,
        )

    def _check_uniform_decline(
        self, history: list[EnergyMeasurement]
    ) -> tuple[bool, dict, float]:
        """Check if all layers are declining uniformly."""
        energies = [m.total_energy for m in history]
        first_half = np.mean(energies[: len(energies) // 2])
        second_half = np.mean(energies[len(energies) // 2 :])

        if first_half <= 0:
            return False, {}, 0.0

        decline_rate = (first_half - second_half) / first_half

        # Check per-layer uniformity
        num_layers = len(history[0].per_layer_energy)
        layer_declines = []
        for layer in range(num_layers):
            layer_energies = [
                m.per_layer_energy[layer]
                for m in history
                if layer < len(m.per_layer_energy)
            ]
            if len(layer_energies) < 4:
                continue
            first = np.mean(layer_energies[: len(layer_energies) // 2])
            second = np.mean(layer_energies[len(layer_energies) // 2 :])
            if first > 0:
                layer_declines.append((first - second) / first)

        is_uniform = (
            decline_rate > self._decline_threshold
            and len(layer_declines) > 0
            and np.std(layer_declines) < 0.1
        )

        info = {
            "overall_decline_rate": float(decline_rate),
            "layer_decline_std": float(np.std(layer_declines)) if layer_declines else 0.0,
        }
        severity = min(float(decline_rate / self._decline_threshold), 1.0) if is_uniform else 0.0
        return is_uniform, info, severity

    def _check_final_layer_collapse(
        self, history: list[EnergyMeasurement]
    ) -> tuple[bool, dict, float]:
        """Check if the final layer energy is collapsing."""
        num_layers = len(history[0].per_layer_energy)
        if num_layers < 2:
            return False, {}, 0.0

        final_layer = num_layers - 1
        final_energies = [
            m.per_layer_energy[final_layer]
            for m in history
            if final_layer < len(m.per_layer_energy)
        ]

        if len(final_energies) < 4:
            return False, {}, 0.0

        first = np.mean(final_energies[: len(final_energies) // 2])
        second = np.mean(final_energies[len(final_energies) // 2 :])

        if first <= 0:
            return False, {}, 0.0

        collapse_rate = (first - second) / first
        is_collapsing = collapse_rate > self._collapse_threshold

        # Compare to other layers
        other_rates = []
        for layer in range(num_layers - 1):
            layer_energies = [
                m.per_layer_energy[layer]
                for m in history
                if layer < len(m.per_layer_energy)
            ]
            if len(layer_energies) >= 4:
                f = np.mean(layer_energies[: len(layer_energies) // 2])
                s = np.mean(layer_energies[len(layer_energies) // 2 :])
                if f > 0:
                    other_rates.append((f - s) / f)

        info = {
            "final_layer_collapse_rate": float(collapse_rate),
            "other_layers_mean_decline": float(np.mean(other_rates)) if other_rates else 0.0,
        }
        severity = min(float(collapse_rate / self._collapse_threshold), 1.0) if is_collapsing else 0.0
        return is_collapsing, info, severity

    def _check_sudden_drop(
        self, history: list[EnergyMeasurement]
    ) -> tuple[bool, dict, float]:
        """Check for sudden energy drops between consecutive measurements."""
        energies = [m.total_energy for m in history]

        max_drop = 0.0
        drop_step = -1
        for i in range(1, len(energies)):
            if energies[i - 1] > 0:
                drop = (energies[i - 1] - energies[i]) / energies[i - 1]
                if drop > max_drop:
                    max_drop = drop
                    drop_step = i

        is_sudden = max_drop > self._drop_threshold

        info = {
            "max_drop": float(max_drop),
            "drop_step": drop_step,
        }
        severity = min(float(max_drop / self._drop_threshold), 1.0) if is_sudden else 0.0
        return is_sudden, info, severity

    def _check_oscillation(
        self, history: list[EnergyMeasurement]
    ) -> tuple[bool, dict, float]:
        """Check for energy oscillation patterns."""
        energies = [m.total_energy for m in history]

        if len(energies) < 6:
            return False, {}, 0.0

        # Count sign changes in differences
        diffs = np.diff(energies)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        max_possible_changes = len(diffs) - 1
        oscillation_rate = sign_changes / max(max_possible_changes, 1)

        # Compute coefficient of variation
        mean_energy = np.mean(energies)
        cv = float(np.std(energies) / mean_energy) if mean_energy > 0 else 0.0

        is_oscillating = oscillation_rate > 0.6 and cv > self._oscillation_threshold

        info = {
            "oscillation_rate": float(oscillation_rate),
            "coefficient_of_variation": cv,
            "sign_changes": int(sign_changes),
        }
        severity = min(float(oscillation_rate), 1.0) if is_oscillating else 0.0
        return is_oscillating, info, severity
