from __future__ import annotations

"""Early warning system for energy decline."""

import numpy as np
from dataclasses import dataclass

from .energy_tracker import EnergyMeasurement


@dataclass
class EnergyPrediction:
    """Prediction of future energy levels."""

    predicted_energy: float
    confidence: float  # 0.0 to 1.0
    horizon: int  # steps ahead
    will_decline: bool
    estimated_steps_to_critical: int | None
    trend_slope: float


class EnergyEarlyWarning:
    """Predicts future energy levels to provide early warning of decline.

    Uses simple linear regression on recent energy history to
    project future values and estimate time to critical levels.
    """

    def __init__(self, critical_fraction: float = 0.5):
        self._critical_fraction = critical_fraction
        self._predictions: list[EnergyPrediction] = []

    @property
    def predictions(self) -> list[EnergyPrediction]:
        return list(self._predictions)

    def predict(
        self,
        history: list[EnergyMeasurement],
        horizon: int = 10,
    ) -> EnergyPrediction:
        """Predict energy level at given horizon.

        Args:
            history: List of past energy measurements.
            horizon: Number of steps to predict ahead.

        Returns:
            EnergyPrediction with projected values.
        """
        if len(history) < 3:
            pred = EnergyPrediction(
                predicted_energy=history[-1].total_energy if history else 0.0,
                confidence=0.0,
                horizon=horizon,
                will_decline=False,
                estimated_steps_to_critical=None,
                trend_slope=0.0,
            )
            self._predictions.append(pred)
            return pred

        energies = np.array([m.total_energy for m in history])
        steps = np.arange(len(energies))

        # Linear regression
        slope, intercept = np.polyfit(steps, energies, 1)

        # Predict future energy
        future_step = len(energies) + horizon
        predicted = slope * future_step + intercept

        # Confidence based on R-squared
        residuals = energies - (slope * steps + intercept)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((energies - np.mean(energies)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        confidence = max(0.0, min(1.0, r_squared))

        # Will it decline?
        will_decline = slope < 0 and predicted < energies[-1]

        # Steps to critical level
        baseline = energies[0]
        critical_level = baseline * self._critical_fraction
        steps_to_critical = None
        if slope < 0 and energies[-1] > critical_level:
            # Solve: slope * t + intercept = critical_level
            t = (critical_level - intercept) / slope
            remaining = t - len(energies)
            if remaining > 0:
                steps_to_critical = int(remaining)

        pred = EnergyPrediction(
            predicted_energy=float(predicted),
            confidence=float(confidence),
            horizon=horizon,
            will_decline=will_decline,
            estimated_steps_to_critical=steps_to_critical,
            trend_slope=float(slope),
        )
        self._predictions.append(pred)
        return pred

    def lead_time(self, history: list[EnergyMeasurement]) -> int | None:
        """Estimate how many steps before energy reaches critical level.

        Args:
            history: List of past energy measurements.

        Returns:
            Estimated steps to critical, or None if not declining.
        """
        pred = self.predict(history, horizon=1)
        return pred.estimated_steps_to_critical
