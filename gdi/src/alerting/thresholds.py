"""Threshold configuration for GDI alerting."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ThresholdConfig:
    """Configuration for GDI alert thresholds."""
    green_max: float = 0.15
    yellow_max: float = 0.40
    orange_max: float = 0.70
    red_min: float = 0.70

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ThresholdConfig":
        """Create ThresholdConfig from a config dictionary.

        Args:
            config: Dictionary with optional 'thresholds' key.

        Returns:
            ThresholdConfig instance.
        """
        thresholds = config.get("thresholds", {})
        return cls(
            green_max=thresholds.get("green_max", 0.15),
            yellow_max=thresholds.get("yellow_max", 0.40),
            orange_max=thresholds.get("orange_max", 0.70),
            red_min=thresholds.get("red_min", 0.70),
        )

    @classmethod
    def from_calibration(
        cls, calibrated: Any
    ) -> "ThresholdConfig":
        """Create ThresholdConfig from calibrated thresholds.

        Args:
            calibrated: CalibratedThresholds instance.

        Returns:
            ThresholdConfig instance.
        """
        return cls(
            green_max=calibrated.green_max,
            yellow_max=calibrated.yellow_max,
            orange_max=calibrated.orange_max,
            red_min=calibrated.red_min,
        )

    def get_level(self, score: float) -> str:
        """Determine alert level from score.

        Args:
            score: GDI composite score.

        Returns:
            Alert level string.
        """
        if score <= self.green_max:
            return "green"
        elif score <= self.yellow_max:
            return "yellow"
        elif score <= self.orange_max:
            return "orange"
        else:
            return "red"
