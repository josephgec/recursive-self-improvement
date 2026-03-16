"""Alert management for GDI drift detection."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..composite.gdi import GDIResult
from .thresholds import ThresholdConfig


@dataclass
class Alert:
    """A drift alert."""
    level: str
    score: float
    iteration: int
    message: str
    action: str = "log"
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Manages GDI alerts with deduplication.

    Processes GDI results and generates alerts, avoiding
    duplicate alerts for the same level.
    """

    def __init__(
        self,
        threshold_config: Optional[ThresholdConfig] = None,
        channels: Optional[List[Any]] = None,
    ):
        """Initialize alert manager.

        Args:
            threshold_config: Alert threshold configuration.
            channels: List of alert channels to notify.
        """
        self.threshold_config = threshold_config or ThresholdConfig()
        self.channels = channels or []
        self._last_level: Optional[str] = None
        self._alert_history: List[Alert] = []

    def process(
        self,
        gdi_result: GDIResult,
        iteration: int,
    ) -> Optional[Alert]:
        """Process a GDI result and generate an alert if needed.

        Deduplicates: won't re-alert for the same level.

        Args:
            gdi_result: The GDI computation result.
            iteration: Current iteration number.

        Returns:
            Alert if generated, None if deduplicated.
        """
        level = gdi_result.alert_level

        # Deduplication: don't re-alert same level
        if level == self._last_level and level != "red":
            return None

        # Green doesn't generate alerts
        if level == "green":
            self._last_level = level
            return None

        action_map = {
            "yellow": "log",
            "orange": "alert",
            "red": "pause",
        }

        alert = Alert(
            level=level,
            score=gdi_result.composite_score,
            iteration=iteration,
            message=self._format_message(gdi_result, iteration),
            action=action_map.get(level, "log"),
        )

        self._alert_history.append(alert)
        self._last_level = level

        # Send to channels
        for channel in self.channels:
            channel.send(alert)

        return alert

    def _format_message(
        self, gdi_result: GDIResult, iteration: int
    ) -> str:
        """Format alert message."""
        return (
            f"GDI Alert [{gdi_result.alert_level.upper()}] at iteration {iteration}: "
            f"score={gdi_result.composite_score:.3f}, "
            f"trend={gdi_result.trend}"
        )

    @property
    def alert_history(self) -> List[Alert]:
        """Get alert history."""
        return self._alert_history

    def reset(self) -> None:
        """Reset alert state."""
        self._last_level = None
        self._alert_history = []
