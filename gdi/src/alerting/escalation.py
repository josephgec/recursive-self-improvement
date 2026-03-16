"""Escalation policies for GDI alerts."""

from typing import List, Optional

from ..composite.gdi import GDIResult


class EscalationPolicy:
    """Determines actions based on alert severity and history.

    Escalation levels:
    - Yellow: log only
    - Orange: alert (notify humans)
    - Red: pause execution
    - 3 consecutive reds: emergency stop
    """

    def __init__(self, consecutive_red_limit: int = 3):
        """Initialize escalation policy.

        Args:
            consecutive_red_limit: Number of consecutive red alerts
                                   before emergency stop.
        """
        self.consecutive_red_limit = consecutive_red_limit
        self._consecutive_reds = 0
        self._history: List[str] = []

    def get_action(self, alert_level: str) -> str:
        """Get the action for a given alert level.

        Args:
            alert_level: One of "green", "yellow", "orange", "red".

        Returns:
            Action string: "none", "log", "alert", "pause", "emergency_stop".
        """
        if alert_level == "red":
            self._consecutive_reds += 1
        else:
            self._consecutive_reds = 0

        self._history.append(alert_level)

        if self._consecutive_reds >= self.consecutive_red_limit:
            return "emergency_stop"

        action_map = {
            "green": "none",
            "yellow": "log",
            "orange": "alert",
            "red": "pause",
        }
        return action_map.get(alert_level, "log")

    def should_pause(self, alert_level: str) -> bool:
        """Check if execution should be paused.

        Pauses on red alert or higher.
        """
        return alert_level == "red"

    def should_rollback(self, alert_level: str, score: float = 0.0) -> bool:
        """Check if a rollback should be triggered.

        Rollback on red with high score.
        """
        return alert_level == "red" and score >= 0.85

    def should_emergency_stop(self) -> bool:
        """Check if emergency stop should be triggered.

        Triggers after consecutive_red_limit consecutive red alerts.
        """
        return self._consecutive_reds >= self.consecutive_red_limit

    def reset(self) -> None:
        """Reset escalation state."""
        self._consecutive_reds = 0
        self._history = []
