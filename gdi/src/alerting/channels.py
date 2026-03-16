"""Alert channels for GDI notifications."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .alert_manager import Alert

logger = logging.getLogger(__name__)


class AlertChannel(ABC):
    """Abstract base class for alert channels."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send an alert through this channel.

        Args:
            alert: The alert to send.

        Returns:
            True if successfully sent.
        """
        ...


class LogChannel(AlertChannel):
    """Alert channel that logs to Python logging."""

    def __init__(self, logger_name: str = "gdi.alerts"):
        self._logger = logging.getLogger(logger_name)
        self._sent: List[Alert] = []

    def send(self, alert: Alert) -> bool:
        """Log the alert."""
        level_map = {
            "yellow": logging.WARNING,
            "orange": logging.ERROR,
            "red": logging.CRITICAL,
        }
        log_level = level_map.get(alert.level, logging.INFO)
        self._logger.log(log_level, alert.message)
        self._sent.append(alert)
        return True

    @property
    def sent_alerts(self) -> List[Alert]:
        """Get list of sent alerts."""
        return self._sent


class WandBChannel(AlertChannel):
    """Mock W&B alert channel for testing.

    In production, this would integrate with Weights & Biases.
    """

    def __init__(self, project: str = "gdi", run_name: str = "default"):
        self.project = project
        self.run_name = run_name
        self._sent: List[Alert] = []

    def send(self, alert: Alert) -> bool:
        """Mock send to W&B."""
        self._sent.append(alert)
        # In production: wandb.alert(title=alert.level, text=alert.message)
        return True

    @property
    def sent_alerts(self) -> List[Alert]:
        """Get list of sent alerts."""
        return self._sent


class WebhookChannel(AlertChannel):
    """Mock webhook alert channel for testing.

    In production, this would send HTTP POST to a webhook URL.
    """

    def __init__(self, url: str = "https://hooks.example.com/gdi"):
        self.url = url
        self._sent: List[Alert] = []

    def send(self, alert: Alert) -> bool:
        """Mock send to webhook."""
        payload = {
            "level": alert.level,
            "score": alert.score,
            "iteration": alert.iteration,
            "message": alert.message,
            "action": alert.action,
        }
        self._sent.append(alert)
        # In production: requests.post(self.url, json=payload)
        return True

    @property
    def sent_alerts(self) -> List[Alert]:
        """Get list of sent alerts."""
        return self._sent
