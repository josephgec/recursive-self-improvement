"""Time series storage for GDI history."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..composite.gdi import GDIResult


class GDITimeSeries:
    """Records and retrieves GDI score history.

    Persists data as JSON for simplicity.
    """

    def __init__(self, store_path: Optional[str] = None):
        """Initialize time series.

        Args:
            store_path: Optional path for JSON persistence.
        """
        self.store_path = store_path
        self._history: List[Dict[str, Any]] = []

        if store_path and os.path.exists(store_path):
            self._load()

    def record(self, result: GDIResult, iteration: Optional[int] = None) -> None:
        """Record a GDI result.

        Args:
            result: GDI computation result.
            iteration: Optional iteration number.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration or len(self._history),
            "composite_score": result.composite_score,
            "alert_level": result.alert_level,
            "trend": result.trend,
            "semantic_score": result.semantic_score,
            "lexical_score": result.lexical_score,
            "structural_score": result.structural_score,
            "distributional_score": result.distributional_score,
        }
        self._history.append(entry)

        if self.store_path:
            self._save()

    def get_history(self) -> List[Dict[str, Any]]:
        """Get full history.

        Returns:
            List of recorded entries.
        """
        return list(self._history)

    def get_window(self, window_size: int) -> List[Dict[str, Any]]:
        """Get the most recent entries.

        Args:
            window_size: Number of recent entries to return.

        Returns:
            List of recent entries.
        """
        return list(self._history[-window_size:])

    def get_scores(self) -> List[float]:
        """Get just the composite scores as a list."""
        return [e["composite_score"] for e in self._history]

    def export(self) -> List[Dict[str, Any]]:
        """Export full history as serializable data.

        Returns:
            List of entry dictionaries.
        """
        return list(self._history)

    def _save(self) -> None:
        """Save history to JSON file."""
        if self.store_path:
            parent = os.path.dirname(self.store_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.store_path, "w") as f:
                json.dump(self._history, f, indent=2)

    def _load(self) -> None:
        """Load history from JSON file."""
        if self.store_path and os.path.exists(self.store_path):
            with open(self.store_path, "r") as f:
                self._history = json.load(f)
