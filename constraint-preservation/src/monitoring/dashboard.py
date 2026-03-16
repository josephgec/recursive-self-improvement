"""DashboardConfig: configuration for monitoring dashboards."""

from __future__ import annotations

from typing import Any, Dict, List


class DashboardConfig:
    """Provides panel configurations for constraint monitoring dashboards."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def get_constraint_panels(self) -> List[Dict[str, Any]]:
        """Return panel definitions for each constraint category."""
        return [
            {
                "title": "Quality Constraints",
                "category": "quality",
                "metrics": [
                    "accuracy_floor",
                    "entropy_floor",
                    "drift_ceiling",
                    "regression_guard",
                    "consistency",
                    "latency_ceiling",
                ],
                "chart_type": "bar",
                "show_threshold": True,
                "show_headroom": True,
            },
            {
                "title": "Safety Constraints",
                "category": "safety",
                "metrics": ["safety_eval"],
                "chart_type": "gauge",
                "show_threshold": True,
                "show_headroom": True,
            },
            {
                "title": "Headroom Trends",
                "category": "all",
                "metrics": ["all"],
                "chart_type": "line",
                "show_threshold": False,
                "show_headroom": True,
            },
            {
                "title": "Violation History",
                "category": "all",
                "metrics": ["violations"],
                "chart_type": "timeline",
                "show_threshold": False,
                "show_headroom": False,
            },
        ]
