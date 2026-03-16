"""Dashboard config: data structures for monitoring dashboard."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PanelConfig:
    """Configuration for a single dashboard panel."""
    name: str = ""
    panel_type: str = "line_chart"  # line_chart, gauge, table, text
    data_source: str = ""
    refresh_interval: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


class DashboardConfig:
    """Configuration for the monitoring dashboard (data structure, no actual UI)."""

    def __init__(self):
        self._panels: List[PanelConfig] = self._default_panels()

    def get_panels(self) -> List[PanelConfig]:
        """Get all dashboard panel configurations."""
        return list(self._panels)

    def add_panel(self, panel: PanelConfig) -> None:
        """Add a panel to the dashboard."""
        self._panels.append(panel)

    def remove_panel(self, name: str) -> bool:
        """Remove a panel by name."""
        before = len(self._panels)
        self._panels = [p for p in self._panels if p.name != name]
        return len(self._panels) < before

    @staticmethod
    def _default_panels() -> List[PanelConfig]:
        """Create default dashboard panels."""
        return [
            PanelConfig(
                name="accuracy_curve",
                panel_type="line_chart",
                data_source="improvement_curve",
            ),
            PanelConfig(
                name="gdi_gauge",
                panel_type="gauge",
                data_source="gdi_monitor",
            ),
            PanelConfig(
                name="car_gauge",
                panel_type="gauge",
                data_source="car_tracker",
            ),
            PanelConfig(
                name="safety_status",
                panel_type="table",
                data_source="constraint_enforcer",
            ),
            PanelConfig(
                name="iteration_log",
                panel_type="table",
                data_source="iteration_logger",
            ),
            PanelConfig(
                name="modification_history",
                panel_type="table",
                data_source="audit_bridge",
            ),
        ]
