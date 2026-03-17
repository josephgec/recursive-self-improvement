"""Interpretability dashboard for monitoring."""

from typing import Dict, List, Any, Optional
import numpy as np

from src.monitoring.time_series import InterpretabilityTimeSeries
from src.monitoring.alert_rules import InterpretabilityAlertRules, Alert


class InterpretabilityDashboard:
    """Dashboard that aggregates interpretability monitoring data."""

    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.time_series = InterpretabilityTimeSeries(max_history=history_length)
        self.alert_rules = InterpretabilityAlertRules()
        self._iterations: List[Dict[str, Any]] = []

    def log_iteration(self, iteration: int, data: Dict[str, Any]) -> List[Alert]:
        """Log data for an iteration and return any triggered alerts.

        Args:
            iteration: Iteration number
            data: Dict with interpretability data (divergence, head_tracking, etc.)

        Returns:
            List of triggered alerts
        """
        # Store iteration data
        entry = {"iteration": iteration, **data}
        self._iterations.append(entry)
        if len(self._iterations) > self.history_length:
            self._iterations = self._iterations[-self.history_length:]

        # Record metrics in time series
        metrics = {}
        div = data.get("divergence")
        if isinstance(div, dict):
            metrics["divergence_ratio"] = div.get("divergence_ratio", 0)
            metrics["internal_change"] = div.get("internal_change", 0)
            metrics["behavioral_change"] = div.get("behavioral_change", 0)

        ht = data.get("head_tracking")
        if isinstance(ht, dict):
            metrics["num_dying_heads"] = ht.get("num_dying_heads", 0)
            metrics["num_role_changes"] = ht.get("num_role_changes", 0)

        da = data.get("deceptive_alignment")
        if isinstance(da, dict):
            metrics["monitoring_sensitivity"] = da.get("monitoring_sensitivity", 0)
            metrics["latent_capability_gap"] = da.get("latent_capability_gap", 0)

        if metrics:
            self.time_series.record(iteration, metrics)

        # Evaluate alert rules
        alerts = self.alert_rules.evaluate(data, iteration)
        return alerts

    def get_panels(self) -> Dict[str, Any]:
        """Get dashboard panels for display.

        Returns a dict of panel data for rendering.
        """
        panels = {}

        # Overview panel
        panels["overview"] = {
            "total_iterations": len(self._iterations),
            "total_alerts": len(self.alert_rules.get_alert_history()),
            "critical_alerts": len(self.alert_rules.get_critical_alerts()),
        }

        # Divergence panel
        div_stats = self.time_series.get_metric_stats("divergence_ratio")
        panels["divergence"] = {
            "stats": div_stats,
            "recent": self.time_series.get_window(10, "divergence_ratio"),
        }

        # Head tracking panel
        panels["head_tracking"] = {
            "dying_heads": self.time_series.get_metric_stats("num_dying_heads"),
            "role_changes": self.time_series.get_metric_stats("num_role_changes"),
        }

        # Deceptive alignment panel
        panels["deceptive_alignment"] = {
            "monitoring_sensitivity": self.time_series.get_metric_stats("monitoring_sensitivity"),
            "latent_capability_gap": self.time_series.get_metric_stats("latent_capability_gap"),
        }

        # Alert panel
        recent_alerts = self.alert_rules.get_alert_history()[-10:]
        panels["alerts"] = {
            "recent": [a.to_dict() for a in recent_alerts],
            "rules": self.alert_rules.get_rules(),
        }

        return panels

    def get_summary(self) -> str:
        """Get a text summary of the dashboard state."""
        panels = self.get_panels()
        lines = []
        lines.append("=== Interpretability Dashboard ===")
        lines.append(f"Iterations: {panels['overview']['total_iterations']}")
        lines.append(f"Total alerts: {panels['overview']['total_alerts']}")
        lines.append(f"Critical alerts: {panels['overview']['critical_alerts']}")

        div = panels["divergence"]["stats"]
        if div["count"] > 0:
            lines.append(f"Divergence ratio: mean={div['mean']:.3f}, max={div['max']:.3f}")

        return "\n".join(lines)
