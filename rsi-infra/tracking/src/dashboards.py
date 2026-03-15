"""Dashboard configuration data structures.

These produce config dicts that *could* be used to configure W&B dashboards
programmatically — no actual W&B API calls are made here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PanelDef:
    """Definition of a single dashboard panel."""

    title: str
    metric_keys: list[str]
    panel_type: str = "line"  # "line" | "bar" | "scatter" | "scalar"
    section: str = "default"


@dataclass
class DashboardConfig:
    """Collection of panels forming a dashboard."""

    name: str
    description: str = ""
    panels: list[PanelDef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON / YAML."""
        return {
            "name": self.name,
            "description": self.description,
            "panels": [
                {
                    "title": p.title,
                    "metric_keys": p.metric_keys,
                    "panel_type": p.panel_type,
                    "section": p.section,
                }
                for p in self.panels
            ],
        }


# ---------------------------------------------------------------------------
# Pre-defined dashboards
# ---------------------------------------------------------------------------

def get_safety_dashboard() -> dict[str, Any]:
    """Return a dashboard config focused on safety metrics."""
    cfg = DashboardConfig(
        name="Safety Dashboard",
        description="Goal-drift, constraint violations, and alerts.",
        panels=[
            PanelDef(
                title="Goal Drift Index",
                metric_keys=["safety/goal_drift_index"],
                panel_type="line",
                section="drift",
            ),
            PanelDef(
                title="Semantic Drift",
                metric_keys=["safety/semantic_drift"],
                panel_type="line",
                section="drift",
            ),
            PanelDef(
                title="Lexical Drift",
                metric_keys=["safety/lexical_drift"],
                panel_type="line",
                section="drift",
            ),
            PanelDef(
                title="Structural Drift",
                metric_keys=["safety/structural_drift"],
                panel_type="line",
                section="drift",
            ),
            PanelDef(
                title="Distributional Drift (KL)",
                metric_keys=["safety/distributional_drift"],
                panel_type="line",
                section="drift",
            ),
            PanelDef(
                title="Constraint Violations",
                metric_keys=["safety/all_passed", "safety/recommendation"],
                panel_type="scalar",
                section="constraints",
            ),
            PanelDef(
                title="Alert Count",
                metric_keys=["safety/alert_count"],
                panel_type="bar",
                section="alerts",
            ),
        ],
    )
    return cfg.to_dict()


def get_collapse_dashboard() -> dict[str, Any]:
    """Return a dashboard config focused on collapse / mode-collapse indicators."""
    cfg = DashboardConfig(
        name="Collapse Dashboard",
        description="Entropy, perplexity, and distributional health metrics.",
        panels=[
            PanelDef(
                title="Token Entropy",
                metric_keys=["collapse/token_entropy"],
                panel_type="line",
                section="entropy",
            ),
            PanelDef(
                title="Entropy Drop",
                metric_keys=["collapse/entropy_drop"],
                panel_type="line",
                section="entropy",
            ),
            PanelDef(
                title="Perplexity",
                metric_keys=["training/perplexity"],
                panel_type="line",
                section="health",
            ),
            PanelDef(
                title="Loss Curve",
                metric_keys=["training/loss"],
                panel_type="line",
                section="health",
            ),
            PanelDef(
                title="Vocabulary Usage",
                metric_keys=["collapse/unique_tokens", "collapse/top_k_concentration"],
                panel_type="line",
                section="vocabulary",
            ),
        ],
    )
    return cfg.to_dict()


def get_overview_dashboard() -> dict[str, Any]:
    """Return an overview dashboard combining key training and safety metrics."""
    cfg = DashboardConfig(
        name="Overview Dashboard",
        description="High-level view of training progress and safety status.",
        panels=[
            PanelDef(
                title="Loss",
                metric_keys=["training/loss"],
                panel_type="line",
                section="training",
            ),
            PanelDef(
                title="Accuracy",
                metric_keys=["training/accuracy"],
                panel_type="line",
                section="training",
            ),
            PanelDef(
                title="Goal Drift Index",
                metric_keys=["safety/goal_drift_index"],
                panel_type="line",
                section="safety",
            ),
            PanelDef(
                title="Capability-Alignment Ratio",
                metric_keys=["safety/car"],
                panel_type="line",
                section="safety",
            ),
            PanelDef(
                title="Safety Score",
                metric_keys=["training/safety_score"],
                panel_type="line",
                section="safety",
            ),
            PanelDef(
                title="Constraint Status",
                metric_keys=["safety/all_passed"],
                panel_type="scalar",
                section="safety",
            ),
        ],
    )
    return cfg.to_dict()
