"""Unified risk dashboard for cross-domain risk visualization.

Computes aggregated risk metrics and generates stakeholder reports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.orchestration.risk_registry import RiskDashboard, RiskRegistry, RiskStatus


class UnifiedRiskDashboard:
    """Unified dashboard for cross-domain risk management.

    Aggregates risk data from all domains and generates
    stakeholder-friendly reports.
    """

    def __init__(self, registry: RiskRegistry):
        self.registry = registry
        self._snapshots: List[RiskDashboard] = []

    def compute(self) -> RiskDashboard:
        """Compute current risk dashboard.

        Returns:
            RiskDashboard with all current risk statuses.
        """
        dashboard = self.registry.check_all()
        self._snapshots.append(dashboard)
        return dashboard

    def generate_stakeholder_report(
        self,
        dashboard: Optional[RiskDashboard] = None,
    ) -> str:
        """Generate a human-readable stakeholder report.

        Args:
            dashboard: Dashboard to report on. If None, computes fresh.

        Returns:
            Formatted report string.
        """
        if dashboard is None:
            dashboard = self.compute()

        lines = [
            "=" * 60,
            "RISK MANAGEMENT DASHBOARD - STAKEHOLDER REPORT",
            "=" * 60,
            "",
            f"Overall Severity: {dashboard.overall_severity.upper()}",
            f"Overall Risk Score: {dashboard.overall_score:.2f}",
            f"Total Risks Monitored: {dashboard.total_risks}",
            f"Critical Risks: {dashboard.critical_count}",
            f"High Risks: {dashboard.high_count}",
            "",
            "-" * 60,
            "RISK DETAILS",
            "-" * 60,
        ]

        for status in dashboard.statuses:
            severity_marker = self._severity_marker(status.severity)
            lines.append(
                f"\n{severity_marker} [{status.risk_id}] {status.name} "
                f"({status.domain})"
            )
            lines.append(f"   Severity: {status.severity}")
            lines.append(f"   Score: {status.score:.2f}")
            if status.mitigations_active:
                lines.append(
                    f"   Active Mitigations: {', '.join(status.mitigations_active)}"
                )
            if status.details:
                for key, value in status.details.items():
                    lines.append(f"   {key}: {value}")

        # Summary
        lines.extend([
            "",
            "-" * 60,
            "RECOMMENDATIONS",
            "-" * 60,
        ])

        if dashboard.critical_count > 0:
            lines.append("  [!] IMMEDIATE ACTION REQUIRED on critical risks")
        if dashboard.high_count > 0:
            lines.append("  [!] Review high-severity risks promptly")
        if dashboard.critical_count == 0 and dashboard.high_count == 0:
            lines.append("  All risks within acceptable parameters")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)

    def _severity_marker(self, severity: str) -> str:
        """Return a text marker for severity level."""
        markers = {
            "critical": "[!!!]",
            "high": "[!! ]",
            "medium": "[!  ]",
            "low": "[   ]",
        }
        return markers.get(severity, "[   ]")

    def get_snapshots(self) -> List[RiskDashboard]:
        """Return historical dashboard snapshots."""
        return list(self._snapshots)

    def get_trend(self) -> Dict[str, Any]:
        """Analyze risk trends across snapshots.

        Returns:
            Dict with trend information.
        """
        if len(self._snapshots) < 2:
            return {"trend": "insufficient_data", "snapshots": len(self._snapshots)}

        recent = self._snapshots[-1]
        previous = self._snapshots[-2]

        score_delta = recent.overall_score - previous.overall_score
        critical_delta = recent.critical_count - previous.critical_count

        if score_delta > 0.1:
            trend = "worsening"
        elif score_delta < -0.1:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "score_delta": score_delta,
            "critical_delta": critical_delta,
            "current_score": recent.overall_score,
            "previous_score": previous.overall_score,
        }
