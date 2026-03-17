"""Report generator for risk management.

Generates comprehensive reports combining data from all risk domains.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime


class ReportGenerator:
    """Generates comprehensive risk management reports.

    Combines data from all risk domains into a single report.
    """

    def __init__(self, title: str = "Risk Management Report"):
        self.title = title
        self._sections: List[Dict[str, Any]] = []

    def add_section(
        self,
        heading: str,
        content: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a section to the report.

        Args:
            heading: Section heading.
            content: Section text content.
            data: Optional structured data.
        """
        self._sections.append({
            "heading": heading,
            "content": content,
            "data": data or {},
        })

    def generate(
        self,
        dashboard: Optional[Any] = None,
        incidents: Optional[List[Any]] = None,
        retrospective: Optional[Any] = None,
    ) -> str:
        """Generate the complete report.

        Args:
            dashboard: RiskDashboard object.
            incidents: List of incidents.
            retrospective: RetrospectiveReport.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 70,
            f"  {self.title}",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
        ]

        # Dashboard section
        if dashboard is not None:
            lines.extend([
                "## Overall Risk Status",
                f"   Severity: {getattr(dashboard, 'overall_severity', 'unknown')}",
                f"   Score: {getattr(dashboard, 'overall_score', 0.0):.2f}",
                f"   Critical: {getattr(dashboard, 'critical_count', 0)}",
                f"   High: {getattr(dashboard, 'high_count', 0)}",
                "",
            ])

            for status in getattr(dashboard, "statuses", []):
                lines.append(
                    f"   [{status.risk_id}] {status.name}: "
                    f"{status.severity} ({status.score:.2f})"
                )
            lines.append("")

        # Incidents section
        if incidents:
            lines.extend([
                "## Incidents",
                f"   Total: {len(incidents)}",
                f"   Open: {sum(1 for i in incidents if getattr(i, 'is_open', True))}",
                "",
            ])
            for incident in incidents:
                lines.append(
                    f"   [{getattr(incident, 'incident_id', '?')}] "
                    f"{getattr(incident, 'title', 'Unknown')}: "
                    f"{getattr(incident, 'status', 'unknown')}"
                )
            lines.append("")

        # Custom sections
        for section in self._sections:
            lines.extend([
                f"## {section['heading']}",
                f"   {section['content']}",
                "",
            ])

        # Retrospective
        if retrospective is not None:
            lines.extend([
                "## Retrospective",
                f"   Period: {getattr(retrospective, 'period', 'unknown')}",
                f"   Total Incidents: {getattr(retrospective, 'total_incidents', 0)}",
                f"   Resolution Rate: {getattr(retrospective, 'resolution_rate', 0.0):.0%}",
                "",
            ])
            for insight in getattr(retrospective, "insights", []):
                lines.append(f"   [{insight.severity}] {insight.finding}")
                lines.append(f"     -> {insight.recommendation}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def clear_sections(self) -> None:
        """Clear all added sections."""
        self._sections.clear()
