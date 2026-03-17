"""Risk retrospective analysis.

Analyzes historical risk data to identify patterns, lessons learned,
and recommendations for improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrospectiveInsight:
    """A single insight from retrospective analysis."""
    category: str
    finding: str
    severity: str
    recommendation: str


@dataclass
class RetrospectiveReport:
    """Complete retrospective report."""
    period: str
    total_incidents: int
    resolved_incidents: int
    insights: List[RetrospectiveInsight] = field(default_factory=list)
    risk_scores_over_time: List[float] = field(default_factory=list)
    domain_breakdown: Dict[str, int] = field(default_factory=dict)

    @property
    def resolution_rate(self) -> float:
        if self.total_incidents == 0:
            return 1.0
        return self.resolved_incidents / self.total_incidents


class RiskRetrospective:
    """Performs retrospective analysis on risk management data.

    Analyzes incidents, risk scores, and mitigation effectiveness
    to generate actionable insights.
    """

    def analyze(
        self,
        incidents: List[Dict[str, Any]],
        risk_scores: List[float],
        period: str = "current",
    ) -> RetrospectiveReport:
        """Run retrospective analysis.

        Args:
            incidents: List of incident dicts with 'domain', 'severity', 'status'.
            risk_scores: List of risk scores over time.
            period: Description of the analysis period.

        Returns:
            RetrospectiveReport with insights.
        """
        total = len(incidents)
        resolved = sum(1 for i in incidents if i.get("status") in ("resolved", "closed"))

        # Domain breakdown
        domain_counts: Dict[str, int] = {}
        for incident in incidents:
            domain = incident.get("domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Generate insights
        insights = []

        # Check for recurring domains
        if domain_counts:
            worst_domain = max(domain_counts, key=domain_counts.get)  # type: ignore[arg-type]
            if domain_counts[worst_domain] > 1:
                insights.append(RetrospectiveInsight(
                    category="recurring_risk",
                    finding=f"Domain '{worst_domain}' had {domain_counts[worst_domain]} incidents",
                    severity="medium",
                    recommendation=f"Strengthen mitigations in {worst_domain} domain",
                ))

        # Check risk score trend
        if len(risk_scores) >= 3:
            first_third = sum(risk_scores[:len(risk_scores)//3]) / (len(risk_scores)//3)
            last_third = risk_scores[-(len(risk_scores)//3):]
            last_avg = sum(last_third) / len(last_third) if last_third else 0
            if last_avg > first_third * 1.2:
                insights.append(RetrospectiveInsight(
                    category="trend",
                    finding="Risk scores are trending upward",
                    severity="high",
                    recommendation="Review and strengthen risk mitigations across all domains",
                ))
            elif last_avg < first_third * 0.8:
                insights.append(RetrospectiveInsight(
                    category="trend",
                    finding="Risk scores are trending downward (improving)",
                    severity="low",
                    recommendation="Continue current risk management practices",
                ))

        # Check resolution rate
        resolution_rate = resolved / total if total > 0 else 1.0
        if resolution_rate < 0.5:
            insights.append(RetrospectiveInsight(
                category="resolution",
                finding=f"Low incident resolution rate: {resolution_rate:.0%}",
                severity="high",
                recommendation="Allocate more resources to incident resolution",
            ))

        # Check for critical incidents
        critical_count = sum(1 for i in incidents if i.get("severity") == "critical")
        if critical_count > 0:
            insights.append(RetrospectiveInsight(
                category="severity",
                finding=f"{critical_count} critical incidents occurred",
                severity="critical",
                recommendation="Review critical incident prevention measures",
            ))

        return RetrospectiveReport(
            period=period,
            total_incidents=total,
            resolved_incidents=resolved,
            insights=insights,
            risk_scores_over_time=risk_scores,
            domain_breakdown=domain_counts,
        )
