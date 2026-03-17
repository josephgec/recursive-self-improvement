from __future__ import annotations

"""GDI (Governance, Deployment, Impact) summary packaging."""

from dataclasses import dataclass, field


@dataclass
class GDISummary:
    """Summary of GDI track status."""

    status: str  # "green", "yellow", "red"
    governance_compliant: bool = True
    deployment_ready: bool = True
    impact_assessed: bool = True
    issues: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def is_green(self) -> bool:
        return self.status == "green"


def package_gdi(history: dict) -> GDISummary:
    """Package GDI track summary from training history.

    Args:
        history: Dictionary with training history data.

    Returns:
        GDISummary with assessment.
    """
    issues = []
    details = {}

    # Check governance
    governance_ok = history.get("governance_review", True)
    if not governance_ok:
        issues.append("Governance review incomplete")

    # Check deployment readiness
    deployment_ok = history.get("deployment_checks_passed", True)
    if not deployment_ok:
        issues.append("Deployment checks failed")

    # Check impact assessment
    impact_ok = history.get("impact_assessed", True)
    if not impact_ok:
        issues.append("Impact assessment incomplete")

    # Check for unresolved failures
    unresolved = history.get("unresolved_failures", [])
    if unresolved:
        issues.extend([f"Unresolved: {f}" for f in unresolved])

    # Determine status
    if issues:
        if any("Unresolved" in i for i in issues):
            status = "red"
        else:
            status = "yellow"
    else:
        status = "green"

    details = {
        "governance_review": governance_ok,
        "deployment_checks": deployment_ok,
        "impact_assessment": impact_ok,
        "num_issues": len(issues),
    }

    return GDISummary(
        status=status,
        governance_compliant=governance_ok,
        deployment_ready=deployment_ok,
        impact_assessed=impact_ok,
        issues=issues,
        details=details,
    )
