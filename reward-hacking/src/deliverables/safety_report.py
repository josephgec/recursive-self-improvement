from __future__ import annotations

"""Safety report generation from safety packages."""

from .phase_gate import SafetyPackage


def generate_safety_report(package: SafetyPackage) -> str:
    """Generate a markdown safety report from a safety package.

    Produces an 8-section report covering all tracks and
    cross-signal analysis.

    Args:
        package: Completed safety package.

    Returns:
        Markdown string with the full report.
    """
    sections = []

    # Section 1: Executive Summary
    sections.append("# Safety Report")
    sections.append("")
    sections.append("## 1. Executive Summary")
    sections.append("")
    sections.append(f"**Phase:** {package.phase}")
    sections.append(f"**Iteration Range:** {package.iteration_range[0]}-{package.iteration_range[1]}")
    sections.append(f"**Overall Status:** {'PASS' if package.all_green else 'FAIL'}")
    sections.append(f"**Summary:** {package.summary}")
    sections.append("")

    # Section 2: GDI Track
    sections.append("## 2. GDI Track")
    sections.append("")
    sections.append(f"**Status:** {package.gdi.status.upper()}")
    sections.append(f"- Governance Compliant: {'Yes' if package.gdi.governance_compliant else 'No'}")
    sections.append(f"- Deployment Ready: {'Yes' if package.gdi.deployment_ready else 'No'}")
    sections.append(f"- Impact Assessed: {'Yes' if package.gdi.impact_assessed else 'No'}")
    if package.gdi.issues:
        sections.append("- Issues:")
        for issue in package.gdi.issues:
            sections.append(f"  - {issue}")
    sections.append("")

    # Section 3: Constraint Track
    sections.append("## 3. Constraint Track")
    sections.append("")
    sections.append(f"**Status:** {package.constraint.status.upper()}")
    sections.append(
        f"- Constraints Met: {package.constraint.constraints_met}"
        f"/{package.constraint.constraints_total}"
    )
    sections.append(f"- Reward Bounded: {'Yes' if package.constraint.reward_bounded else 'No'}")
    sections.append(f"- Entropy Maintained: {'Yes' if package.constraint.entropy_maintained else 'No'}")
    sections.append(f"- Energy Stable: {'Yes' if package.constraint.energy_stable else 'No'}")
    if package.constraint.issues:
        sections.append("- Issues:")
        for issue in package.constraint.issues:
            sections.append(f"  - {issue}")
    sections.append("")

    # Section 4: Interpretability Track
    sections.append("## 4. Interpretability Track")
    sections.append("")
    sections.append(f"**Status:** {package.interp.status.upper()}")
    sections.append(f"- Energy Interpretable: {'Yes' if package.interp.energy_interpretable else 'No'}")
    sections.append(f"- Homogenization Checked: {'Yes' if package.interp.homogenization_checked else 'No'}")
    sections.append(f"- Activations Tracked: {'Yes' if package.interp.activations_tracked else 'No'}")
    if package.interp.issues:
        sections.append("- Issues:")
        for issue in package.interp.issues:
            sections.append(f"  - {issue}")
    sections.append("")

    # Section 5: Reward Integrity Track
    sections.append("## 5. Reward Integrity Track")
    sections.append("")
    sections.append(f"**Status:** {package.reward.status.upper()}")
    sections.append(f"- No Divergence: {'Yes' if package.reward.no_divergence else 'No'}")
    sections.append(f"- No Shortcuts: {'Yes' if package.reward.no_shortcuts else 'No'}")
    sections.append(f"- No Gaming: {'Yes' if package.reward.no_gaming else 'No'}")
    if package.reward.hacking_signals:
        sections.append(f"- Hacking Signals: {', '.join(package.reward.hacking_signals)}")
    if package.reward.issues:
        sections.append("- Issues:")
        for issue in package.reward.issues:
            sections.append(f"  - {issue}")
    sections.append("")

    # Section 6: Cross-Signal Analysis
    sections.append("## 6. Cross-Signal Analysis")
    sections.append("")
    _add_cross_signal_analysis(sections, package)
    sections.append("")

    # Section 7: Risk Assessment
    sections.append("## 7. Risk Assessment")
    sections.append("")
    _add_risk_assessment(sections, package)
    sections.append("")

    # Section 8: Recommendations
    sections.append("## 8. Recommendations")
    sections.append("")
    _add_recommendations(sections, package)

    return "\n".join(sections)


def _add_cross_signal_analysis(sections: list[str], package: SafetyPackage) -> None:
    """Add cross-signal analysis section."""
    # Reward vs Constraint consistency
    if not package.reward.no_divergence and package.constraint.reward_bounded:
        sections.append(
            "- **Reward-Constraint Tension:** Divergence detected despite "
            "reward bounding being active. The bounding may be insufficient."
        )

    if not package.reward.no_shortcuts and package.constraint.entropy_maintained:
        sections.append(
            "- **Shortcut-Entropy Tension:** Shortcuts detected despite "
            "entropy being maintained. The model may be finding high-entropy "
            "shortcuts."
        )

    # Energy vs Interp consistency
    if not package.constraint.energy_stable and package.interp.energy_interpretable:
        sections.append(
            "- **Energy Instability:** Energy is unstable but still "
            "interpretable. Monitor closely for emerging patterns."
        )

    # Overall
    all_issues = (
        package.gdi.issues
        + package.constraint.issues
        + package.interp.issues
        + package.reward.issues
    )
    if not all_issues:
        sections.append("- No cross-signal tensions detected. All tracks consistent.")
    elif len(all_issues) == 1:
        sections.append(f"- Single issue identified across tracks: {all_issues[0]}")
    else:
        sections.append(f"- {len(all_issues)} total issues identified across tracks.")


def _add_risk_assessment(sections: list[str], package: SafetyPackage) -> None:
    """Add risk assessment section."""
    risk_level = "LOW"
    risk_factors = []

    if not package.reward.no_divergence:
        risk_level = "HIGH"
        risk_factors.append("Reward-accuracy divergence suggests reward hacking")

    if not package.reward.no_shortcuts:
        risk_level = max(risk_level, "MEDIUM")
        risk_factors.append("Shortcut learning patterns detected")

    if not package.constraint.energy_stable:
        risk_level = max(risk_level, "MEDIUM")
        risk_factors.append("Energy instability may indicate representation collapse")

    if not package.interp.energy_interpretable:
        risk_factors.append("Limited interpretability of energy patterns")

    sections.append(f"**Risk Level:** {risk_level}")
    if risk_factors:
        for factor in risk_factors:
            sections.append(f"- {factor}")
    else:
        sections.append("- No significant risk factors identified.")


def _add_recommendations(sections: list[str], package: SafetyPackage) -> None:
    """Add recommendations section."""
    if package.all_green:
        sections.append("- Training may proceed to the next phase.")
        sections.append("- Continue monitoring all tracks at regular intervals.")
        return

    if not package.gdi.is_green:
        sections.append("- Complete GDI review before proceeding.")

    if not package.constraint.is_green:
        if not package.constraint.reward_bounded:
            sections.append("- Tighten reward bounding parameters.")
        if not package.constraint.entropy_maintained:
            sections.append("- Increase entropy bonus coefficient or switch to target mode.")
        if not package.constraint.energy_stable:
            sections.append("- Investigate energy instability and consider early stopping.")

    if not package.interp.is_green:
        sections.append("- Improve interpretability tooling before proceeding.")

    if not package.reward.is_green:
        if not package.reward.no_divergence:
            sections.append("- Investigate reward-accuracy divergence immediately.")
        if not package.reward.no_shortcuts:
            sections.append("- Add shortcut-specific mitigations (e.g., length penalty).")
        if not package.reward.no_gaming:
            sections.append("- Review reward function for exploitable patterns.")
