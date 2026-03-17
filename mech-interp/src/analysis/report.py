"""Generate comprehensive interpretability reports."""

from typing import Dict, List, Any, Optional
from datetime import datetime


def generate_report(
    iteration: int = 0,
    divergence_data: Optional[Dict] = None,
    diff_data: Optional[Dict] = None,
    head_tracking_data: Optional[Dict] = None,
    deceptive_alignment_data: Optional[Dict] = None,
    alerts: Optional[List[Dict]] = None,
    anomaly_summary: Optional[Dict] = None,
    time_series_data: Optional[Dict] = None,
    recommendations: Optional[List[str]] = None,
) -> str:
    """Generate an 8-section markdown report.

    Sections:
    1. Executive Summary
    2. Divergence Analysis
    3. Activation Diff Analysis
    4. Head Tracking Analysis
    5. Deceptive Alignment Probes
    6. Anomaly Summary
    7. Alerts and Warnings
    8. Recommendations
    """
    lines = []

    # Section 1: Executive Summary
    lines.append("# Mechanistic Interpretability Report")
    lines.append("")
    lines.append(f"**Iteration:** {iteration}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")

    alert_count = len(alerts) if alerts else 0
    critical_count = sum(1 for a in (alerts or []) if isinstance(a, dict) and a.get("severity") == "critical")
    status = "CRITICAL" if critical_count > 0 else "WARNING" if alert_count > 0 else "OK"
    lines.append(f"**Status:** {status}")
    lines.append(f"**Active Alerts:** {alert_count} ({critical_count} critical)")
    lines.append("")

    # Section 2: Divergence Analysis
    lines.append("## 2. Divergence Analysis")
    lines.append("")
    if divergence_data:
        lines.append(f"- Divergence Ratio: {divergence_data.get('divergence_ratio', 'N/A'):.3f}" if isinstance(divergence_data.get('divergence_ratio'), (int, float)) else f"- Divergence Ratio: {divergence_data.get('divergence_ratio', 'N/A')}")
        lines.append(f"- Internal Change: {divergence_data.get('internal_change', 'N/A'):.4f}" if isinstance(divergence_data.get('internal_change'), (int, float)) else f"- Internal Change: {divergence_data.get('internal_change', 'N/A')}")
        lines.append(f"- Behavioral Change: {divergence_data.get('behavioral_change', 'N/A'):.4f}" if isinstance(divergence_data.get('behavioral_change'), (int, float)) else f"- Behavioral Change: {divergence_data.get('behavioral_change', 'N/A')}")
        lines.append(f"- Z-Score: {divergence_data.get('z_score', 'N/A'):.2f}" if isinstance(divergence_data.get('z_score'), (int, float)) else f"- Z-Score: {divergence_data.get('z_score', 'N/A')}")
        lines.append(f"- Anomalous: {divergence_data.get('is_anomalous', False)}")
        lines.append(f"- Safety Flag: {divergence_data.get('safety_flag', False)}")
    else:
        lines.append("No divergence data available (first iteration or no comparison).")
    lines.append("")

    # Section 3: Activation Diff Analysis
    lines.append("## 3. Activation Diff Analysis")
    lines.append("")
    if diff_data:
        lines.append(f"- Most Changed Layers: {diff_data.get('most_changed_layers', [])}")
        lines.append(f"- Safety Disproportionate: {diff_data.get('safety_disproportionate', False)}")
        lines.append(f"- Overall Change Magnitude: {diff_data.get('overall_change', 0):.4f}" if isinstance(diff_data.get('overall_change'), (int, float)) else f"- Overall Change: {diff_data.get('overall_change', 'N/A')}")
    else:
        lines.append("No diff data available.")
    lines.append("")

    # Section 4: Head Tracking Analysis
    lines.append("## 4. Head Tracking Analysis")
    lines.append("")
    if head_tracking_data:
        lines.append(f"- Dying Heads: {head_tracking_data.get('num_dying_heads', 0)}")
        lines.append(f"- Narrowing Heads: {head_tracking_data.get('num_narrowing_heads', 0)}")
        lines.append(f"- Role Changes: {head_tracking_data.get('num_role_changes', 0)}")
        summary = head_tracking_data.get("summary", {})
        if summary:
            lines.append(f"- Mean Entropy: {summary.get('mean_entropy', 0):.3f}" if isinstance(summary.get('mean_entropy'), (int, float)) else "- Mean Entropy: N/A")
    else:
        lines.append("No head tracking data available.")
    lines.append("")

    # Section 5: Deceptive Alignment Probes
    lines.append("## 5. Deceptive Alignment Probes")
    lines.append("")
    if deceptive_alignment_data:
        lines.append(f"- Monitoring Sensitivity: {deceptive_alignment_data.get('monitoring_sensitivity', 0):.3f}" if isinstance(deceptive_alignment_data.get('monitoring_sensitivity'), (int, float)) else "- Monitoring Sensitivity: N/A")
        lines.append(f"- Context-Dependent Safety: {deceptive_alignment_data.get('context_dependent_safety', 0):.3f}" if isinstance(deceptive_alignment_data.get('context_dependent_safety'), (int, float)) else "- Context-Dependent Safety: N/A")
        lines.append(f"- Latent Capability Gap: {deceptive_alignment_data.get('latent_capability_gap', 0):.3f}" if isinstance(deceptive_alignment_data.get('latent_capability_gap'), (int, float)) else "- Latent Capability Gap: N/A")
        lines.append(f"- Paraphrase Consistency: {deceptive_alignment_data.get('paraphrase_consistency', 0):.3f}" if isinstance(deceptive_alignment_data.get('paraphrase_consistency'), (int, float)) else "- Paraphrase Consistency: N/A")
        lines.append(f"- Suspicious: {deceptive_alignment_data.get('is_suspicious', False)}")
        flags = deceptive_alignment_data.get("flags", [])
        if flags:
            lines.append(f"- Flags: {', '.join(flags)}")
    else:
        lines.append("No deceptive alignment data available.")
    lines.append("")

    # Section 6: Anomaly Summary
    lines.append("## 6. Anomaly Summary")
    lines.append("")
    if anomaly_summary:
        lines.append(f"- Total Checks: {anomaly_summary.get('total_checks', 0)}")
        lines.append(f"- Anomalous: {anomaly_summary.get('total_anomalous', 0)}")
        lines.append(f"- Anomaly Rate: {anomaly_summary.get('anomaly_rate', 0):.1%}" if isinstance(anomaly_summary.get('anomaly_rate'), (int, float)) else "- Anomaly Rate: N/A")
        lines.append(f"- Safety Flagged: {anomaly_summary.get('total_safety_flagged', 0)}")
    else:
        lines.append("No anomaly summary available.")
    lines.append("")

    # Section 7: Alerts and Warnings
    lines.append("## 7. Alerts and Warnings")
    lines.append("")
    if alerts:
        for alert in alerts:
            if isinstance(alert, dict):
                sev = alert.get("severity", "info")
                name = alert.get("rule_name", "unknown")
                desc = alert.get("description", "")
                lines.append(f"- **[{sev.upper()}]** {name}: {desc}")
            else:
                lines.append(f"- {alert}")
    else:
        lines.append("No alerts triggered.")
    lines.append("")

    # Section 8: Recommendations
    lines.append("## 8. Recommendations")
    lines.append("")
    if recommendations:
        for rec in recommendations:
            lines.append(f"- {rec}")
    else:
        # Auto-generate recommendations
        recs = _auto_recommendations(
            divergence_data, diff_data, head_tracking_data,
            deceptive_alignment_data, alerts
        )
        for rec in recs:
            lines.append(f"- {rec}")
    lines.append("")

    return "\n".join(lines)


def _auto_recommendations(
    divergence_data: Optional[Dict],
    diff_data: Optional[Dict],
    head_tracking_data: Optional[Dict],
    deceptive_alignment_data: Optional[Dict],
    alerts: Optional[List[Dict]],
) -> List[str]:
    """Auto-generate recommendations based on data."""
    recs = []

    if divergence_data and divergence_data.get("is_anomalous"):
        recs.append("Investigate high divergence ratio between internal and behavioral changes")

    if diff_data and diff_data.get("safety_disproportionate"):
        recs.append("Review safety-related activation changes — disproportionate shift detected")

    if head_tracking_data:
        if head_tracking_data.get("num_dying_heads", 0) > 0:
            recs.append("Monitor dying attention heads — potential capacity loss")
        if head_tracking_data.get("num_role_changes", 0) > 0:
            recs.append("Review head role changes — potential silent reorganization")

    if deceptive_alignment_data and deceptive_alignment_data.get("is_suspicious"):
        recs.append("CRITICAL: Deceptive alignment indicators detected — manual review required")

    alerts = alerts or []
    critical = [a for a in alerts if isinstance(a, dict) and a.get("severity") == "critical"]
    if critical:
        recs.append("Address all critical alerts before proceeding with further modifications")

    if not recs:
        recs.append("No significant concerns detected. Continue monitoring.")

    return recs
