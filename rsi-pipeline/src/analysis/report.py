"""Report generator: creates comprehensive markdown reports."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def generate_report(
    pipeline_result: Optional[Dict[str, Any]] = None,
    safety_report: Optional[Dict[str, Any]] = None,
    convergence: Optional[Dict[str, Any]] = None,
    paradigm_contribution: Optional[Dict[str, Any]] = None,
    improvement_curve: Optional[List] = None,
    lineage: Optional[List] = None,
) -> str:
    """Generate a comprehensive markdown report with 12 sections.

    Sections:
    1. Executive Summary
    2. Pipeline Configuration
    3. Iteration Summary
    4. Improvement Curve
    5. Convergence Analysis
    6. Safety Status
    7. GDI Trajectory
    8. CAR Trajectory
    9. Modification History
    10. Paradigm Contributions
    11. Lineage Analysis
    12. Recommendations
    """
    pr = pipeline_result or {}
    sr = safety_report or {}
    conv = convergence or {}
    pc = paradigm_contribution or {}
    curve = improvement_curve or []
    lin = lineage or []

    sections = []

    # 1. Executive Summary
    sections.append("# RSI Pipeline Report\n")
    sections.append("## 1. Executive Summary\n")
    sections.append(f"- Total iterations: {pr.get('total_iterations', 0)}")
    sections.append(f"- Successful improvements: {pr.get('successful_improvements', 0)}")
    sections.append(f"- Rollbacks: {pr.get('rollbacks', 0)}")
    sections.append(f"- Emergency stops: {pr.get('emergency_stops', 0)}")
    sections.append(f"- Final accuracy: {pr.get('final_accuracy', 0.0):.4f}")
    sections.append(f"- Initial accuracy: {pr.get('initial_accuracy', 0.0):.4f}")
    sections.append(f"- Total accuracy gain: {pr.get('total_accuracy_gain', 0.0):.4f}")
    sections.append(f"- Improvement rate: {pr.get('improvement_rate', 0.0):.2%}")
    sections.append(f"- Reason stopped: {pr.get('reason_stopped', 'N/A')}\n")

    # 2. Pipeline Configuration
    sections.append("## 2. Pipeline Configuration\n")
    sections.append("- Paradigms: SOAR, SymCode/CTM, Godel Agent, RLM")
    sections.append("- Verification: Dual (Empirical + Compactness) with Pareto filter")
    sections.append("- Safety: GDI, Constraints, CAR, Emergency Stop\n")

    # 3. Iteration Summary
    sections.append("## 3. Iteration Summary\n")
    iter_results = pr.get("iteration_results", [])
    if iter_results:
        sections.append("| Iteration | Improved | Accuracy Before | Accuracy After | Safety |")
        sections.append("|-----------|----------|-----------------|----------------|--------|")
        for ir in iter_results[:20]:  # limit to 20 rows
            sections.append(
                f"| {ir.get('iteration', '?')} "
                f"| {ir.get('improved', False)} "
                f"| {ir.get('accuracy_before', 0):.4f} "
                f"| {ir.get('accuracy_after', 0):.4f} "
                f"| {ir.get('safety_verdict', '?')} |"
            )
    sections.append("")

    # 4. Improvement Curve
    sections.append("## 4. Improvement Curve\n")
    if curve:
        for iteration, accuracy in curve:
            sections.append(f"- Iteration {iteration}: {accuracy:.4f}")
    else:
        sections.append("No improvement curve data available.")
    sections.append("")

    # 5. Convergence Analysis
    sections.append("## 5. Convergence Analysis\n")
    sections.append(f"- Converged: {conv.get('converged', False)}")
    sections.append(f"- Estimated ceiling: {conv.get('ceiling', 'N/A')}")
    sections.append(f"- Marginal returns: {conv.get('marginal_returns', 'N/A')}\n")

    # 6. Safety Status
    sections.append("## 6. Safety Status\n")
    current = sr.get("current_status", {})
    sections.append(f"- GDI score: {current.get('gdi_score', 0.0):.4f}")
    sections.append(f"- CAR score: {current.get('car_score', 1.0):.4f}")
    sections.append(f"- Constraints satisfied: {current.get('constraints_satisfied', True)}")
    sections.append(f"- Consecutive rollbacks: {current.get('consecutive_rollbacks', 0)}")
    sections.append(f"- Emergency stop: {current.get('emergency_stop', False)}\n")

    # 7. GDI Trajectory
    sections.append("## 7. GDI Trajectory\n")
    gdi = sr.get("gdi_trajectory", {})
    sections.append(f"- Trend: {gdi.get('trend', 'N/A')}")
    sections.append(f"- Current: {gdi.get('current', 0.0)}")
    sections.append(f"- Max: {gdi.get('max', 0.0)}")
    sections.append(f"- Average: {gdi.get('avg', 0.0)}\n")

    # 8. CAR Trajectory
    sections.append("## 8. CAR Trajectory\n")
    car = sr.get("car_trajectory", {})
    sections.append(f"- Trend: {car.get('trend', 'N/A')}")
    sections.append(f"- Current: {car.get('current', 1.0)}")
    sections.append(f"- Min: {car.get('min', 1.0)}")
    sections.append(f"- Average: {car.get('avg', 1.0)}\n")

    # 9. Modification History
    sections.append("## 9. Modification History\n")
    violations = sr.get("violation_summary", {})
    sections.append(f"- Total violations: {violations.get('total', 0)}")
    sections.append(f"- Risk assessment: {sr.get('risk_assessment', 'N/A')}\n")

    # 10. Paradigm Contributions
    sections.append("## 10. Paradigm Contributions\n")
    sections.append(f"- SOAR efficiency: {pc.get('soar', {}).get('efficiency', 'N/A')}")
    sections.append(f"- Modification success rate: {pc.get('success_rate', 'N/A')}")
    verification = pc.get("verification", {})
    sections.append(f"- Empirical pass/fail: {verification.get('empirical_pass', 0)}/{verification.get('empirical_fail', 0)}")
    sections.append(f"- Compactness pass/fail: {verification.get('compactness_pass', 0)}/{verification.get('compactness_fail', 0)}\n")

    # 11. Lineage Analysis
    sections.append("## 11. Lineage Analysis\n")
    if lin:
        sections.append(f"- Total modifications tracked: {len(lin)}")
        improved = sum(1 for e in lin if e.get("improved", False))
        sections.append(f"- Improvements: {improved}")
    else:
        sections.append("No lineage data available.")
    sections.append("")

    # 12. Recommendations
    sections.append("## 12. Recommendations\n")
    risk = sr.get("risk_assessment", "low")
    if risk == "critical":
        sections.append("- CRITICAL: Pipeline in emergency state. Manual review required.")
    elif risk == "high":
        sections.append("- HIGH RISK: GDI drift is high. Consider reducing modification scope.")
    elif risk == "medium":
        sections.append("- MEDIUM RISK: Monitor safety metrics closely.")
    else:
        sections.append("- LOW RISK: Pipeline operating within safe parameters.")

    if conv.get("converged", False):
        sections.append("- Pipeline has converged. Consider changing strategy or stopping.")

    sections.append("")

    return "\n".join(sections)
