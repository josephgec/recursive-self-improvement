"""Report generation: constraint satisfaction, headroom, rejections, tightness, trends."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from src.checker.verdict import SuiteVerdict
from src.monitoring.headroom import HeadroomMonitor, HeadroomReport
from src.monitoring.trend import TrendDetector
from src.analysis.rejection_analysis import RejectionAnalyzer
from src.analysis.constraint_tightness import ConstraintTightnessAnalyzer


def generate_report(
    verdict: SuiteVerdict,
    audit_entries: List[Dict[str, Any]],
    trend_detector: Optional[TrendDetector] = None,
    headroom_warning_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Generate a comprehensive analysis report.

    Sections:
    1. Constraint satisfaction
    2. Headroom analysis
    3. Rejection analysis
    4. Constraint tightness
    5. Trends
    """
    # 1. Constraint satisfaction
    satisfaction = {}
    for name, result in verdict.results.items():
        satisfaction[name] = {
            "satisfied": result.satisfied,
            "measured_value": result.measured_value,
            "threshold": result.threshold,
            "headroom": result.headroom,
        }

    # 2. Headroom
    monitor = HeadroomMonitor(warning_threshold=headroom_warning_threshold)
    headroom_report = monitor.compute_all(verdict)

    # 3. Rejections
    analyzer = RejectionAnalyzer(audit_entries)
    rejection_info = {
        "rejection_rate": analyzer.rejection_rate(),
        "by_constraint": analyzer.rejection_by_constraint(),
        "by_modification_type": analyzer.rejection_by_modification_type(),
    }

    # 4. Tightness
    tightness_analyzer = ConstraintTightnessAnalyzer(audit_entries)
    tightness = tightness_analyzer.analyze()
    tightness_suggestions = tightness_analyzer.suggest_adjustments()

    # 5. Trends
    trends_section: Dict[str, Any] = {}
    if trend_detector is not None:
        trend_results = trend_detector.compute_trends()
        trends_section = {
            name: {
                "slope": t.slope,
                "direction": t.direction,
                "predicted_steps_to_violation": t.predicted_steps_to_violation,
                "warning": t.warning,
            }
            for name, t in trend_results.items()
        }

    return {
        "generated_at": time.time(),
        "overall_passed": verdict.passed,
        "constraint_satisfaction": satisfaction,
        "headroom": {
            "values": headroom_report.headrooms,
            "at_risk": headroom_report.at_risk,
        },
        "rejections": rejection_info,
        "tightness": tightness,
        "tightness_suggestions": tightness_suggestions,
        "trends": trends_section,
    }
