"""Report generation for GDI analysis."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..composite.gdi import GDIResult
from ..monitoring.time_series import GDITimeSeries
from .signal_decomposition import SignalDecompositionAnalyzer
from .drift_characterization import DriftCharacterizer
from .early_warning import EarlyWarningAnalyzer


def generate_report(
    gdi_history: Optional[List[GDIResult]] = None,
    time_series: Optional[GDITimeSeries] = None,
    accuracy_history: Optional[List[float]] = None,
    alert_log: Optional[List[Dict[str, Any]]] = None,
    calibration_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive GDI report.

    Includes:
    - GDI trajectory
    - Per-signal analysis
    - Signal decomposition
    - Drift characterization
    - Calibration info
    - Early warning analysis
    - Alert log

    Args:
        gdi_history: List of GDI results.
        time_series: Optional GDITimeSeries for historical data.
        accuracy_history: Optional accuracy scores for early warning.
        alert_log: Optional list of alert dictionaries.
        calibration_info: Optional calibration data.

    Returns:
        Report dictionary.
    """
    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "title": "Goal Drift Index Report",
    }

    # GDI trajectory
    if gdi_history:
        decomposer = SignalDecompositionAnalyzer()
        characterizer = DriftCharacterizer()

        trajectories = decomposer.plot_signal_trajectories(gdi_history)
        report["trajectory"] = trajectories

        # Per-signal analysis
        if gdi_history:
            latest = gdi_history[-1]
            report["latest"] = {
                "composite_score": latest.composite_score,
                "alert_level": latest.alert_level,
                "trend": latest.trend,
                "semantic": latest.semantic_score,
                "lexical": latest.lexical_score,
                "structural": latest.structural_score,
                "distributional": latest.distributional_score,
            }

            # Decomposition
            report["decomposition"] = decomposer.decompose(latest)

            # Characterization
            char_result = characterizer.characterize(latest)
            report["characterization"] = {
                "drift_type": char_result.drift_type,
                "confidence": char_result.confidence,
                "description": char_result.description,
                "signals_involved": char_result.signals_involved,
            }

            # Primary driver across history
            report["primary_driver"] = decomposer.identify_primary_driver(
                gdi_history
            )

    # Time series data
    if time_series:
        report["time_series"] = time_series.export()

    # Early warning
    if gdi_history and accuracy_history:
        ewa = EarlyWarningAnalyzer()
        gdi_scores = [r.composite_score for r in gdi_history]
        report["early_warning"] = ewa.compute_lead_time(
            gdi_scores, accuracy_history
        )

    # Calibration
    if calibration_info:
        report["calibration"] = calibration_info

    # Alert log
    if alert_log:
        report["alert_log"] = alert_log

    return report
