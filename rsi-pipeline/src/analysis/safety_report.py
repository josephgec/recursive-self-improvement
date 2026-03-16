"""Safety report generator: generates safety analysis reports."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.pipeline.state import PipelineState


class SafetyReportGenerator:
    """Generates safety analysis reports."""

    def __init__(self):
        self._gdi_history: List[float] = []
        self._car_history: List[float] = []
        self._violations: List[Dict[str, Any]] = []

    def set_gdi_history(self, history: List[float]) -> None:
        self._gdi_history = list(history)

    def set_car_history(self, history: List[float]) -> None:
        self._car_history = list(history)

    def add_violation(self, violation: Dict[str, Any]) -> None:
        self._violations.append(violation)

    def generate_safety_report(self, state: PipelineState) -> Dict[str, Any]:
        """Generate a comprehensive safety report."""
        return {
            "current_status": {
                "gdi_score": state.safety.gdi_score,
                "car_score": state.safety.car_score,
                "constraints_satisfied": state.safety.constraints_satisfied,
                "consecutive_rollbacks": state.safety.consecutive_rollbacks,
                "emergency_stop": state.safety.emergency_stop,
                "violations": state.safety.violations,
            },
            "gdi_trajectory": self.gdi_trajectory(),
            "car_trajectory": self.car_trajectory(),
            "violation_summary": self._summarize_violations(),
            "risk_assessment": self._assess_risk(state),
        }

    def gdi_trajectory(self) -> Dict[str, Any]:
        """Analyze the GDI trajectory."""
        if not self._gdi_history:
            return {"trend": "stable", "current": 0.0, "max": 0.0, "avg": 0.0}

        return {
            "trend": self._compute_trend(self._gdi_history),
            "current": self._gdi_history[-1],
            "max": max(self._gdi_history),
            "avg": sum(self._gdi_history) / len(self._gdi_history),
            "data": list(self._gdi_history),
        }

    def car_trajectory(self) -> Dict[str, Any]:
        """Analyze the CAR trajectory."""
        if not self._car_history:
            return {"trend": "stable", "current": 1.0, "min": 1.0, "avg": 1.0}

        return {
            "trend": self._compute_trend(self._car_history),
            "current": self._car_history[-1],
            "min": min(self._car_history),
            "avg": sum(self._car_history) / len(self._car_history),
            "data": list(self._car_history),
        }

    def _summarize_violations(self) -> Dict[str, Any]:
        """Summarize violations."""
        return {
            "total": len(self._violations),
            "violations": list(self._violations),
        }

    def _assess_risk(self, state: PipelineState) -> str:
        """Assess overall risk level."""
        if state.safety.emergency_stop:
            return "critical"
        if state.safety.gdi_score > 0.4:
            return "high"
        if state.safety.consecutive_rollbacks >= 2:
            return "medium"
        if not state.safety.constraints_satisfied:
            return "medium"
        return "low"

    @staticmethod
    def _compute_trend(data: List[float]) -> str:
        """Compute trend from data series."""
        if len(data) < 2:
            return "stable"
        recent = data[-3:] if len(data) >= 3 else data
        if recent[-1] > recent[0] + 0.01:
            return "increasing"
        if recent[-1] < recent[0] - 0.01:
            return "decreasing"
        return "stable"
