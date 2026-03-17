"""Base classes for success criteria evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CriterionResult:
    """Result of evaluating a single criterion."""

    passed: bool
    confidence: float
    measured_value: Any
    threshold: Any
    margin: float
    supporting_evidence: List[str] = field(default_factory=list)
    methodology: str = ""
    caveats: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    criterion_name: str = ""

    def summary(self) -> str:
        """One-line summary of this criterion result."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.criterion_name}: "
            f"measured={self.measured_value}, threshold={self.threshold}, "
            f"margin={self.margin:+.3f}, confidence={self.confidence:.2f}"
        )


@dataclass
class Evidence:
    """Container for all evidence used in evaluation."""

    phase_0: Dict[str, Any] = field(default_factory=dict)
    phase_1: Dict[str, Any] = field(default_factory=dict)
    phase_2: Dict[str, Any] = field(default_factory=dict)
    phase_3: Dict[str, Any] = field(default_factory=dict)
    phase_4: Dict[str, Any] = field(default_factory=dict)
    safety: Dict[str, Any] = field(default_factory=dict)
    publications: List[Dict[str, Any]] = field(default_factory=list)
    audit_trail: Dict[str, Any] = field(default_factory=dict)

    def get_improvement_curve(self) -> List[float]:
        """Extract the improvement curve across phases."""
        curve = []
        for phase_key in ["phase_0", "phase_1", "phase_2", "phase_3", "phase_4"]:
            phase_data = getattr(self, phase_key)
            if "score" in phase_data:
                curve.append(phase_data["score"])
        return curve

    def get_collapse_curve(self) -> List[float]:
        """Extract the collapse/control curve across phases."""
        curve = []
        for phase_key in ["phase_0", "phase_1", "phase_2", "phase_3", "phase_4"]:
            phase_data = getattr(self, phase_key)
            if "collapse_score" in phase_data:
                curve.append(phase_data["collapse_score"])
        return curve

    def get_ablation_results(self) -> Dict[str, Dict[str, List[float]]]:
        """Extract ablation (paired) results for paradigm improvement."""
        results: Dict[str, Dict[str, List[float]]] = {}
        for phase_key in ["phase_0", "phase_1", "phase_2", "phase_3", "phase_4"]:
            phase_data = getattr(self, phase_key)
            ablations = phase_data.get("ablations", {})
            for paradigm, data in ablations.items():
                if paradigm not in results:
                    results[paradigm] = {"with": [], "without": []}
                results[paradigm]["with"].append(data.get("with", 0.0))
                results[paradigm]["without"].append(data.get("without", 0.0))
        return results

    def get_gdi_readings(self) -> List[Dict[str, Any]]:
        """Extract GDI readings from safety evidence."""
        return self.safety.get("gdi_readings", [])

    def get_phases_monitored(self) -> List[str]:
        """List which phases have GDI monitoring."""
        return self.safety.get("phases_monitored", [])


class SuccessCriterion(ABC):
    """Abstract base class for a success criterion."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name of this criterion."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        ...

    @property
    @abstractmethod
    def threshold_description(self) -> str:
        """Description of the pass/fail threshold."""
        ...

    @abstractmethod
    def evaluate(self, evidence: Evidence) -> CriterionResult:
        """Evaluate this criterion against the provided evidence."""
        ...

    @property
    @abstractmethod
    def required_evidence(self) -> List[str]:
        """List of evidence fields this criterion requires."""
        ...
