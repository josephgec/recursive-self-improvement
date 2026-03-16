"""HeadroomMonitor: track how close constraints are to being violated."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.checker.verdict import SuiteVerdict
from src.constraints.base import ConstraintResult


@dataclass
class HeadroomReport:
    """Report on constraint headroom across all constraints."""

    headrooms: Dict[str, float] = field(default_factory=dict)
    at_risk: List[str] = field(default_factory=list)
    details: Dict[str, Dict] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["Headroom Report:"]
        for name, hr in sorted(self.headrooms.items(), key=lambda x: x[1]):
            flag = " [AT RISK]" if name in self.at_risk else ""
            lines.append(f"  {name}: {hr:.4f}{flag}")
        return "\n".join(lines)


class HeadroomMonitor:
    """Monitor headroom across all constraints."""

    def __init__(self, warning_threshold: float = 0.05) -> None:
        self._warning_threshold = warning_threshold

    def compute_all(self, verdict: SuiteVerdict) -> HeadroomReport:
        """Compute headroom for every constraint in the verdict."""
        headrooms: Dict[str, float] = {}
        details: Dict[str, Dict] = {}

        for name, result in verdict.results.items():
            headrooms[name] = result.headroom
            details[name] = {
                "measured_value": result.measured_value,
                "threshold": result.threshold,
                "headroom": result.headroom,
                "satisfied": result.satisfied,
            }

        at_risk = self.identify_at_risk(headrooms)

        return HeadroomReport(
            headrooms=headrooms,
            at_risk=at_risk,
            details=details,
        )

    def identify_at_risk(
        self, headrooms: Dict[str, float]
    ) -> List[str]:
        """Identify constraints with headroom below the warning threshold."""
        return [
            name
            for name, hr in headrooms.items()
            if 0 <= hr < self._warning_threshold
        ]

    def plot_headroom_dashboard(self, report: HeadroomReport) -> str:
        """Produce a text-based headroom dashboard (no matplotlib dependency)."""
        lines = ["=" * 60, "HEADROOM DASHBOARD", "=" * 60]
        max_bar = 40

        sorted_items = sorted(report.headrooms.items(), key=lambda x: x[1])
        max_hr = max(abs(h) for _, h in sorted_items) if sorted_items else 1.0
        if max_hr == 0:
            max_hr = 1.0

        for name, hr in sorted_items:
            bar_len = int(abs(hr) / max_hr * max_bar)
            if hr < 0:
                bar = "X" * bar_len
                status = "VIOLATED"
            elif name in report.at_risk:
                bar = "!" * bar_len
                status = "AT RISK"
            else:
                bar = "#" * bar_len
                status = "OK"

            lines.append(f"  {name:25s} [{bar:<{max_bar}s}] {hr:+.4f}  {status}")

        lines.append("=" * 60)
        return "\n".join(lines)
