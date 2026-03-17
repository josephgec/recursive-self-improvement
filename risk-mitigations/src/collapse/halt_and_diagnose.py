"""Halt-and-diagnose protocol for model collapse emergencies.

Determines when training should be halted based on health indicators,
and produces diagnostic reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HaltReport:
    """Report from a halt-and-diagnose decision."""
    should_halt: bool
    reasons: List[str] = field(default_factory=list)
    entropy_trend: Optional[str] = None
    kl_trend: Optional[str] = None
    consecutive_degradations: int = 0
    severity: str = "none"  # "none", "warning", "critical"
    recommended_action: Optional[str] = None

    @property
    def is_critical(self) -> bool:
        return self.severity == "critical"


class HaltAndDiagnoseProtocol:
    """Protocol for deciding when to halt training due to collapse indicators.

    Triggers halt based on:
    - Entropy drop exceeding threshold
    - KL divergence increase exceeding threshold
    - Consecutive degradations exceeding max
    """

    def __init__(
        self,
        entropy_drop_threshold: float = 0.3,
        kl_increase_threshold: float = 2.0,
        max_consecutive_degradations: int = 3,
    ):
        self.entropy_drop_threshold = entropy_drop_threshold
        self.kl_increase_threshold = kl_increase_threshold
        self.max_consecutive_degradations = max_consecutive_degradations
        self._halt_history: List[HaltReport] = []
        self._halted = False

    def should_halt(self, history: Dict[str, List[float]]) -> HaltReport:
        """Determine if training should be halted.

        Args:
            history: Dict with 'entropy' and/or 'kl_divergence' time series.

        Returns:
            HaltReport with decision and diagnostics.
        """
        reasons = []
        halt = False
        severity = "none"
        entropy_trend = None
        kl_trend = None
        consecutive_degradations = 0

        # Check entropy drop
        entropy = history.get("entropy", [])
        if len(entropy) >= 2:
            recent_drop = entropy[-2] - entropy[-1]
            if entropy[-2] > 0:
                relative_drop = recent_drop / entropy[-2]
            else:
                relative_drop = 0.0

            if relative_drop > self.entropy_drop_threshold:
                reasons.append(
                    f"Entropy dropped {relative_drop:.1%} "
                    f"(threshold: {self.entropy_drop_threshold:.1%})"
                )
                halt = True
                entropy_trend = "sharp_decline"
            elif recent_drop > 0:
                entropy_trend = "declining"
            else:
                entropy_trend = "stable_or_increasing"

        # Check KL divergence increase
        kl = history.get("kl_divergence", [])
        if len(kl) >= 2:
            kl_increase = kl[-1] - kl[-2]
            if kl_increase > self.kl_increase_threshold:
                reasons.append(
                    f"KL divergence increased by {kl_increase:.2f} "
                    f"(threshold: {self.kl_increase_threshold:.2f})"
                )
                halt = True
                kl_trend = "spiking"
            elif kl_increase > 0:
                kl_trend = "increasing"
            else:
                kl_trend = "stable_or_decreasing"

        # Check consecutive degradations
        quality = history.get("quality_score", [])
        if len(quality) >= 2:
            count = 0
            for i in range(len(quality) - 1, 0, -1):
                if quality[i] < quality[i - 1]:
                    count += 1
                else:
                    break
            consecutive_degradations = count

            if count >= self.max_consecutive_degradations:
                reasons.append(
                    f"{count} consecutive quality degradations "
                    f"(max: {self.max_consecutive_degradations})"
                )
                halt = True

        # Determine severity
        if halt and len(reasons) >= 2:
            severity = "critical"
        elif halt:
            severity = "critical"
        elif consecutive_degradations >= 2 or entropy_trend == "declining":
            severity = "warning"

        # Determine recommended action
        if severity == "critical":
            recommended_action = "halt_and_rollback"
        elif severity == "warning":
            recommended_action = "increase_alpha"
        else:
            recommended_action = None

        report = HaltReport(
            should_halt=halt,
            reasons=reasons,
            entropy_trend=entropy_trend,
            kl_trend=kl_trend,
            consecutive_degradations=consecutive_degradations,
            severity=severity,
            recommended_action=recommended_action,
        )

        self._halt_history.append(report)
        if halt:
            self._halted = True

        return report

    def execute_halt(self) -> Dict[str, Any]:
        """Execute the halt protocol.

        Returns:
            Dict with halt execution details.
        """
        self._halted = True
        return {
            "status": "halted",
            "action": "training_stopped",
            "message": "Training halted due to collapse indicators",
            "halt_history_length": len(self._halt_history),
        }

    @property
    def is_halted(self) -> bool:
        return self._halted

    def resume(self) -> None:
        """Resume training after halt."""
        self._halted = False

    def get_history(self) -> List[HaltReport]:
        """Return halt decision history."""
        return list(self._halt_history)
