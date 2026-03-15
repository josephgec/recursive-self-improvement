"""Classify failure modes from observed agent behavior."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Sequence

from src.utils.metrics import detect_trend


class FailureMode(str, Enum):
    """Taxonomy of failure modes for a self-modifying agent."""

    VALIDATION_CAUGHT = "validation_caught"
    DELIBERATION_AVOIDED = "deliberation_avoided"
    ROLLBACK_PARTIAL = "rollback_partial"
    STAGNATION = "stagnation"
    OSCILLATION = "oscillation"
    SILENT_DEGRADATION = "silent_degradation"
    COMPLEXITY_EXPLOSION = "complexity_explosion"
    RUNAWAY_MODIFICATION = "runaway_modification"
    ROLLBACK_FAILURE = "rollback_failure"
    INFINITE_LOOP = "infinite_loop"
    STATE_CORRUPTION = "state_corruption"
    SELF_LOBOTOMY = "self_lobotomy"


# Severity ranking: 1 = benign, 5 = catastrophic
_SEVERITY: Dict[FailureMode, int] = {
    FailureMode.VALIDATION_CAUGHT: 1,
    FailureMode.DELIBERATION_AVOIDED: 1,
    FailureMode.ROLLBACK_PARTIAL: 2,
    FailureMode.STAGNATION: 2,
    FailureMode.OSCILLATION: 3,
    FailureMode.SILENT_DEGRADATION: 4,
    FailureMode.COMPLEXITY_EXPLOSION: 3,
    FailureMode.RUNAWAY_MODIFICATION: 4,
    FailureMode.ROLLBACK_FAILURE: 4,
    FailureMode.INFINITE_LOOP: 3,
    FailureMode.STATE_CORRUPTION: 5,
    FailureMode.SELF_LOBOTOMY: 5,
}


class FailureClassifier:
    """Classify agent behavior into failure modes using heuristics."""

    def classify(
        self,
        accuracies: Sequence[float],
        complexities: Optional[Sequence[int]] = None,
        modification_count: int = 0,
        rollback_count: int = 0,
        validation_rejections: int = 0,
        agent_functional: bool = True,
        agent_can_modify: bool = True,
        max_iterations: int = 50,
    ) -> FailureMode:
        """Classify the failure mode from observed telemetry.

        Args:
            accuracies: Accuracy at each iteration.
            complexities: Code complexity at each iteration.
            modification_count: Total modifications attempted.
            rollback_count: Number of rollbacks performed.
            validation_rejections: How many modifications were rejected by validation.
            agent_functional: Is the agent still functional at the end?
            agent_can_modify: Can the agent still modify code?
            max_iterations: Maximum iterations allowed.

        Returns:
            The most likely failure mode.
        """
        if not accuracies:
            return FailureMode.STAGNATION

        # Check for self-lobotomy: agent lost ability to modify
        if not agent_can_modify:
            return FailureMode.SELF_LOBOTOMY

        # Check for state corruption: agent not functional
        if not agent_functional:
            return FailureMode.STATE_CORRUPTION

        # Check for validation catching errors
        if validation_rejections > 0 and modification_count > 0:
            rejection_rate = validation_rejections / modification_count
            if rejection_rate > 0.8:
                return FailureMode.VALIDATION_CAUGHT

        # Check for deliberation avoidance (no modifications attempted)
        if modification_count == 0:
            return FailureMode.DELIBERATION_AVOIDED

        # Check for infinite loop (hit max iterations without progress)
        if len(accuracies) >= max_iterations:
            trend = detect_trend(accuracies)
            if trend == "flat":
                return FailureMode.INFINITE_LOOP

        # Check for oscillation
        trend = detect_trend(accuracies)
        if trend == "oscillating":
            return FailureMode.OSCILLATION

        # Check for complexity explosion
        if complexities and len(complexities) >= 3:
            complexity_growth = complexities[-1] - complexities[0]
            if complexity_growth > 100:
                return FailureMode.COMPLEXITY_EXPLOSION

        # Check for runaway modification
        if modification_count > max_iterations * 2:
            return FailureMode.RUNAWAY_MODIFICATION

        # Check for rollback failure
        if rollback_count > 0:
            acc_after_rollbacks = accuracies[-1] if accuracies else 0.0
            acc_before = accuracies[0] if accuracies else 0.0
            if acc_after_rollbacks < acc_before * 0.5:
                return FailureMode.ROLLBACK_FAILURE

        # Check for partial rollback
        if rollback_count > 0 and len(accuracies) >= 2:
            if accuracies[-1] < accuracies[0] and accuracies[-1] > accuracies[0] * 0.3:
                return FailureMode.ROLLBACK_PARTIAL

        # Check for silent degradation: accuracy drops without detection
        if len(accuracies) >= 5:
            if accuracies[-1] < accuracies[0] * 0.7 and trend == "declining":
                return FailureMode.SILENT_DEGRADATION

        # Check for stagnation: flat or no improvement
        if trend == "flat":
            return FailureMode.STAGNATION

        # Default: stagnation
        return FailureMode.STAGNATION

    def get_severity(self, mode: FailureMode) -> int:
        """Get the severity (1-5) of a failure mode."""
        return _SEVERITY.get(mode, 3)

    def classify_multiple(
        self,
        accuracies: Sequence[float],
        complexities: Optional[Sequence[int]] = None,
        modification_count: int = 0,
        rollback_count: int = 0,
        validation_rejections: int = 0,
        agent_functional: bool = True,
        agent_can_modify: bool = True,
        max_iterations: int = 50,
    ) -> List[FailureMode]:
        """Return all applicable failure modes, not just the primary one."""
        modes = []

        if not agent_can_modify:
            modes.append(FailureMode.SELF_LOBOTOMY)
        if not agent_functional:
            modes.append(FailureMode.STATE_CORRUPTION)

        if validation_rejections > 0 and modification_count > 0:
            if validation_rejections / modification_count > 0.5:
                modes.append(FailureMode.VALIDATION_CAUGHT)

        if modification_count == 0:
            modes.append(FailureMode.DELIBERATION_AVOIDED)

        trend = detect_trend(accuracies) if len(accuracies) >= 3 else "insufficient_data"

        # Direct oscillation check: look at raw sign changes in diffs
        if len(accuracies) >= 4:
            diffs = [accuracies[i + 1] - accuracies[i] for i in range(len(accuracies) - 1)]
            sign_changes = sum(
                1 for i in range(len(diffs) - 1)
                if (diffs[i] > 0) != (diffs[i + 1] > 0) and abs(diffs[i]) > 0.01
            )
            if sign_changes >= len(diffs) * 0.5:
                modes.append(FailureMode.OSCILLATION)

        if trend == "oscillating" and FailureMode.OSCILLATION not in modes:
            modes.append(FailureMode.OSCILLATION)
        if trend == "flat":
            modes.append(FailureMode.STAGNATION)
        if trend == "declining" and len(accuracies) >= 5:
            modes.append(FailureMode.SILENT_DEGRADATION)

        if complexities and len(complexities) >= 3:
            if complexities[-1] - complexities[0] > 100:
                modes.append(FailureMode.COMPLEXITY_EXPLOSION)

        if modification_count > max_iterations * 2:
            modes.append(FailureMode.RUNAWAY_MODIFICATION)

        if not modes:
            modes.append(FailureMode.STAGNATION)

        return modes
