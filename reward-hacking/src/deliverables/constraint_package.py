from __future__ import annotations

"""Constraint satisfaction summary packaging."""

from dataclasses import dataclass, field


@dataclass
class ConstraintSummary:
    """Summary of constraint satisfaction track."""

    status: str  # "green", "yellow", "red"
    reward_bounded: bool = True
    entropy_maintained: bool = True
    energy_stable: bool = True
    constraints_met: int = 0
    constraints_total: int = 3
    issues: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def is_green(self) -> bool:
        return self.status == "green"


def package_constraints(history: dict) -> ConstraintSummary:
    """Package constraint track summary from training history.

    Args:
        history: Dictionary with training history data.

    Returns:
        ConstraintSummary with assessment.
    """
    issues = []
    met = 0
    total = 3

    # Check reward bounding
    reward_bounded = history.get("reward_bounded", True)
    if reward_bounded:
        met += 1
    else:
        issues.append("Reward bounding constraint violated")

    # Check entropy maintenance
    entropy_maintained = history.get("entropy_above_min", True)
    if entropy_maintained:
        met += 1
    else:
        issues.append("Entropy fell below minimum threshold")

    # Check energy stability
    energy_stable = history.get("energy_stable", True)
    if energy_stable:
        met += 1
    else:
        issues.append("Energy instability detected")

    # Check for unresolved constraint failures
    unresolved = history.get("unresolved_failures", [])
    if unresolved:
        for f in unresolved:
            issues.append(f"Unresolved constraint failure: {f}")

    # Status
    if met == total and not unresolved:
        status = "green"
    elif met >= total - 1 and not unresolved:
        status = "yellow"
    else:
        status = "red"

    details = {
        "constraints_met": met,
        "constraints_total": total,
        "reward_bounded": reward_bounded,
        "entropy_maintained": entropy_maintained,
        "energy_stable": energy_stable,
    }

    return ConstraintSummary(
        status=status,
        reward_bounded=reward_bounded,
        entropy_maintained=entropy_maintained,
        energy_stable=energy_stable,
        constraints_met=met,
        constraints_total=total,
        issues=issues,
        details=details,
    )
