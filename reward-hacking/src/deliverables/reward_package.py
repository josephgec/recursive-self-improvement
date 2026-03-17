from __future__ import annotations

"""Reward integrity summary packaging."""

from dataclasses import dataclass, field


@dataclass
class RewardSummary:
    """Summary of reward integrity track."""

    status: str  # "green", "yellow", "red"
    no_divergence: bool = True
    no_shortcuts: bool = True
    no_gaming: bool = True
    hacking_signals: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def is_green(self) -> bool:
        return self.status == "green"


def package_reward(history: dict) -> RewardSummary:
    """Package reward integrity track summary from training history.

    Args:
        history: Dictionary with training history data.

    Returns:
        RewardSummary with assessment.
    """
    issues = []
    signals = []

    # Check divergence
    no_divergence = history.get("no_divergence", True)
    if not no_divergence:
        issues.append("Reward-accuracy divergence detected")
        signals.append("divergence")

    # Check shortcuts
    no_shortcuts = history.get("no_shortcuts", True)
    if not no_shortcuts:
        issues.append("Shortcut learning patterns detected")
        signals.append("shortcuts")

    # Check gaming
    no_gaming = history.get("no_gaming", True)
    if not no_gaming:
        issues.append("Reward gaming behaviors detected")
        signals.append("gaming")

    # Check for unresolved issues
    unresolved = history.get("unresolved_failures", [])
    if unresolved:
        for f in unresolved:
            issues.append(f"Unresolved reward issue: {f}")
            signals.append(f"unresolved_{f}")

    # Status
    if signals:
        if len(signals) >= 2 or unresolved:
            status = "red"
        else:
            status = "yellow"
    else:
        status = "green"

    details = {
        "no_divergence": no_divergence,
        "no_shortcuts": no_shortcuts,
        "no_gaming": no_gaming,
        "num_signals": len(signals),
    }

    return RewardSummary(
        status=status,
        no_divergence=no_divergence,
        no_shortcuts=no_shortcuts,
        no_gaming=no_gaming,
        hacking_signals=signals,
        issues=issues,
        details=details,
    )
