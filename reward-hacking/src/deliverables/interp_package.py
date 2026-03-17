from __future__ import annotations

"""Interpretability summary packaging."""

from dataclasses import dataclass, field


@dataclass
class InterpSummary:
    """Summary of interpretability track."""

    status: str  # "green", "yellow", "red"
    energy_interpretable: bool = True
    homogenization_checked: bool = True
    activations_tracked: bool = True
    issues: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def is_green(self) -> bool:
        return self.status == "green"


def package_interp(history: dict) -> InterpSummary:
    """Package interpretability track summary from training history.

    Args:
        history: Dictionary with training history data.

    Returns:
        InterpSummary with assessment.
    """
    issues = []

    # Check energy interpretability
    energy_ok = history.get("energy_interpretable", True)
    if not energy_ok:
        issues.append("Energy patterns not interpretable")

    # Check homogenization analysis
    homog_checked = history.get("homogenization_checked", True)
    if not homog_checked:
        issues.append("Homogenization analysis not completed")

    # Check activation tracking
    activations_ok = history.get("activations_tracked", True)
    if not activations_ok:
        issues.append("Activation tracking not active")

    # Check for unresolved issues
    unresolved = history.get("unresolved_failures", [])
    if unresolved:
        for f in unresolved:
            issues.append(f"Unresolved interpretability issue: {f}")

    # Status
    if issues:
        if any("Unresolved" in i for i in issues):
            status = "red"
        else:
            status = "yellow"
    else:
        status = "green"

    details = {
        "energy_interpretable": energy_ok,
        "homogenization_checked": homog_checked,
        "activations_tracked": activations_ok,
        "num_issues": len(issues),
    }

    return InterpSummary(
        status=status,
        energy_interpretable=energy_ok,
        homogenization_checked=homog_checked,
        activations_tracked=activations_ok,
        issues=issues,
        details=details,
    )
