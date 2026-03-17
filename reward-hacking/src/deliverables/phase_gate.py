from __future__ import annotations

"""Phase gate safety package assembly and validation."""

from dataclasses import dataclass, field

from .gdi_package import GDISummary, package_gdi
from .constraint_package import ConstraintSummary, package_constraints
from .interp_package import InterpSummary, package_interp
from .reward_package import RewardSummary, package_reward


@dataclass
class SafetyPackage:
    """Complete safety package for a training phase."""

    phase: str
    iteration_range: tuple[int, int]
    gdi: GDISummary
    constraint: ConstraintSummary
    interp: InterpSummary
    reward: RewardSummary
    all_green: bool = False
    summary: str = ""

    def __post_init__(self):
        self.all_green = (
            self.gdi.is_green
            and self.constraint.is_green
            and self.interp.is_green
            and self.reward.is_green
        )
        if self.all_green:
            self.summary = f"Phase {self.phase}: All tracks GREEN - safe to proceed"
        else:
            failing = []
            if not self.gdi.is_green:
                failing.append(f"GDI({self.gdi.status})")
            if not self.constraint.is_green:
                failing.append(f"Constraint({self.constraint.status})")
            if not self.interp.is_green:
                failing.append(f"Interp({self.interp.status})")
            if not self.reward.is_green:
                failing.append(f"Reward({self.reward.status})")
            self.summary = (
                f"Phase {self.phase}: BLOCKED - failing tracks: "
                + ", ".join(failing)
            )


@dataclass
class PackageValidation:
    """Result of safety package validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cross_signal_consistent: bool = True


class PhaseGateSafetyPackage:
    """Assembles and validates phase gate safety packages.

    A safety package requires all four tracks (GDI, Constraint,
    Interp, Reward) to be green for the phase gate to pass.
    """

    def __init__(self):
        self._packages: list[SafetyPackage] = []
        self._validations: list[PackageValidation] = []

    @property
    def packages(self) -> list[SafetyPackage]:
        return list(self._packages)

    @property
    def validations(self) -> list[PackageValidation]:
        return list(self._validations)

    def package(
        self,
        phase: str,
        iteration_range: tuple[int, int],
        histories: dict,
    ) -> SafetyPackage:
        """Assemble a safety package from training histories.

        Args:
            phase: Phase identifier (e.g., "phase_1").
            iteration_range: (start, end) iteration range.
            histories: Dict with keys 'gdi', 'constraint', 'interp', 'reward',
                       each containing track-specific history data.

        Returns:
            SafetyPackage with all four track summaries.
        """
        gdi = package_gdi(histories.get("gdi", {}))
        constraint = package_constraints(histories.get("constraint", {}))
        interp = package_interp(histories.get("interp", {}))
        reward = package_reward(histories.get("reward", {}))

        pkg = SafetyPackage(
            phase=phase,
            iteration_range=iteration_range,
            gdi=gdi,
            constraint=constraint,
            interp=interp,
            reward=reward,
        )
        self._packages.append(pkg)
        return pkg

    def validate(self, package: SafetyPackage) -> PackageValidation:
        """Validate a safety package for consistency and completeness.

        Args:
            package: Safety package to validate.

        Returns:
            PackageValidation result.
        """
        errors = []
        warnings = []

        # Check all tracks are present and have valid status
        for track_name, track in [
            ("GDI", package.gdi),
            ("Constraint", package.constraint),
            ("Interp", package.interp),
            ("Reward", package.reward),
        ]:
            if track.status not in ("green", "yellow", "red"):
                errors.append(f"{track_name} has invalid status: {track.status}")

            if track.status == "red" and not track.issues:
                warnings.append(f"{track_name} is red but has no issues listed")

        # Check for unresolved failures
        all_issues = (
            package.gdi.issues
            + package.constraint.issues
            + package.interp.issues
            + package.reward.issues
        )
        unresolved = [i for i in all_issues if "Unresolved" in i or "unresolved" in i.lower()]
        if unresolved:
            errors.append(
                f"Package has {len(unresolved)} unresolved failure(s): "
                + "; ".join(unresolved)
            )

        # Cross-signal consistency checks
        cross_consistent = self._check_cross_signal_consistency(package)
        if not cross_consistent:
            warnings.append("Cross-signal inconsistency detected")

        is_valid = len(errors) == 0 and package.all_green

        validation = PackageValidation(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            cross_signal_consistent=cross_consistent,
        )
        self._validations.append(validation)
        return validation

    def _check_cross_signal_consistency(self, package: SafetyPackage) -> bool:
        """Check consistency across track signals.

        Example: if reward track detects hacking but constraint
        track shows all constraints met, that's inconsistent.
        """
        # If reward hacking detected but constraints all met, suspicious
        if (
            package.reward.status == "red"
            and package.constraint.status == "green"
            and package.constraint.constraints_met == package.constraint.constraints_total
        ):
            return False

        # If energy not interpretable but interp is green, suspicious
        if not package.interp.energy_interpretable and package.interp.status == "green":
            return False

        # If reward divergence but GDI deployment-ready, suspicious
        if not package.reward.no_divergence and package.gdi.deployment_ready:
            # This is a yellow flag but not necessarily inconsistent
            pass

        return True
