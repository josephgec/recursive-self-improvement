"""Interpretability hooks for the Godel self-improvement system."""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List

from src.probing.probe_set import ProbeSet
from src.probing.extractor import ActivationExtractor, ActivationSnapshot
from src.probing.diff import ActivationDiff, ActivationDiffResult
from src.anomaly.divergence_detector import BehavioralInternalDivergenceDetector, DivergenceCheckResult


@dataclass
class InterpretabilityCheckResult:
    """Result of an interpretability check during Godel modification."""
    should_block: bool
    reason: str
    divergence_result: Optional[DivergenceCheckResult] = None
    diff_result: Optional[ActivationDiffResult] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "should_block": self.should_block,
            "reason": self.reason,
            "warnings": self.warnings,
        }
        if self.divergence_result:
            d["divergence"] = self.divergence_result.to_dict()
        if self.diff_result:
            d["diff"] = self.diff_result.to_dict()
        return d


class GodelInterpretabilityHooks:
    """Hooks that integrate with the Godel self-improvement loop.

    Provides before/after modification checks and blocking capability.
    """

    def __init__(self, model: Any,
                 probe_set: Optional[ProbeSet] = None,
                 max_divergence_ratio: float = 5.0,
                 max_safety_shift: float = 0.5,
                 block_on_critical: bool = True):
        self.model = model
        self.probe_set = probe_set or ProbeSet()
        self.extractor = ActivationExtractor(model)
        self.differ = ActivationDiff()
        self.divergence_detector = BehavioralInternalDivergenceDetector()
        self.max_divergence_ratio = max_divergence_ratio
        self.max_safety_shift = max_safety_shift
        self.block_on_critical = block_on_critical

        self._before_snapshot: Optional[ActivationSnapshot] = None
        self._iteration = 0

    def before_modification(self) -> ActivationSnapshot:
        """Capture activation snapshot before model modification."""
        probes = self.probe_set.get_all()
        self._before_snapshot = self.extractor.extract(probes)
        self._before_snapshot.metadata["iteration"] = self._iteration
        self._before_snapshot.metadata["phase"] = "before"
        return self._before_snapshot

    def after_modification(self, behavioral_change: float = 0.0) -> InterpretabilityCheckResult:
        """Check activations after modification and decide if it should be blocked.

        Args:
            behavioral_change: Magnitude of behavioral change [0, 1]

        Returns:
            InterpretabilityCheckResult with blocking decision
        """
        if self._before_snapshot is None:
            return InterpretabilityCheckResult(
                should_block=False,
                reason="No before snapshot available",
            )

        # Extract after snapshot
        probes = self.probe_set.get_all()
        after_snapshot = self.extractor.extract(probes)
        after_snapshot.metadata["iteration"] = self._iteration
        after_snapshot.metadata["phase"] = "after"

        # Compute diff
        diff_result = self.differ.compute(self._before_snapshot, after_snapshot)

        # Check divergence
        div_result = self.divergence_detector.check(
            diff_result, behavioral_change, self._iteration
        )

        # Determine blocking
        warnings = []
        should_block = False
        reason = "OK"

        if div_result.divergence_ratio > self.max_divergence_ratio:
            warnings.append(
                f"High divergence ratio: {div_result.divergence_ratio:.2f} > {self.max_divergence_ratio}"
            )
            if self.block_on_critical:
                should_block = True
                reason = f"Divergence ratio {div_result.divergence_ratio:.2f} exceeds maximum {self.max_divergence_ratio}"

        if diff_result.safety_disproportionate:
            warnings.append(
                f"Safety-disproportionate change: ratio={diff_result.safety_change_ratio:.2f}"
            )
            if diff_result.safety_change_ratio > self.max_safety_shift and self.block_on_critical:
                should_block = True
                reason = f"Safety change ratio {diff_result.safety_change_ratio:.2f} exceeds maximum"

        if div_result.is_anomalous:
            warnings.append(f"Anomalous divergence (z-score={div_result.z_score:.2f})")

        self._iteration += 1
        self._before_snapshot = None

        return InterpretabilityCheckResult(
            should_block=should_block,
            reason=reason,
            divergence_result=div_result,
            diff_result=diff_result,
            warnings=warnings,
        )

    def should_block(self, check_result: InterpretabilityCheckResult) -> bool:
        """Determine if a modification should be blocked."""
        return check_result.should_block
