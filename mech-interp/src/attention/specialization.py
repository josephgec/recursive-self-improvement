"""Track attention head specialization over time."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

from src.probing.extractor import HeadStats


@dataclass
class HeadShift:
    """Represents a shift in head behavior."""
    layer: int
    head: int
    entropy_before: float
    entropy_after: float
    entropy_change: float
    specialization_before: float
    specialization_after: float


@dataclass
class HeadRoleChange:
    """Represents a change in head role classification."""
    layer: int
    head: int
    role_before: str
    role_after: str
    iteration: int
    magnitude: float


@dataclass
class HeadTrackingResult:
    """Result of head specialization tracking."""
    shifts: List[HeadShift] = field(default_factory=list)
    dying_heads: List[Tuple[int, int]] = field(default_factory=list)
    narrowing_heads: List[Tuple[int, int]] = field(default_factory=list)
    role_changes: List[HeadRoleChange] = field(default_factory=list)
    summary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "num_shifts": len(self.shifts),
            "num_dying_heads": len(self.dying_heads),
            "num_narrowing_heads": len(self.narrowing_heads),
            "num_role_changes": len(self.role_changes),
            "dying_heads": [list(h) for h in self.dying_heads],
            "narrowing_heads": [list(h) for h in self.narrowing_heads],
            "summary": self.summary,
        }


def measure_specialization(entropy: float, max_entropy: float = 3.0) -> float:
    """Convert entropy to a specialization score (0=uniform, 1=specialized)."""
    if max_entropy <= 0:
        return 0.0
    return max(0.0, 1.0 - entropy / max_entropy)


class HeadSpecializationTracker:
    """Track head specialization over iterations."""

    def __init__(self, dying_threshold: float = 0.05,
                 narrowing_entropy_drop: float = 0.3,
                 role_change_threshold: float = 0.5):
        self.dying_threshold = dying_threshold
        self.narrowing_entropy_drop = narrowing_entropy_drop
        self.role_change_threshold = role_change_threshold
        self.history: List[Dict[Tuple[int, int], HeadStats]] = []
        self.iteration = 0

    def track(self, head_stats: List[HeadStats]) -> HeadTrackingResult:
        """Record head stats and compute shifts from previous iteration."""
        current = {}
        for hs in head_stats:
            current[(hs.layer, hs.head)] = hs

        result = HeadTrackingResult()

        if self.history:
            previous = self.history[-1]
            result = self.compute_shifts(previous, current)

        self.history.append(current)
        self.iteration += 1

        # Add dying and narrowing head detection
        result.dying_heads = self.detect_dying_heads(head_stats)
        result.narrowing_heads = self.detect_narrowing_heads()

        # Summary
        if head_stats:
            entropies = [hs.entropy for hs in head_stats]
            result.summary = {
                "mean_entropy": float(np.mean(entropies)),
                "std_entropy": float(np.std(entropies)),
                "min_entropy": float(np.min(entropies)),
                "max_entropy": float(np.max(entropies)),
                "num_heads": len(head_stats),
            }

        return result

    def compute_shifts(self, previous: Dict[Tuple[int, int], HeadStats],
                       current: Dict[Tuple[int, int], HeadStats]) -> HeadTrackingResult:
        """Compute shifts between two sets of head stats."""
        result = HeadTrackingResult()
        common_keys = set(previous.keys()) & set(current.keys())

        for key in sorted(common_keys):
            prev = previous[key]
            curr = current[key]
            entropy_change = curr.entropy - prev.entropy
            spec_before = measure_specialization(prev.entropy)
            spec_after = measure_specialization(curr.entropy)

            if abs(entropy_change) > 0.01:
                result.shifts.append(HeadShift(
                    layer=key[0],
                    head=key[1],
                    entropy_before=prev.entropy,
                    entropy_after=curr.entropy,
                    entropy_change=entropy_change,
                    specialization_before=spec_before,
                    specialization_after=spec_after,
                ))

            # Role change detection
            role_before = self._classify_role(prev)
            role_after = self._classify_role(curr)
            if role_before != role_after:
                result.role_changes.append(HeadRoleChange(
                    layer=key[0],
                    head=key[1],
                    role_before=role_before,
                    role_after=role_after,
                    iteration=self.iteration,
                    magnitude=abs(entropy_change),
                ))

        return result

    def detect_dying_heads(self, head_stats: List[HeadStats]) -> List[Tuple[int, int]]:
        """Detect heads with near-uniform (very high entropy / low max attention)."""
        dying = []
        for hs in head_stats:
            # A dying head has very low max attention (near uniform)
            if hs.max_attention < self.dying_threshold:
                dying.append((hs.layer, hs.head))
        return dying

    def detect_narrowing_heads(self) -> List[Tuple[int, int]]:
        """Detect heads whose entropy has been consistently dropping."""
        if len(self.history) < 2:
            return []

        narrowing = []
        current = self.history[-1]
        previous = self.history[-2]

        for key in set(current.keys()) & set(previous.keys()):
            curr = current[key]
            prev = previous[key]
            if prev.entropy - curr.entropy > self.narrowing_entropy_drop:
                narrowing.append(key)

        return narrowing

    def detect_role_changes(self) -> List[HeadRoleChange]:
        """Detect role changes across full history."""
        if len(self.history) < 2:
            return []

        changes = []
        for i in range(1, len(self.history)):
            prev = self.history[i - 1]
            curr = self.history[i]
            for key in set(prev.keys()) & set(curr.keys()):
                role_before = self._classify_role(prev[key])
                role_after = self._classify_role(curr[key])
                if role_before != role_after:
                    changes.append(HeadRoleChange(
                        layer=key[0],
                        head=key[1],
                        role_before=role_before,
                        role_after=role_after,
                        iteration=i,
                        magnitude=abs(curr[key].entropy - prev[key].entropy),
                    ))
        return changes

    def _classify_role(self, hs: HeadStats) -> str:
        """Classify a head's role based on its stats."""
        if hs.max_attention < self.dying_threshold:
            return "uniform"
        if hs.sparsity > 0.8:
            return "sparse"
        if hs.entropy < 0.5:
            return "local"
        return "global"
