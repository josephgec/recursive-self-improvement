"""Track attention head roles over iterations."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.probing.extractor import HeadStats


@dataclass
class HeadRole:
    """Role classification for an attention head."""
    layer: int
    head: int
    role: str  # "local", "global", "sparse", "uniform"
    confidence: float
    iteration: int


class HeadRoleTracker:
    """Track and classify attention head roles over iterations."""

    def __init__(self, local_threshold: float = 0.5,
                 sparse_threshold: float = 0.8,
                 uniform_threshold: float = 0.05):
        self.local_threshold = local_threshold
        self.sparse_threshold = sparse_threshold
        self.uniform_threshold = uniform_threshold
        self.history: List[Dict[Tuple[int, int], HeadRole]] = []
        self.iteration = 0

    def classify_role(self, hs: HeadStats) -> str:
        """Classify a head's role based on its stats.

        - "uniform": near-uniform attention (dying head)
        - "sparse": very sparse attention pattern
        - "local": low entropy, focused attention
        - "global": moderate-high entropy, distributed attention
        """
        if hs.max_attention < self.uniform_threshold:
            return "uniform"
        if hs.sparsity > self.sparse_threshold:
            return "sparse"
        if hs.entropy < self.local_threshold:
            return "local"
        return "global"

    def classify_with_confidence(self, hs: HeadStats) -> Tuple[str, float]:
        """Classify role with confidence score."""
        role = self.classify_role(hs)
        if role == "uniform":
            conf = 1.0 - hs.max_attention / self.uniform_threshold if self.uniform_threshold > 0 else 1.0
        elif role == "sparse":
            conf = (hs.sparsity - self.sparse_threshold) / (1.0 - self.sparse_threshold + 1e-10)
        elif role == "local":
            conf = 1.0 - hs.entropy / self.local_threshold if self.local_threshold > 0 else 1.0
        else:
            conf = min(1.0, hs.entropy / 3.0)
        return role, max(0.0, min(1.0, conf))

    def track(self, head_stats: List[HeadStats]) -> Dict[Tuple[int, int], HeadRole]:
        """Track head roles for current iteration."""
        current = {}
        for hs in head_stats:
            role, confidence = self.classify_with_confidence(hs)
            key = (hs.layer, hs.head)
            current[key] = HeadRole(
                layer=hs.layer,
                head=hs.head,
                role=role,
                confidence=confidence,
                iteration=self.iteration,
            )
        self.history.append(current)
        self.iteration += 1
        return current

    def get_role_history(self, layer: int, head: int) -> List[str]:
        """Get role history for a specific head."""
        key = (layer, head)
        return [
            snap[key].role for snap in self.history
            if key in snap
        ]

    def get_role_changes(self) -> List[Dict]:
        """Get all role changes across history."""
        changes = []
        for i in range(1, len(self.history)):
            prev = self.history[i - 1]
            curr = self.history[i]
            for key in set(prev.keys()) & set(curr.keys()):
                if prev[key].role != curr[key].role:
                    changes.append({
                        "layer": key[0],
                        "head": key[1],
                        "iteration": i,
                        "from": prev[key].role,
                        "to": curr[key].role,
                    })
        return changes

    def get_stable_roles(self) -> Dict[Tuple[int, int], str]:
        """Return heads whose role has been consistent throughout history."""
        if not self.history:
            return {}

        all_keys = set()
        for snap in self.history:
            all_keys.update(snap.keys())

        stable = {}
        for key in all_keys:
            roles = [snap[key].role for snap in self.history if key in snap]
            if len(roles) == len(self.history) and len(set(roles)) == 1:
                stable[key] = roles[0]
        return stable
