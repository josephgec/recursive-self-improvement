"""Detect dead (near-uniform) attention heads."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from src.probing.extractor import HeadStats


@dataclass
class DeadHead:
    """A detected dead attention head."""
    layer: int
    head: int
    max_attention: float
    entropy: float
    reason: str


class DeadHeadDetector:
    """Detect attention heads that are effectively dead (near-uniform)."""

    def __init__(self, max_attention_threshold: float = 0.05,
                 entropy_threshold: float = None):
        self.max_attention_threshold = max_attention_threshold
        self.entropy_threshold = entropy_threshold

    def detect(self, head_stats: List[HeadStats]) -> List[DeadHead]:
        """Detect dead heads from a list of head statistics."""
        dead = []
        for hs in head_stats:
            reasons = []

            # Low max attention = near-uniform distribution
            if hs.max_attention < self.max_attention_threshold:
                reasons.append(f"max_attention={hs.max_attention:.4f} < {self.max_attention_threshold}")

            # Optionally check high entropy
            if self.entropy_threshold is not None and hs.entropy > self.entropy_threshold:
                reasons.append(f"entropy={hs.entropy:.4f} > {self.entropy_threshold}")

            if reasons:
                dead.append(DeadHead(
                    layer=hs.layer,
                    head=hs.head,
                    max_attention=hs.max_attention,
                    entropy=hs.entropy,
                    reason="; ".join(reasons),
                ))
        return dead

    def detect_mass_death(self, head_stats: List[HeadStats],
                          threshold_fraction: float = 0.3) -> bool:
        """Check if a large fraction of heads are dead."""
        if not head_stats:
            return False
        dead = self.detect(head_stats)
        return len(dead) / len(head_stats) > threshold_fraction

    def get_dead_head_ids(self, head_stats: List[HeadStats]) -> List[Tuple[int, int]]:
        """Return (layer, head) tuples for dead heads."""
        dead = self.detect(head_stats)
        return [(d.layer, d.head) for d in dead]
