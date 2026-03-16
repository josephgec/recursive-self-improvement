"""Hindsight conditions and policies for the hindsight target experiment."""

from dataclasses import dataclass
from typing import List


class HindsightTargetPolicy:
    """Determines what the hindsight rationalization targets."""

    def __init__(self, policy_type: str, total_iterations: int = 20):
        self.policy_type = policy_type
        self.total_iterations = total_iterations

    def get_target(self, iteration: int) -> str:
        """Return the hindsight target for this iteration.

        Returns one of: 'weights', 'library', 'both', 'none'.
        """
        if self.policy_type == "weights_only":
            return "weights"
        elif self.policy_type == "library_only":
            return "library"
        elif self.policy_type == "both":
            return "both"
        elif self.policy_type == "none":
            return "none"
        elif self.policy_type == "weights_then_library":
            midpoint = self.total_iterations // 2
            return "weights" if iteration < midpoint else "library"
        elif self.policy_type == "library_then_weights":
            midpoint = self.total_iterations // 2
            return "library" if iteration < midpoint else "weights"
        return "none"


@dataclass
class HindsightCondition:
    """A single condition in the hindsight target experiment."""

    name: str
    description: str
    policy: HindsightTargetPolicy


def build_hindsight_conditions(total_iterations: int = 20) -> List[HindsightCondition]:
    """Build all 6 hindsight conditions."""
    return [
        HindsightCondition(
            name="weights_only",
            description="Hindsight targets model weights only",
            policy=HindsightTargetPolicy("weights_only", total_iterations),
        ),
        HindsightCondition(
            name="library_only",
            description="Hindsight targets code library only",
            policy=HindsightTargetPolicy("library_only", total_iterations),
        ),
        HindsightCondition(
            name="both",
            description="Hindsight targets both weights and library",
            policy=HindsightTargetPolicy("both", total_iterations),
        ),
        HindsightCondition(
            name="none",
            description="No hindsight rationalization (baseline)",
            policy=HindsightTargetPolicy("none", total_iterations),
        ),
        HindsightCondition(
            name="weights_then_library",
            description="First half targets weights, second half targets library",
            policy=HindsightTargetPolicy("weights_then_library", total_iterations),
        ),
        HindsightCondition(
            name="library_then_weights",
            description="First half targets library, second half targets weights",
            policy=HindsightTargetPolicy("library_then_weights", total_iterations),
        ),
    ]
