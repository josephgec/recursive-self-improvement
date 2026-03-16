"""Frequency conditions and policies for the modification frequency experiment."""

from dataclasses import dataclass
from typing import Any, List, Optional


class ModificationFrequencyPolicy:
    """Determines when the pipeline should self-modify."""

    def __init__(self, policy_type: str, param: Optional[Any] = None):
        self.policy_type = policy_type
        self.param = param
        self._no_improve_count = 0
        self._last_accuracy = 0.0

    def should_modify(self, iteration: int, accuracy: float) -> bool:
        """Whether the pipeline should modify itself at this iteration."""
        if self.policy_type == "every_task":
            return True
        elif self.policy_type == "every_n":
            n = self.param or 5
            return iteration > 0 and iteration % n == 0
        elif self.policy_type == "threshold":
            threshold = self.param or 0.9
            return accuracy >= threshold
        elif self.policy_type == "plateau":
            plateau_len = self.param or 5
            if accuracy > self._last_accuracy + 0.001:
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
            self._last_accuracy = accuracy
            return self._no_improve_count >= plateau_len
        elif self.policy_type == "never":
            return False
        return False

    def reset(self):
        """Reset internal state."""
        self._no_improve_count = 0
        self._last_accuracy = 0.0


@dataclass
class FrequencyCondition:
    """A single condition in the modification frequency experiment."""

    name: str
    description: str
    policy: ModificationFrequencyPolicy


def build_frequency_conditions() -> List[FrequencyCondition]:
    """Build all 7 frequency conditions."""
    return [
        FrequencyCondition(
            name="every_task",
            description="Modify after every single task",
            policy=ModificationFrequencyPolicy("every_task"),
        ),
        FrequencyCondition(
            name="every_5",
            description="Modify every 5 iterations",
            policy=ModificationFrequencyPolicy("every_n", param=5),
        ),
        FrequencyCondition(
            name="every_10",
            description="Modify every 10 iterations",
            policy=ModificationFrequencyPolicy("every_n", param=10),
        ),
        FrequencyCondition(
            name="every_20",
            description="Modify every 20 iterations",
            policy=ModificationFrequencyPolicy("every_n", param=20),
        ),
        FrequencyCondition(
            name="threshold_90",
            description="Modify only when accuracy exceeds 90%",
            policy=ModificationFrequencyPolicy("threshold", param=0.9),
        ),
        FrequencyCondition(
            name="plateau_5",
            description="Modify after 5 iterations without improvement",
            policy=ModificationFrequencyPolicy("plateau", param=5),
        ),
        FrequencyCondition(
            name="never",
            description="Never modify (baseline)",
            policy=ModificationFrequencyPolicy("never"),
        ),
    ]
