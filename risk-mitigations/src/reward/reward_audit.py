"""Reward audit trail for tracking and analyzing reward signals.

Logs all reward signals with inputs/outputs and detects suspicious patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import Counter


@dataclass
class AuditEntry:
    """Single audit trail entry."""
    input_data: Any
    output_data: Any
    reward: float
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternReport:
    """Report on detected reward patterns."""
    patterns_detected: List[str]
    suspicious: bool
    reward_trend: str  # "increasing", "decreasing", "stable", "volatile"
    mean_reward: float
    std_reward: float
    details: Dict[str, Any] = field(default_factory=dict)


class RewardAuditTrail:
    """Audit trail for reward signals.

    Logs every reward signal and can detect suspicious patterns
    that may indicate reward hacking.
    """

    def __init__(self, max_entries: int = 10000):
        self._entries: List[AuditEntry] = []
        self._max_entries = max_entries
        self._iteration_counter = 0

    def log(
        self,
        input_data: Any,
        output_data: Any,
        reward: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Log a reward signal.

        Args:
            input_data: The input that produced the output.
            output_data: The output that received the reward.
            reward: The reward value.
            metadata: Optional additional metadata.

        Returns:
            The created AuditEntry.
        """
        entry = AuditEntry(
            input_data=input_data,
            output_data=output_data,
            reward=reward,
            iteration=self._iteration_counter,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._iteration_counter += 1

        # Enforce max size
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        return entry

    def detect_patterns(self) -> PatternReport:
        """Analyze the audit trail for suspicious patterns.

        Returns:
            PatternReport with detected patterns.
        """
        if not self._entries:
            return PatternReport(
                patterns_detected=[],
                suspicious=False,
                reward_trend="stable",
                mean_reward=0.0,
                std_reward=0.0,
            )

        rewards = [e.reward for e in self._entries]
        n = len(rewards)
        mean_r = sum(rewards) / n
        var_r = sum((r - mean_r) ** 2 for r in rewards) / n if n > 1 else 0.0
        std_r = var_r ** 0.5

        patterns = []
        suspicious = False
        details: Dict[str, Any] = {}

        # Detect monotonically increasing rewards (possible gaming)
        if n >= 5:
            recent = rewards[-5:]
            if all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1)):
                patterns.append("monotonically_increasing_rewards")
                suspicious = True

        # Detect reward clustering (same reward repeated)
        reward_counts = Counter(round(r, 2) for r in rewards)
        most_common_count = reward_counts.most_common(1)[0][1] if reward_counts else 0
        if most_common_count > n * 0.5 and n >= 10:
            patterns.append("reward_clustering")
            suspicious = True
            details["most_common_reward"] = reward_counts.most_common(1)[0][0]

        # Detect sudden jumps
        if n >= 3:
            for i in range(1, n):
                if abs(rewards[i] - rewards[i - 1]) > 3 * std_r and std_r > 0:
                    patterns.append("sudden_reward_jump")
                    suspicious = True
                    break

        # Determine trend
        if n >= 3:
            first_half_mean = sum(rewards[: n // 2]) / (n // 2)
            second_half_mean = sum(rewards[n // 2 :]) / (n - n // 2)
            diff = second_half_mean - first_half_mean

            if diff > 0.1 * abs(mean_r) if mean_r != 0 else diff > 0.1:
                trend = "increasing"
            elif diff < -0.1 * abs(mean_r) if mean_r != 0 else diff < -0.1:
                trend = "decreasing"
            elif std_r > abs(mean_r) * 0.5 if mean_r != 0 else std_r > 0.5:
                trend = "volatile"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return PatternReport(
            patterns_detected=patterns,
            suspicious=suspicious,
            reward_trend=trend,
            mean_reward=mean_r,
            std_reward=std_r,
            details=details,
        )

    def get_entries(self, last_n: Optional[int] = None) -> List[AuditEntry]:
        """Return audit entries, optionally limited to last N."""
        if last_n is not None:
            return list(self._entries[-last_n:])
        return list(self._entries)

    def get_reward_history(self) -> List[float]:
        """Return just the reward values."""
        return [e.reward for e in self._entries]

    @property
    def entry_count(self) -> int:
        return len(self._entries)
