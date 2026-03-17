from __future__ import annotations

"""Reward distribution monitoring and anomaly detection."""

import numpy as np
from dataclasses import dataclass


@dataclass
class AnomalyReport:
    """Report from anomaly detection."""

    is_anomalous: bool
    z_score: float
    current_mean: float
    baseline_mean: float
    current_std: float
    description: str


@dataclass
class DistributionSnapshot:
    """Snapshot of reward distribution at a point in time."""

    step: int
    mean: float
    std: float
    min_val: float
    max_val: float
    skewness: float
    count: int


class RewardMonitor:
    """Monitors reward distribution over time and detects anomalies.

    Tracks distribution statistics and flags unusual shifts that
    may indicate reward hacking.
    """

    def __init__(self, window: int = 50, z_threshold: float = 3.0):
        self._window = window
        self._z_threshold = z_threshold
        self._all_rewards: list[float] = []
        self._snapshots: list[DistributionSnapshot] = []
        self._anomalies: list[AnomalyReport] = []
        self._step = 0

    @property
    def snapshots(self) -> list[DistributionSnapshot]:
        return list(self._snapshots)

    @property
    def anomalies(self) -> list[AnomalyReport]:
        return list(self._anomalies)

    @property
    def reward_count(self) -> int:
        return len(self._all_rewards)

    def record(self, reward: float) -> None:
        """Record a single reward value."""
        self._all_rewards.append(reward)
        self._step += 1

    def record_batch(self, rewards: np.ndarray) -> None:
        """Record a batch of reward values."""
        for r in np.asarray(rewards).flatten():
            self.record(float(r))

    def take_snapshot(self) -> DistributionSnapshot | None:
        """Take a snapshot of the current reward distribution.

        Uses the most recent window of rewards.
        """
        if not self._all_rewards:
            return None

        recent = self._all_rewards[-self._window:]
        arr = np.array(recent)

        mean = float(np.mean(arr))
        std = float(np.std(arr))

        # Compute skewness
        if std > 1e-10 and len(arr) > 2:
            skewness = float(np.mean(((arr - mean) / std) ** 3))
        else:
            skewness = 0.0

        snapshot = DistributionSnapshot(
            step=self._step,
            mean=mean,
            std=std,
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
            skewness=skewness,
            count=len(arr),
        )
        self._snapshots.append(snapshot)
        return snapshot

    def detect_anomaly(self) -> AnomalyReport:
        """Check for anomalies in recent reward distribution.

        Compares recent window to overall baseline.
        """
        if len(self._all_rewards) < self._window * 2:
            return AnomalyReport(
                is_anomalous=False,
                z_score=0.0,
                current_mean=float(np.mean(self._all_rewards)) if self._all_rewards else 0.0,
                baseline_mean=float(np.mean(self._all_rewards)) if self._all_rewards else 0.0,
                current_std=float(np.std(self._all_rewards)) if self._all_rewards else 0.0,
                description="Insufficient data for anomaly detection",
            )

        recent = np.array(self._all_rewards[-self._window:])
        baseline = np.array(self._all_rewards[:-self._window])

        current_mean = float(np.mean(recent))
        baseline_mean = float(np.mean(baseline))
        baseline_std = float(np.std(baseline))
        current_std = float(np.std(recent))

        if baseline_std < 1e-10:
            z_score = 0.0
        else:
            z_score = abs(current_mean - baseline_mean) / baseline_std

        is_anomalous = z_score > self._z_threshold

        if is_anomalous:
            direction = "increase" if current_mean > baseline_mean else "decrease"
            description = (
                f"Anomalous reward {direction} detected: "
                f"z-score={z_score:.2f} (threshold={self._z_threshold})"
            )
        else:
            description = "No anomaly detected"

        report = AnomalyReport(
            is_anomalous=is_anomalous,
            z_score=z_score,
            current_mean=current_mean,
            baseline_mean=baseline_mean,
            current_std=current_std,
            description=description,
        )
        self._anomalies.append(report)
        return report

    def get_trend(self, num_windows: int = 5) -> str:
        """Analyze the trend in reward distribution.

        Returns "increasing", "decreasing", "stable", or "volatile".
        """
        if len(self._all_rewards) < self._window * num_windows:
            return "insufficient_data"

        window_means = []
        for i in range(num_windows):
            start = len(self._all_rewards) - self._window * (num_windows - i)
            end = start + self._window
            window_means.append(float(np.mean(self._all_rewards[start:end])))

        diffs = np.diff(window_means)
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))

        if std_diff > abs(mean_diff) * 2:
            return "volatile"
        elif mean_diff > 0.1:
            return "increasing"
        elif mean_diff < -0.1:
            return "decreasing"
        else:
            return "stable"
