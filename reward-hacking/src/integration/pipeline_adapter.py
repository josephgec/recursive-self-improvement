from __future__ import annotations

"""Pipeline adapter for composing reward processing components."""

import numpy as np
from dataclasses import dataclass, field

from ..bounding.process_reward import ProcessRewardShaper, ShapedReward
from ..bounding.reward_monitor import RewardMonitor


@dataclass
class PipelineResult:
    """Result from the reward processing pipeline."""

    raw_rewards: list[float]
    shaped_rewards: list[float]
    anomaly_detected: bool
    trend: str
    step: int


class PipelineRewardAdapter:
    """Integrates reward shaping with monitoring in a pipeline.

    Composes the ProcessRewardShaper with RewardMonitor for
    a complete reward processing and monitoring pipeline.
    """

    def __init__(
        self,
        shaper: ProcessRewardShaper | None = None,
        monitor: RewardMonitor | None = None,
    ):
        self._shaper = shaper or ProcessRewardShaper()
        self._monitor = monitor or RewardMonitor()
        self._step = 0
        self._results: list[PipelineResult] = []

    @property
    def shaper(self) -> ProcessRewardShaper:
        return self._shaper

    @property
    def monitor(self) -> RewardMonitor:
        return self._monitor

    @property
    def results(self) -> list[PipelineResult]:
        return list(self._results)

    def process(self, raw_rewards: np.ndarray) -> PipelineResult:
        """Process a batch of raw rewards through the pipeline.

        Args:
            raw_rewards: Array of raw reward values.

        Returns:
            PipelineResult with shaped rewards and monitoring info.
        """
        self._step += 1
        raw_list = list(np.asarray(raw_rewards).flatten())

        # Shape rewards
        shaped_results = self._shaper.shape_batch(np.array(raw_list))
        shaped_list = [sr.final for sr in shaped_results]

        # Monitor
        self._monitor.record_batch(np.array(raw_list))
        self._monitor.take_snapshot()
        anomaly = self._monitor.detect_anomaly()
        trend = self._monitor.get_trend()

        result = PipelineResult(
            raw_rewards=raw_list,
            shaped_rewards=shaped_list,
            anomaly_detected=anomaly.is_anomalous,
            trend=trend,
            step=self._step,
        )
        self._results.append(result)
        return result

    def reset(self) -> None:
        """Reset the pipeline."""
        self._shaper.reset()
        self._step = 0
        self._results.clear()
