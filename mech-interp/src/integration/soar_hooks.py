"""Interpretability hooks for the SOAR training system."""

from typing import Any, Optional, List, Dict
import numpy as np

from src.probing.probe_set import ProbeSet
from src.probing.extractor import ActivationExtractor, ActivationSnapshot
from src.probing.diff import ActivationDiff
from src.attention.head_extractor import HeadExtractor
from src.attention.specialization import HeadSpecializationTracker
from src.anomaly.divergence_detector import BehavioralInternalDivergenceDetector


class SOARInterpretabilityHooks:
    """Hooks for SOAR (Self-Organized Autonomous Reinforcement) integration.

    Monitors training steps and epochs for interpretability signals.
    """

    def __init__(self, model: Any,
                 probe_set: Optional[ProbeSet] = None,
                 check_every_n_steps: int = 10):
        self.model = model
        self.probe_set = probe_set or ProbeSet()
        self.extractor = ActivationExtractor(model)
        self.head_extractor = HeadExtractor(model)
        self.differ = ActivationDiff()
        self.divergence_detector = BehavioralInternalDivergenceDetector()
        self.head_tracker = HeadSpecializationTracker()
        self.check_every_n_steps = check_every_n_steps

        self._step_count = 0
        self._epoch_count = 0
        self._last_snapshot: Optional[ActivationSnapshot] = None
        self._anomalies: List[Dict] = []

    def after_training_step(self, step: int, loss: float = 0.0,
                            behavioral_change: float = 0.0) -> Optional[Dict]:
        """Called after each training step.

        Returns check result dict if a check was performed, None otherwise.
        """
        self._step_count = step

        if step % self.check_every_n_steps != 0:
            return None

        probes = self.probe_set.get_all()
        current_snapshot = self.extractor.extract(probes)

        result = {"step": step, "loss": loss, "checked": True}

        if self._last_snapshot is not None:
            diff = self.differ.compute(self._last_snapshot, current_snapshot)
            div_result = self.divergence_detector.check(
                diff, behavioral_change, step
            )
            result["divergence_ratio"] = div_result.divergence_ratio
            result["is_anomalous"] = div_result.is_anomalous
            result["safety_flag"] = div_result.safety_flag

            if div_result.is_anomalous:
                self._anomalies.append({
                    "step": step,
                    "type": "divergence",
                    "ratio": div_result.divergence_ratio,
                    "z_score": div_result.z_score,
                })

        self._last_snapshot = current_snapshot
        return result

    def after_training_epoch(self, epoch: int,
                             behavioral_change: float = 0.0) -> Dict:
        """Called after each training epoch. Always performs a full check."""
        self._epoch_count = epoch

        probes = self.probe_set.get_all()
        current_snapshot = self.extractor.extract(probes)

        # Head analysis
        head_stats = self.head_extractor.extract_aggregate_stats(probes[:5])
        head_result = self.head_tracker.track(head_stats)

        result = {
            "epoch": epoch,
            "head_tracking": head_result.to_dict(),
        }

        if self._last_snapshot is not None:
            diff = self.differ.compute(self._last_snapshot, current_snapshot)
            div_result = self.divergence_detector.check(
                diff, behavioral_change, epoch * 1000
            )
            result["divergence_ratio"] = div_result.divergence_ratio
            result["is_anomalous"] = div_result.is_anomalous
            result["safety_flag"] = div_result.safety_flag
            result["safety_disproportionate"] = diff.safety_disproportionate

        self._last_snapshot = current_snapshot
        return result

    def detect_training_anomalies(self) -> List[Dict]:
        """Return all detected training anomalies."""
        return list(self._anomalies)

    def get_step_count(self) -> int:
        return self._step_count

    def get_epoch_count(self) -> int:
        return self._epoch_count
