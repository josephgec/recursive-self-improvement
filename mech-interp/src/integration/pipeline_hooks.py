"""Interpretability hooks for the general improvement pipeline."""

from typing import Any, Optional, Dict, List

from src.probing.probe_set import ProbeSet
from src.probing.extractor import ActivationExtractor, ActivationSnapshot
from src.probing.diff import ActivationDiff
from src.anomaly.divergence_detector import BehavioralInternalDivergenceDetector
from src.attention.head_extractor import HeadExtractor
from src.attention.specialization import HeadSpecializationTracker
from src.anomaly.deceptive_alignment import DeceptiveAlignmentProber
from src.monitoring.alert_rules import InterpretabilityAlertRules
from src.monitoring.time_series import InterpretabilityTimeSeries


class PipelineInterpretabilityHooks:
    """Hooks for the general self-improvement pipeline.

    Provides after_iteration callback that runs all interpretability checks.
    """

    def __init__(self, model: Any,
                 probe_set: Optional[ProbeSet] = None,
                 run_deceptive_probes: bool = True):
        self.model = model
        self.probe_set = probe_set or ProbeSet()
        self.extractor = ActivationExtractor(model)
        self.head_extractor = HeadExtractor(model)
        self.differ = ActivationDiff()
        self.divergence_detector = BehavioralInternalDivergenceDetector()
        self.head_tracker = HeadSpecializationTracker()
        self.alert_rules = InterpretabilityAlertRules()
        self.time_series = InterpretabilityTimeSeries()
        self.run_deceptive_probes = run_deceptive_probes

        self._last_snapshot: Optional[ActivationSnapshot] = None
        self._iteration = 0
        self._results: List[Dict] = []

    def after_iteration(self, iteration: int,
                        behavioral_change: float = 0.0,
                        probe_accuracy: float = 0.9,
                        output_accuracy: float = 0.7) -> Dict:
        """Run all interpretability checks after a pipeline iteration.

        Returns a comprehensive results dict.
        """
        self._iteration = iteration
        probes = self.probe_set.get_all()

        # Extract activations
        current_snapshot = self.extractor.extract(probes)

        result: Dict[str, Any] = {"iteration": iteration}

        # Diff and divergence
        if self._last_snapshot is not None:
            diff = self.differ.compute(self._last_snapshot, current_snapshot)
            div_result = self.divergence_detector.check(
                diff, behavioral_change, iteration
            )
            result["divergence"] = div_result.to_dict()
            result["diff_summary"] = {
                "most_changed_layers": diff.most_changed_layers,
                "safety_disproportionate": diff.safety_disproportionate,
                "overall_change": diff.overall_change_magnitude,
            }
        else:
            result["divergence"] = None
            result["diff_summary"] = None

        # Head tracking
        head_stats = self.head_extractor.extract_aggregate_stats(probes[:5])
        head_result = self.head_tracker.track(head_stats)
        result["head_tracking"] = head_result.to_dict()

        # Deceptive alignment probes
        if self.run_deceptive_probes:
            da_prober = DeceptiveAlignmentProber(self.model)
            da_report = da_prober.run_all_probes(probe_accuracy, output_accuracy)
            result["deceptive_alignment"] = da_report.to_dict()
        else:
            result["deceptive_alignment"] = None

        # Alert rules
        alerts = self.alert_rules.evaluate(result)
        result["alerts"] = alerts

        # Time series
        self.time_series.record(iteration, {
            "divergence_ratio": result.get("divergence", {}).get("divergence_ratio", 0) if result.get("divergence") else 0,
            "behavioral_change": behavioral_change,
            "num_alerts": len(alerts),
        })

        self._last_snapshot = current_snapshot
        self._results.append(result)
        return result

    def get_results_history(self) -> List[Dict]:
        """Return all iteration results."""
        return list(self._results)
