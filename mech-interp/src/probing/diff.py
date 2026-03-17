"""Compute differences between activation snapshots."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

from src.probing.extractor import ActivationSnapshot, LayerStats


@dataclass
class LayerDiff:
    """Difference metrics for a single layer."""
    layer_name: str
    mean_shift: float
    std_shift: float
    norm_shift: float
    direction_similarity: float  # cosine similarity of activation vectors
    per_probe_changes: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "layer_name": self.layer_name,
            "mean_shift": self.mean_shift,
            "std_shift": self.std_shift,
            "norm_shift": self.norm_shift,
            "direction_similarity": self.direction_similarity,
            "per_probe_changes": self.per_probe_changes,
        }


@dataclass
class ActivationDiffResult:
    """Result of comparing two activation snapshots."""
    layer_diffs: Dict[str, LayerDiff] = field(default_factory=dict)
    most_changed_layers: List[str] = field(default_factory=list)
    most_changed_probes: List[str] = field(default_factory=list)
    safety_disproportionate: bool = False
    safety_change_ratio: float = 0.0
    overall_change_magnitude: float = 0.0

    def to_dict(self) -> dict:
        return {
            "layer_diffs": {k: v.to_dict() for k, v in self.layer_diffs.items()},
            "most_changed_layers": self.most_changed_layers,
            "most_changed_probes": self.most_changed_probes,
            "safety_disproportionate": self.safety_disproportionate,
            "safety_change_ratio": self.safety_change_ratio,
            "overall_change_magnitude": self.overall_change_magnitude,
        }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class ActivationDiff:
    """Computes differences between two activation snapshots."""

    def __init__(self, safety_disproportionate_factor: float = 2.0):
        self.safety_disproportionate_factor = safety_disproportionate_factor

    def compute(self, before: ActivationSnapshot, after: ActivationSnapshot) -> ActivationDiffResult:
        """Compare two snapshots and return diff result."""
        result = ActivationDiffResult()

        # Find common probes and layers
        common_probes = set(before.get_probe_ids()) & set(after.get_probe_ids())
        all_layers = set(before.get_all_layer_names()) & set(after.get_all_layer_names())

        if not common_probes or not all_layers:
            return result

        # Per-layer diffs
        layer_changes = {}
        for layer_name in sorted(all_layers):
            per_probe_changes = {}
            mean_shifts = []
            std_shifts = []
            norm_shifts = []
            cosine_sims = []

            for probe_id in sorted(common_probes):
                before_stats = before.get_layer_stats(probe_id, layer_name)
                after_stats = after.get_layer_stats(probe_id, layer_name)

                if before_stats is None or after_stats is None:
                    continue

                ms = abs(after_stats.mean - before_stats.mean)
                mean_shifts.append(ms)
                std_shifts.append(abs(after_stats.std - before_stats.std))
                norm_shifts.append(abs(after_stats.norm - before_stats.norm))

                if before_stats.activations is not None and after_stats.activations is not None:
                    cos_sim = _cosine_similarity(before_stats.activations, after_stats.activations)
                    cosine_sims.append(cos_sim)
                    change = float(np.linalg.norm(after_stats.activations - before_stats.activations))
                else:
                    cos_sim = 1.0
                    cosine_sims.append(cos_sim)
                    change = ms

                per_probe_changes[probe_id] = change

            layer_diff = LayerDiff(
                layer_name=layer_name,
                mean_shift=float(np.mean(mean_shifts)) if mean_shifts else 0.0,
                std_shift=float(np.mean(std_shifts)) if std_shifts else 0.0,
                norm_shift=float(np.mean(norm_shifts)) if norm_shifts else 0.0,
                direction_similarity=float(np.mean(cosine_sims)) if cosine_sims else 1.0,
                per_probe_changes=per_probe_changes,
            )
            result.layer_diffs[layer_name] = layer_diff

            # Track total change per layer
            total_change = sum(per_probe_changes.values())
            layer_changes[layer_name] = total_change

        # Most changed layers (sorted by total change, top 3)
        sorted_layers = sorted(layer_changes.items(), key=lambda x: x[1], reverse=True)
        result.most_changed_layers = [l[0] for l in sorted_layers[:3]]

        # Most changed probes (aggregate across layers)
        probe_total_changes: Dict[str, float] = {}
        for layer_diff in result.layer_diffs.values():
            for probe_id, change in layer_diff.per_probe_changes.items():
                probe_total_changes[probe_id] = probe_total_changes.get(probe_id, 0) + change

        sorted_probes = sorted(probe_total_changes.items(), key=lambda x: x[1], reverse=True)
        result.most_changed_probes = [p[0] for p in sorted_probes[:5]]

        # Overall change magnitude
        all_changes = list(probe_total_changes.values())
        result.overall_change_magnitude = float(np.mean(all_changes)) if all_changes else 0.0

        # Safety disproportionate check
        safety_probes = [pid for pid in common_probes if "safety" in pid]
        non_safety_probes = [pid for pid in common_probes if "safety" not in pid]

        if safety_probes and non_safety_probes:
            safety_change = np.mean([probe_total_changes.get(p, 0) for p in safety_probes])
            non_safety_change = np.mean([probe_total_changes.get(p, 0) for p in non_safety_probes])

            if non_safety_change > 1e-10:
                result.safety_change_ratio = float(safety_change / non_safety_change)
                result.safety_disproportionate = (
                    result.safety_change_ratio > self.safety_disproportionate_factor
                )
            else:
                result.safety_change_ratio = float(safety_change) if safety_change > 0 else 0.0
                result.safety_disproportionate = safety_change > 1e-10

        return result
