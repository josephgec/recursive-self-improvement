"""Per-layer activation change analysis."""

from typing import Dict, List, Optional, Any
import numpy as np

from src.probing.diff import ActivationDiffResult, LayerDiff


class ActivationAnalysis:
    """Analyze activation changes per layer."""

    def __init__(self):
        self._history: List[ActivationDiffResult] = []

    def add_diff(self, diff: ActivationDiffResult) -> None:
        """Add a diff result to analysis history."""
        self._history.append(diff)

    def analyze_layer(self, layer_name: str) -> Dict[str, Any]:
        """Analyze changes for a specific layer across history."""
        mean_shifts = []
        direction_sims = []
        probe_changes_all = []

        for diff in self._history:
            ld = diff.layer_diffs.get(layer_name)
            if ld:
                mean_shifts.append(ld.mean_shift)
                direction_sims.append(ld.direction_similarity)
                probe_changes_all.extend(ld.per_probe_changes.values())

        if not mean_shifts:
            return {"layer_name": layer_name, "data_points": 0}

        return {
            "layer_name": layer_name,
            "data_points": len(mean_shifts),
            "avg_mean_shift": float(np.mean(mean_shifts)),
            "max_mean_shift": float(np.max(mean_shifts)),
            "avg_direction_similarity": float(np.mean(direction_sims)),
            "min_direction_similarity": float(np.min(direction_sims)),
            "avg_probe_change": float(np.mean(probe_changes_all)) if probe_changes_all else 0.0,
            "max_probe_change": float(np.max(probe_changes_all)) if probe_changes_all else 0.0,
        }

    def analyze_all_layers(self) -> List[Dict[str, Any]]:
        """Analyze all layers that appear in history."""
        all_layers = set()
        for diff in self._history:
            all_layers.update(diff.layer_diffs.keys())

        results = []
        for layer_name in sorted(all_layers):
            results.append(self.analyze_layer(layer_name))
        return results

    def get_most_changed_layers(self, top_n: int = 3) -> List[str]:
        """Get the layers with the most aggregate change."""
        layer_totals: Dict[str, float] = {}
        for diff in self._history:
            for layer_name, ld in diff.layer_diffs.items():
                total = sum(ld.per_probe_changes.values())
                layer_totals[layer_name] = layer_totals.get(layer_name, 0) + total

        sorted_layers = sorted(layer_totals.items(), key=lambda x: x[1], reverse=True)
        return [l[0] for l in sorted_layers[:top_n]]

    def get_safety_layer_analysis(self) -> Dict[str, Any]:
        """Analyze which layers show most change for safety probes."""
        layer_safety_changes: Dict[str, List[float]] = {}

        for diff in self._history:
            for layer_name, ld in diff.layer_diffs.items():
                for probe_id, change in ld.per_probe_changes.items():
                    if "safety" in probe_id:
                        if layer_name not in layer_safety_changes:
                            layer_safety_changes[layer_name] = []
                        layer_safety_changes[layer_name].append(change)

        result = {}
        for layer_name, changes in layer_safety_changes.items():
            result[layer_name] = {
                "mean_safety_change": float(np.mean(changes)),
                "max_safety_change": float(np.max(changes)),
                "num_observations": len(changes),
            }
        return result
