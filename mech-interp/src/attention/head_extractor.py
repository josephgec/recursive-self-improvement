"""Extract per-head attention statistics from a model."""

from typing import List, Dict, Any
import numpy as np

from src.probing.probe_set import ProbeInput
from src.probing.extractor import HeadStats


def _compute_entropy(probs: np.ndarray) -> float:
    """Compute entropy of a probability distribution."""
    p = probs.flatten()
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


class HeadExtractor:
    """Extract per-head attention statistics from a model."""

    def __init__(self, model: Any):
        self.model = model

    def extract_head_patterns(self, inputs: List[ProbeInput]) -> Dict[str, List[HeadStats]]:
        """Extract per-head stats for each probe input.

        Returns: probe_id -> list of HeadStats
        """
        results = {}
        for probe in inputs:
            head_patterns = self.model.get_head_patterns(probe.text)
            stats_list = []
            for layer_name, patterns in sorted(head_patterns.items()):
                layer_idx = int(layer_name.split("_")[1])
                for head_idx in range(patterns.shape[0]):
                    head_pattern = patterns[head_idx]
                    # Average over rows for entropy
                    avg_row = head_pattern.mean(axis=0)
                    avg_row_norm = avg_row / (avg_row.sum() + 1e-12)
                    entropy = _compute_entropy(avg_row_norm)
                    max_attn = float(head_pattern.max())
                    sparsity = float(np.mean(head_pattern < 0.01))

                    stats_list.append(HeadStats(
                        layer=layer_idx,
                        head=head_idx,
                        entropy=entropy,
                        max_attention=max_attn,
                        sparsity=sparsity,
                    ))
            results[probe.probe_id] = stats_list
        return results

    def extract_aggregate_stats(self, inputs: List[ProbeInput]) -> List[HeadStats]:
        """Extract aggregated head stats across all inputs."""
        per_probe = self.extract_head_patterns(inputs)
        if not per_probe:
            return []

        # Aggregate by (layer, head)
        aggregated: Dict[tuple, List[HeadStats]] = {}
        for stats_list in per_probe.values():
            for hs in stats_list:
                key = (hs.layer, hs.head)
                if key not in aggregated:
                    aggregated[key] = []
                aggregated[key].append(hs)

        result = []
        for (layer, head), stats_list in sorted(aggregated.items()):
            avg_entropy = np.mean([s.entropy for s in stats_list])
            avg_max_attn = np.mean([s.max_attention for s in stats_list])
            avg_sparsity = np.mean([s.sparsity for s in stats_list])
            result.append(HeadStats(
                layer=layer,
                head=head,
                entropy=float(avg_entropy),
                max_attention=float(avg_max_attn),
                sparsity=float(avg_sparsity),
            ))
        return result
