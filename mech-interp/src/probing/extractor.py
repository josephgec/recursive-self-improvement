"""Activation extraction using mock models."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol
import numpy as np

from src.probing.probe_set import ProbeInput


class ModelProtocol(Protocol):
    """Protocol for models that can produce activations."""
    num_layers: int
    num_heads: int
    hidden_dim: int

    def get_activations(self, text: str) -> Dict[str, np.ndarray]:
        """Return activations per layer for given input text."""
        ...

    def get_head_patterns(self, text: str) -> Dict[str, np.ndarray]:
        """Return attention head patterns per layer."""
        ...


class MockModel:
    """Mock model producing deterministic activations from input hash.

    Layer L for input hash H: activations[i] = sin(H * (L+1) * i)
    """

    def __init__(self, num_layers: int = 12, num_heads: int = 12,
                 hidden_dim: int = 768, activation_dim: int = 64):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.activation_dim = activation_dim

    def _input_hash(self, text: str) -> int:
        """Deterministic hash for input text."""
        return hash(text) % 10007 + 1

    def get_activations(self, text: str) -> Dict[str, np.ndarray]:
        """Return activations per layer."""
        h = self._input_hash(text)
        result = {}
        for layer in range(self.num_layers):
            indices = np.arange(self.activation_dim, dtype=np.float64)
            activations = np.sin(h * (layer + 1) * indices * 0.01)
            result[f"layer_{layer}"] = activations
        return result

    def get_head_patterns(self, text: str) -> Dict[str, np.ndarray]:
        """Return attention patterns per layer. Shape: (num_heads, seq_len, seq_len)."""
        h = self._input_hash(text)
        seq_len = min(len(text.split()) + 2, 20)
        result = {}
        for layer in range(self.num_layers):
            patterns = np.zeros((self.num_heads, seq_len, seq_len))
            for head in range(self.num_heads):
                seed = h * (layer + 1) * (head + 1)
                rng = np.random.RandomState(seed % (2**31))
                raw = rng.exponential(1.0, (seq_len, seq_len))
                # Apply causal mask
                mask = np.tril(np.ones((seq_len, seq_len)))
                raw = raw * mask
                # Normalize rows
                row_sums = raw.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1, row_sums)
                patterns[head] = raw / row_sums
            result[f"layer_{layer}"] = patterns
        return result


class MockModifiedModel(MockModel):
    """Mock model with perturbation on specific layers to simulate modification."""

    def __init__(self, num_layers: int = 12, num_heads: int = 12,
                 hidden_dim: int = 768, activation_dim: int = 64,
                 perturbed_layers: Optional[List[int]] = None,
                 perturbation_scale: float = 0.5):
        super().__init__(num_layers, num_heads, hidden_dim, activation_dim)
        self.perturbed_layers = perturbed_layers or [num_layers // 2]
        self.perturbation_scale = perturbation_scale

    def get_activations(self, text: str) -> Dict[str, np.ndarray]:
        """Return activations with perturbation on specified layers."""
        result = super().get_activations(text)
        h = self._input_hash(text)
        for layer in self.perturbed_layers:
            key = f"layer_{layer}"
            if key in result:
                rng = np.random.RandomState((h + layer * 997) % (2**31))
                perturbation = rng.randn(self.activation_dim) * self.perturbation_scale
                result[key] = result[key] + perturbation
        return result

    def get_head_patterns(self, text: str) -> Dict[str, np.ndarray]:
        """Return attention patterns with perturbation on specified layers."""
        result = super().get_head_patterns(text)
        h = self._input_hash(text)
        for layer in self.perturbed_layers:
            key = f"layer_{layer}"
            if key in result:
                patterns = result[key]
                for head in range(self.num_heads):
                    rng = np.random.RandomState((h + layer * 997 + head * 31) % (2**31))
                    noise = rng.exponential(self.perturbation_scale, patterns[head].shape)
                    mask = np.tril(np.ones(patterns[head].shape))
                    noise = noise * mask
                    perturbed = patterns[head] + noise
                    row_sums = perturbed.sum(axis=1, keepdims=True)
                    row_sums = np.where(row_sums == 0, 1, row_sums)
                    patterns[head] = perturbed / row_sums
                result[key] = patterns
        return result


@dataclass
class HeadStats:
    """Statistics for a single attention head."""
    layer: int
    head: int
    entropy: float
    max_attention: float
    sparsity: float

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "head": self.head,
            "entropy": self.entropy,
            "max_attention": self.max_attention,
            "sparsity": self.sparsity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HeadStats":
        return cls(**d)


@dataclass
class LayerStats:
    """Summary statistics for activations at a single layer."""
    layer_name: str
    mean: float
    std: float
    norm: float
    sparsity: float
    activations: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        d = {
            "layer_name": self.layer_name,
            "mean": self.mean,
            "std": self.std,
            "norm": self.norm,
            "sparsity": self.sparsity,
        }
        if self.activations is not None:
            d["activations"] = self.activations.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LayerStats":
        acts = None
        if "activations" in d:
            acts = np.array(d["activations"])
        return cls(
            layer_name=d["layer_name"],
            mean=d["mean"],
            std=d["std"],
            norm=d["norm"],
            sparsity=d["sparsity"],
            activations=acts,
        )


@dataclass
class ActivationSnapshot:
    """A complete snapshot of activations for all probes."""
    probe_activations: Dict[str, Dict[str, LayerStats]] = field(default_factory=dict)
    # probe_id -> layer_name -> LayerStats
    head_stats: Dict[str, List[HeadStats]] = field(default_factory=dict)
    # probe_id -> list of HeadStats
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_layer_stats(self, probe_id: str, layer_name: str) -> Optional[LayerStats]:
        if probe_id in self.probe_activations:
            return self.probe_activations[probe_id].get(layer_name)
        return None

    def get_all_layer_names(self) -> List[str]:
        """Return sorted unique layer names across all probes."""
        names = set()
        for probe_acts in self.probe_activations.values():
            names.update(probe_acts.keys())
        return sorted(names)

    def get_probe_ids(self) -> List[str]:
        return list(self.probe_activations.keys())

    def to_dict(self) -> dict:
        return {
            "probe_activations": {
                pid: {ln: ls.to_dict() for ln, ls in layers.items()}
                for pid, layers in self.probe_activations.items()
            },
            "head_stats": {
                pid: [hs.to_dict() for hs in stats]
                for pid, stats in self.head_stats.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActivationSnapshot":
        pa = {}
        for pid, layers in d.get("probe_activations", {}).items():
            pa[pid] = {ln: LayerStats.from_dict(ls) for ln, ls in layers.items()}
        hs = {}
        for pid, stats in d.get("head_stats", {}).items():
            hs[pid] = [HeadStats.from_dict(s) for s in stats]
        return cls(
            probe_activations=pa,
            head_stats=hs,
            metadata=d.get("metadata", {}),
        )


def _compute_sparsity(arr: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of values below threshold."""
    return float(np.mean(np.abs(arr) < threshold))


def _compute_entropy(probs: np.ndarray) -> float:
    """Compute entropy of a probability distribution."""
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p + 1e-12)))


class ActivationExtractor:
    """Extracts activations from a model using probe inputs."""

    def __init__(self, model: Any):
        self.model = model

    def extract(self, probe_inputs: List[ProbeInput]) -> ActivationSnapshot:
        """Extract activations for all probe inputs."""
        snapshot = ActivationSnapshot(
            metadata={"num_probes": len(probe_inputs)}
        )

        for probe in probe_inputs:
            # Get activations
            activations = self.model.get_activations(probe.text)
            layer_stats = {}
            for layer_name, acts in activations.items():
                stats = LayerStats(
                    layer_name=layer_name,
                    mean=float(np.mean(acts)),
                    std=float(np.std(acts)),
                    norm=float(np.linalg.norm(acts)),
                    sparsity=_compute_sparsity(acts),
                    activations=acts.copy(),
                )
                layer_stats[layer_name] = stats
            snapshot.probe_activations[probe.probe_id] = layer_stats

            # Get head patterns
            try:
                head_patterns = self.model.get_head_patterns(probe.text)
                head_stats_list = []
                for layer_name, patterns in head_patterns.items():
                    layer_idx = int(layer_name.split("_")[1])
                    for head_idx in range(patterns.shape[0]):
                        head_pattern = patterns[head_idx]
                        avg_row = head_pattern.mean(axis=0)
                        avg_row = avg_row / (avg_row.sum() + 1e-12)
                        entropy = _compute_entropy(avg_row)
                        max_attn = float(head_pattern.max())
                        sparsity = _compute_sparsity(head_pattern, threshold=0.01)
                        head_stats_list.append(HeadStats(
                            layer=layer_idx,
                            head=head_idx,
                            entropy=entropy,
                            max_attention=max_attn,
                            sparsity=sparsity,
                        ))
                snapshot.head_stats[probe.probe_id] = head_stats_list
            except (AttributeError, TypeError):
                pass

        return snapshot
