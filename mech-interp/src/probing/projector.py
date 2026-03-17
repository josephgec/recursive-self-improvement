"""Dimensionality reduction for activation visualization."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

from src.probing.extractor import ActivationSnapshot


@dataclass
class ProjectionResult:
    """Result of dimensionality reduction."""
    coordinates: np.ndarray  # (n_samples, 2 or 3)
    labels: List[str]
    explained_variance: Optional[List[float]] = None

    def to_dict(self) -> dict:
        d = {
            "coordinates": self.coordinates.tolist(),
            "labels": self.labels,
        }
        if self.explained_variance is not None:
            d["explained_variance"] = self.explained_variance
        return d


class DimensionalityReducer:
    """PCA-based dimensionality reduction for activations."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def reduce(self, snapshot: ActivationSnapshot,
               layer_name: Optional[str] = None) -> ProjectionResult:
        """Reduce activations to 2D/3D for visualization.

        If layer_name is specified, reduce that layer only.
        Otherwise, concatenate all layers.
        """
        vectors = []
        labels = []

        for probe_id in sorted(snapshot.get_probe_ids()):
            probe_acts = snapshot.probe_activations[probe_id]
            if layer_name:
                if layer_name in probe_acts and probe_acts[layer_name].activations is not None:
                    vectors.append(probe_acts[layer_name].activations)
                    labels.append(probe_id)
            else:
                # Concatenate all layers
                layer_vecs = []
                for ln in sorted(probe_acts.keys()):
                    if probe_acts[ln].activations is not None:
                        layer_vecs.append(probe_acts[ln].activations)
                if layer_vecs:
                    vectors.append(np.concatenate(layer_vecs))
                    labels.append(probe_id)

        if not vectors:
            return ProjectionResult(
                coordinates=np.zeros((0, self.n_components)),
                labels=[],
            )

        X = np.array(vectors)
        return self._pca(X, labels)

    def _pca(self, X: np.ndarray, labels: List[str]) -> ProjectionResult:
        """Simple PCA implementation."""
        n_samples, n_features = X.shape
        n_comp = min(self.n_components, n_samples, n_features)

        # Center
        mean = X.mean(axis=0)
        X_centered = X - mean

        # Covariance
        if n_samples > 1:
            cov = np.dot(X_centered.T, X_centered) / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1][:n_comp]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Project
            projected = np.dot(X_centered, eigenvectors)

            # Explained variance
            total_var = eigenvalues.sum() if eigenvalues.sum() > 0 else 1.0
            explained = (eigenvalues / total_var).tolist()
        else:
            projected = np.zeros((n_samples, n_comp))
            explained = [0.0] * n_comp

        # Pad if needed
        if projected.shape[1] < self.n_components:
            pad = np.zeros((projected.shape[0], self.n_components - projected.shape[1]))
            projected = np.hstack([projected, pad])
            explained.extend([0.0] * (self.n_components - len(explained)))

        return ProjectionResult(
            coordinates=projected,
            labels=labels,
            explained_variance=explained[:self.n_components],
        )

    def reduce_multi(self, snapshots: List[ActivationSnapshot],
                     snapshot_labels: List[str],
                     layer_name: Optional[str] = None) -> ProjectionResult:
        """Reduce activations from multiple snapshots into shared space."""
        vectors = []
        labels = []

        for snap, snap_label in zip(snapshots, snapshot_labels):
            for probe_id in sorted(snap.get_probe_ids()):
                probe_acts = snap.probe_activations[probe_id]
                if layer_name:
                    if layer_name in probe_acts and probe_acts[layer_name].activations is not None:
                        vectors.append(probe_acts[layer_name].activations)
                        labels.append(f"{snap_label}/{probe_id}")
                else:
                    layer_vecs = []
                    for ln in sorted(probe_acts.keys()):
                        if probe_acts[ln].activations is not None:
                            layer_vecs.append(probe_acts[ln].activations)
                    if layer_vecs:
                        vectors.append(np.concatenate(layer_vecs))
                        labels.append(f"{snap_label}/{probe_id}")

        if not vectors:
            return ProjectionResult(
                coordinates=np.zeros((0, self.n_components)),
                labels=[],
            )

        X = np.array(vectors)
        return self._pca(X, labels)
