"""Tests for activation extraction."""

import numpy as np
import pytest

from src.probing.extractor import (
    MockModel, MockModifiedModel, ActivationExtractor,
    ActivationSnapshot, LayerStats, HeadStats,
    _compute_sparsity, _compute_entropy,
)
from src.probing.probe_set import ProbeInput


class TestMockModel:
    """Test the mock model produces deterministic activations."""

    def test_deterministic_activations(self, mock_model):
        """Same input should produce same activations."""
        acts1 = mock_model.get_activations("test input")
        acts2 = mock_model.get_activations("test input")
        for key in acts1:
            np.testing.assert_array_equal(acts1[key], acts2[key])

    def test_different_inputs_different_activations(self, mock_model):
        """Different inputs should produce different activations."""
        acts1 = mock_model.get_activations("input A")
        acts2 = mock_model.get_activations("input B")
        different = False
        for key in acts1:
            if not np.allclose(acts1[key], acts2[key]):
                different = True
        assert different

    def test_layer_count(self, mock_model):
        """Should have correct number of layers."""
        acts = mock_model.get_activations("test")
        assert len(acts) == mock_model.num_layers

    def test_activation_dim(self, mock_model):
        """Each layer should have correct activation dimension."""
        acts = mock_model.get_activations("test")
        for key, arr in acts.items():
            assert arr.shape == (mock_model.activation_dim,)

    def test_head_patterns_shape(self, mock_model):
        """Head patterns should have correct shape."""
        patterns = mock_model.get_head_patterns("hello world test")
        assert len(patterns) == mock_model.num_layers
        for key, arr in patterns.items():
            assert arr.shape[0] == mock_model.num_heads
            # Should be (num_heads, seq_len, seq_len)
            assert arr.ndim == 3
            assert arr.shape[1] == arr.shape[2]

    def test_head_patterns_normalized(self, mock_model):
        """Attention patterns should sum to ~1 along last axis."""
        patterns = mock_model.get_head_patterns("hello world")
        for key, arr in patterns.items():
            row_sums = arr.sum(axis=2)
            # Only check rows that have non-zero attention
            for h in range(arr.shape[0]):
                for r in range(arr.shape[1]):
                    if arr[h, r].sum() > 0:
                        np.testing.assert_almost_equal(row_sums[h, r], 1.0, decimal=5)


class TestMockModifiedModel:
    """Test the modified mock model."""

    def test_perturbed_layers_differ(self, mock_model, mock_modified_model):
        """Perturbed layers should differ from original."""
        text = "test input"
        orig = mock_model.get_activations(text)
        modified = mock_modified_model.get_activations(text)

        # Layer 2 is perturbed
        assert not np.allclose(orig["layer_2"], modified["layer_2"])

    def test_non_perturbed_layers_same(self, mock_model, mock_modified_model):
        """Non-perturbed layers should be identical."""
        text = "test input"
        orig = mock_model.get_activations(text)
        modified = mock_modified_model.get_activations(text)

        # Layer 0 should be same
        np.testing.assert_array_equal(orig["layer_0"], modified["layer_0"])

    def test_perturbed_head_patterns(self, mock_model, mock_modified_model):
        """Perturbed head patterns should differ."""
        text = "hello world"
        orig = mock_model.get_head_patterns(text)
        modified = mock_modified_model.get_head_patterns(text)
        assert not np.allclose(orig["layer_2"], modified["layer_2"])


class TestActivationExtractor:
    """Test the activation extractor."""

    def test_extract_returns_snapshot(self, mock_model, sample_probes):
        """Extract should return an ActivationSnapshot."""
        extractor = ActivationExtractor(mock_model)
        snapshot = extractor.extract(sample_probes)
        assert isinstance(snapshot, ActivationSnapshot)

    def test_extract_all_probes(self, mock_model, sample_probes):
        """Should extract for all probes."""
        extractor = ActivationExtractor(mock_model)
        snapshot = extractor.extract(sample_probes)
        assert len(snapshot.get_probe_ids()) == len(sample_probes)

    def test_extract_all_layers(self, mock_model, sample_probes):
        """Should extract all layers."""
        extractor = ActivationExtractor(mock_model)
        snapshot = extractor.extract(sample_probes)
        layers = snapshot.get_all_layer_names()
        assert len(layers) == mock_model.num_layers

    def test_layer_stats_computed(self, sample_snapshot):
        """LayerStats should have computed summary statistics."""
        probe_ids = sample_snapshot.get_probe_ids()
        layers = sample_snapshot.get_all_layer_names()

        for pid in probe_ids:
            for ln in layers:
                stats = sample_snapshot.get_layer_stats(pid, ln)
                assert stats is not None
                assert isinstance(stats.mean, float)
                assert isinstance(stats.std, float)
                assert isinstance(stats.norm, float)
                assert isinstance(stats.sparsity, float)
                assert stats.activations is not None

    def test_head_stats_extracted(self, sample_snapshot):
        """Head stats should be extracted for probes."""
        assert len(sample_snapshot.head_stats) > 0
        for pid, stats in sample_snapshot.head_stats.items():
            assert len(stats) > 0
            for hs in stats:
                assert isinstance(hs, HeadStats)
                assert isinstance(hs.entropy, float)
                assert isinstance(hs.max_attention, float)

    def test_snapshot_metadata(self, sample_snapshot, sample_probes):
        """Snapshot should have metadata."""
        assert sample_snapshot.metadata["num_probes"] == len(sample_probes)


class TestLayerStats:
    """Test LayerStats serialization."""

    def test_to_dict_and_back(self):
        """LayerStats should roundtrip through dict."""
        acts = np.array([1.0, 2.0, 3.0])
        stats = LayerStats(
            layer_name="layer_0",
            mean=2.0,
            std=0.816,
            norm=3.742,
            sparsity=0.0,
            activations=acts,
        )
        d = stats.to_dict()
        restored = LayerStats.from_dict(d)
        assert restored.layer_name == "layer_0"
        assert restored.mean == 2.0
        np.testing.assert_array_almost_equal(restored.activations, acts)


class TestHeadStats:
    """Test HeadStats serialization."""

    def test_to_dict_and_back(self):
        """HeadStats should roundtrip through dict."""
        hs = HeadStats(layer=1, head=3, entropy=1.5, max_attention=0.8, sparsity=0.2)
        d = hs.to_dict()
        restored = HeadStats.from_dict(d)
        assert restored.layer == 1
        assert restored.head == 3
        assert restored.entropy == 1.5


class TestHelperFunctions:
    """Test helper functions."""

    def test_compute_sparsity(self):
        """Test sparsity computation."""
        arr = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        # With threshold 0.01: 3 of 5 are below threshold
        sparsity = _compute_sparsity(arr, threshold=0.01)
        assert sparsity == pytest.approx(0.6)

    def test_compute_entropy_uniform(self):
        """Uniform distribution should have max entropy."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = _compute_entropy(probs)
        assert entropy > 0

    def test_compute_entropy_peaked(self):
        """Peaked distribution should have low entropy."""
        probs = np.array([0.97, 0.01, 0.01, 0.01])
        entropy = _compute_entropy(probs)
        uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
        uniform_entropy = _compute_entropy(uniform_probs)
        assert entropy < uniform_entropy

    def test_compute_entropy_empty(self):
        """Empty array should return 0."""
        assert _compute_entropy(np.array([])) == 0.0


class TestActivationSnapshot:
    """Test ActivationSnapshot serialization."""

    def test_to_dict_and_back(self, sample_snapshot):
        """Snapshot should roundtrip through dict."""
        d = sample_snapshot.to_dict()
        restored = ActivationSnapshot.from_dict(d)
        assert set(restored.get_probe_ids()) == set(sample_snapshot.get_probe_ids())
        assert set(restored.get_all_layer_names()) == set(sample_snapshot.get_all_layer_names())

    def test_get_layer_stats_missing(self, sample_snapshot):
        """Should return None for missing probe/layer."""
        assert sample_snapshot.get_layer_stats("nonexistent", "layer_0") is None
