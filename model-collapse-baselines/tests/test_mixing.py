"""Tests for src.data.mixing.mix_data."""

from __future__ import annotations

import pytest
from datasets import Dataset

from src.data.mixing import mix_data


# ------------------------------------------------------------------
# Fixtures local to this module
# ------------------------------------------------------------------


@pytest.fixture()
def real_ds():
    return Dataset.from_dict(
        {"text": [f"real_{i}" for i in range(50)]}
    )


@pytest.fixture()
def synth_ds():
    return Dataset.from_dict(
        {"text": [f"synth_{i}" for i in range(50)]}
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestMixDataCounts:
    """Verify that the output size and real/synthetic counts are correct."""

    def test_half_and_half(self, real_ds, synth_ds):
        mixed = mix_data(real_ds, synth_ds, alpha=0.5, total_size=20, seed=0)
        assert len(mixed) == 20
        sources = mixed["source"]
        assert sources.count("real") == 10
        assert sources.count("synthetic") == 10

    def test_mostly_real(self, real_ds, synth_ds):
        mixed = mix_data(real_ds, synth_ds, alpha=0.8, total_size=10, seed=0)
        assert len(mixed) == 10
        assert mixed["source"].count("real") == 8
        assert mixed["source"].count("synthetic") == 2

    def test_rounding(self, real_ds, synth_ds):
        """Round(alpha * total) should give the real count."""
        mixed = mix_data(real_ds, synth_ds, alpha=0.3, total_size=10, seed=0)
        assert len(mixed) == 10
        assert mixed["source"].count("real") == 3
        assert mixed["source"].count("synthetic") == 7


class TestAlphaEdgeCases:
    """Edge cases: alpha=0 (pure synthetic) and alpha=1 (pure real)."""

    def test_alpha_zero(self, real_ds, synth_ds):
        mixed = mix_data(real_ds, synth_ds, alpha=0.0, total_size=15, seed=0)
        assert len(mixed) == 15
        assert all(s == "synthetic" for s in mixed["source"])

    def test_alpha_one(self, real_ds, synth_ds):
        mixed = mix_data(real_ds, synth_ds, alpha=1.0, total_size=15, seed=0)
        assert len(mixed) == 15
        assert all(s == "real" for s in mixed["source"])

    def test_alpha_out_of_range(self, real_ds, synth_ds):
        with pytest.raises(ValueError, match="alpha must be in"):
            mix_data(real_ds, synth_ds, alpha=1.5, total_size=10)

    def test_alpha_negative(self, real_ds, synth_ds):
        with pytest.raises(ValueError, match="alpha must be in"):
            mix_data(real_ds, synth_ds, alpha=-0.1, total_size=10)


class TestDeterministicShuffle:
    """Same seed should produce the same output order."""

    def test_same_seed_same_order(self, real_ds, synth_ds):
        m1 = mix_data(real_ds, synth_ds, alpha=0.5, total_size=20, seed=123)
        m2 = mix_data(real_ds, synth_ds, alpha=0.5, total_size=20, seed=123)
        assert m1["text"] == m2["text"]
        assert m1["source"] == m2["source"]

    def test_different_seed_different_order(self, real_ds, synth_ds):
        m1 = mix_data(real_ds, synth_ds, alpha=0.5, total_size=20, seed=1)
        m2 = mix_data(real_ds, synth_ds, alpha=0.5, total_size=20, seed=2)
        # Very unlikely to be identical with different seeds.
        assert m1["text"] != m2["text"]


class TestSourceColumn:
    """The output should always contain a 'source' column."""

    def test_source_column_present(self, real_ds, synth_ds):
        mixed = mix_data(real_ds, synth_ds, alpha=0.5, total_size=10, seed=0)
        assert "source" in mixed.column_names

    def test_source_values(self, real_ds, synth_ds):
        mixed = mix_data(real_ds, synth_ds, alpha=0.5, total_size=10, seed=0)
        valid_sources = {"real", "synthetic"}
        assert all(s in valid_sources for s in mixed["source"])

    def test_total_size_zero(self, real_ds, synth_ds):
        mixed = mix_data(real_ds, synth_ds, alpha=0.5, total_size=0, seed=0)
        assert len(mixed) == 0
        assert "source" in mixed.column_names


class TestSamplingWithReplacement:
    """When source is too small, sampling with replacement should work."""

    def test_oversample_real(self, synth_ds):
        tiny_real = Dataset.from_dict({"text": ["r1", "r2"]})
        with pytest.warns(UserWarning, match="sampling with replacement"):
            mixed = mix_data(
                tiny_real, synth_ds, alpha=1.0, total_size=10, seed=0
            )
        assert len(mixed) == 10
        assert all(s == "real" for s in mixed["source"])

    def test_oversample_synthetic(self, real_ds):
        tiny_synth = Dataset.from_dict({"text": ["s1"]})
        with pytest.warns(UserWarning, match="sampling with replacement"):
            mixed = mix_data(
                real_ds, tiny_synth, alpha=0.0, total_size=5, seed=0
            )
        assert len(mixed) == 5
        assert all(s == "synthetic" for s in mixed["source"])
