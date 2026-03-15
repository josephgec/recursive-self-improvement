"""Tests for CTM table building, loading, and lookup."""

import json
import os
import tempfile

import pytest

from src.bdm.ctm_table import CTMTable


class TestCTMTableBuild:
    """Tests for building CTM tables."""

    def test_build_small_table(self):
        """Build a table from 1-state 2-symbol TMs."""
        table = CTMTable()
        table.build(max_states=1, max_symbols=2, max_steps=20, block_size=8)
        assert table.is_built
        assert table.size > 0

    def test_build_includes_common_strings(self):
        """Built table should include common short strings."""
        table = CTMTable()
        table.build(max_states=1, max_symbols=2, max_steps=20, block_size=8)
        # "0" and "1" should be in any TM enumeration
        k0 = table.lookup("0")
        k1 = table.lookup("1")
        assert k0 > 0
        assert k1 > 0

    def test_fallback_only_table(self):
        """Fallback-only table should have precomputed values."""
        table = CTMTable.with_fallback_only()
        assert table.is_built
        assert table.size > 0
        assert table.lookup("0") == 1.0
        assert table.lookup("1") == 1.0

    def test_lookup_unknown_string(self):
        """Unknown strings should get length-based estimate."""
        table = CTMTable.with_fallback_only()
        # A string not in the fallback table
        unknown = "0110100110"
        k = table.lookup(unknown)
        # Should be >= length (upper bound)
        assert k == len(unknown)

    def test_algorithmic_probability(self):
        """Algorithmic probability should be 2^(-K)."""
        table = CTMTable.with_fallback_only()
        k = table.lookup("0")
        prob = table.algorithmic_probability("0")
        assert abs(prob - 2 ** (-k)) < 1e-10


class TestCTMTablePersistence:
    """Tests for saving and loading CTM tables."""

    def test_save_and_load(self, tmp_path):
        """Save and reload should preserve all data."""
        path = str(tmp_path / "test_ctm.json")

        table = CTMTable()
        table.build(max_states=1, max_symbols=2, max_steps=20, block_size=8)

        original_size = table.size
        original_k0 = table.lookup("0")

        table.save(path)
        assert os.path.exists(path)

        loaded = CTMTable()
        loaded.load(path)

        assert loaded.is_built
        assert loaded.size == original_size
        assert loaded.lookup("0") == original_k0

    def test_load_or_build_creates_file(self, tmp_path):
        """load_or_build should create file if it doesn't exist."""
        path = str(tmp_path / "new_ctm.json")
        assert not os.path.exists(path)

        table = CTMTable()
        table.load_or_build(path, max_states=1, max_symbols=2, max_steps=20, block_size=8)

        assert table.is_built
        assert os.path.exists(path)

    def test_load_or_build_loads_existing(self, tmp_path):
        """load_or_build should load from file if it exists."""
        path = str(tmp_path / "existing_ctm.json")

        # Build and save
        table1 = CTMTable()
        table1.build(max_states=1, max_symbols=2, max_steps=20, block_size=8)
        table1.save(path)
        size1 = table1.size

        # Load
        table2 = CTMTable()
        table2.load_or_build(path, max_states=1, max_symbols=2, max_steps=20, block_size=8)
        assert table2.size == size1


class TestCTMTableOrdering:
    """Tests for complexity ordering properties."""

    def test_simple_less_complex_than_random_looking(self):
        """Simpler strings should have lower complexity."""
        table = CTMTable.with_fallback_only()

        # "0000" (constant) should be simpler than "0110" (more random)
        k_constant = table.lookup("0000")
        k_varied = table.lookup("0110")
        assert k_constant < k_varied

    def test_constant_simpler_than_periodic(self):
        """Constant strings should be simpler than periodic ones."""
        table = CTMTable.with_fallback_only()

        k_const = table.lookup("0000")
        k_periodic = table.lookup("0101")
        assert k_const < k_periodic

    def test_symmetric_strings_similar_complexity(self):
        """Strings and their complements should have similar complexity."""
        table = CTMTable.with_fallback_only()

        k_00 = table.lookup("00")
        k_11 = table.lookup("11")
        assert k_00 == k_11

    def test_get_table_returns_copy(self):
        """get_table should return a copy, not the internal dict."""
        table = CTMTable.with_fallback_only()
        t = table.get_table()
        original_size = table.size
        t["new_key"] = 999
        assert table.size == original_size
