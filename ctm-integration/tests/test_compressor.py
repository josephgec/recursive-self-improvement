"""Tests for compression baselines."""

import pytest

from src.bdm.compressor import CompressionBaseline


class TestCompressionBaseline:
    """Tests for compression-based complexity measures."""

    def test_shannon_entropy_constant(self):
        """Constant string should have zero entropy."""
        cb = CompressionBaseline()
        assert cb.shannon_entropy("0000000") == 0.0

    def test_shannon_entropy_maximum(self):
        """Equiprobable binary string should have entropy near 1.0."""
        cb = CompressionBaseline()
        entropy = cb.shannon_entropy("01010101")
        assert abs(entropy - 1.0) < 0.01

    def test_shannon_entropy_empty(self):
        """Empty string should have zero entropy."""
        cb = CompressionBaseline()
        assert cb.shannon_entropy("") == 0.0

    def test_gzip_complexity(self):
        """gzip complexity should be positive."""
        cb = CompressionBaseline()
        result = cb.gzip_complexity("01010101")
        assert result > 0

    def test_gzip_empty(self):
        """Empty string should have zero gzip complexity."""
        cb = CompressionBaseline()
        assert cb.gzip_complexity("") == 0.0

    def test_lzma_complexity(self):
        """lzma complexity should be positive."""
        cb = CompressionBaseline()
        result = cb.lzma_complexity("01010101")
        assert result > 0

    def test_lzma_empty(self):
        """Empty string should have zero lzma complexity."""
        cb = CompressionBaseline()
        assert cb.lzma_complexity("") == 0.0

    def test_compare_all(self):
        """compare_all should return all metrics."""
        cb = CompressionBaseline()
        result = cb.compare_all("01010101")
        assert "shannon_entropy" in result
        assert "gzip_complexity" in result
        assert "lzma_complexity" in result
        assert "length" in result
        assert result["length"] == 8.0

    def test_repetitive_lower_complexity(self):
        """Repetitive data should compress better than random."""
        cb = CompressionBaseline()
        repetitive = "0" * 100
        random_like = "0110100110010110" * 6 + "0110"

        gzip_rep = cb.gzip_complexity(repetitive)
        gzip_rand = cb.gzip_complexity(random_like)
        assert gzip_rep < gzip_rand
