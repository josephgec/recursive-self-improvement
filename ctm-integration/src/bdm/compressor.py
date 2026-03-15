"""Compression-based complexity baselines (Shannon entropy, gzip, lzma)."""

from __future__ import annotations

import gzip
import lzma
import math
from collections import Counter
from typing import Dict, Union


class CompressionBaseline:
    """Compression-based complexity measurements for comparison with BDM."""

    def shannon_entropy(self, data: str) -> float:
        """Compute Shannon entropy in bits per symbol.

        H(X) = -sum(p(x) * log2(p(x)))

        Args:
            data: Input string.

        Returns:
            Shannon entropy in bits per symbol.
        """
        if not data:
            return 0.0

        counts = Counter(data)
        total = len(data)
        entropy = 0.0

        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def gzip_complexity(self, data: str) -> float:
        """Estimate complexity via gzip compression ratio.

        Returns the compressed size as a fraction of the original size.

        Args:
            data: Input string.

        Returns:
            Compressed size in bytes (normalized by input length).
        """
        if not data:
            return 0.0

        data_bytes = data.encode("utf-8")
        compressed = gzip.compress(data_bytes, compresslevel=9)

        # Return compressed bits per input character
        return (len(compressed) * 8) / len(data)

    def lzma_complexity(self, data: str) -> float:
        """Estimate complexity via LZMA compression ratio.

        LZMA typically achieves better compression than gzip.

        Args:
            data: Input string.

        Returns:
            Compressed size in bytes (normalized by input length).
        """
        if not data:
            return 0.0

        data_bytes = data.encode("utf-8")
        compressed = lzma.compress(data_bytes)

        return (len(compressed) * 8) / len(data)

    def compare_all(self, data: str) -> Dict[str, float]:
        """Compute all compression-based complexity measures.

        Args:
            data: Input string.

        Returns:
            Dictionary with all complexity measures.
        """
        return {
            "shannon_entropy": self.shannon_entropy(data),
            "gzip_complexity": self.gzip_complexity(data),
            "lzma_complexity": self.lzma_complexity(data),
            "length": float(len(data)),
        }
