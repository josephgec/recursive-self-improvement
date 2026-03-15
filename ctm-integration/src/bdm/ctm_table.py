"""CTM (Coding Theorem Method) table: maps binary strings to algorithmic probability."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from src.utils.turing_machines import enumerate_tms, count_tms


# Pre-computed fallback table for common short binary strings.
# These are approximate K values (algorithmic complexity) for strings up to length 12.
# Derived from published CTM tables (Soler-Toscano et al., 2014).
_PRECOMPUTED_FALLBACK: Dict[str, float] = {
    # Length 1
    "0": 1.0,
    "1": 1.0,
    # Length 2
    "00": 2.30,
    "01": 3.70,
    "10": 3.70,
    "11": 2.30,
    # Length 3
    "000": 3.23,
    "001": 5.41,
    "010": 6.13,
    "011": 5.41,
    "100": 5.41,
    "101": 6.13,
    "110": 5.41,
    "111": 3.23,
    # Length 4
    "0000": 3.91,
    "0001": 6.79,
    "0010": 7.71,
    "0011": 6.29,
    "0100": 7.94,
    "0101": 7.83,
    "0110": 7.27,
    "0111": 6.79,
    "1000": 6.79,
    "1001": 7.27,
    "1010": 7.83,
    "1011": 7.94,
    "1100": 6.29,
    "1101": 7.71,
    "1110": 6.79,
    "1111": 3.91,
    # Length 5-8: representative samples
    "00000": 4.50,
    "00001": 7.50,
    "01010": 7.20,
    "10101": 7.20,
    "11111": 4.50,
    "00000000": 5.80,
    "01010101": 7.00,
    "10101010": 7.00,
    "11111111": 5.80,
    "00110011": 7.50,
    "11001100": 7.50,
    "01100110": 8.00,
    "10011001": 8.00,
    # Length 12: representative samples
    "000000000000": 6.50,
    "010101010101": 7.50,
    "101010101010": 7.50,
    "111111111111": 6.50,
    "001100110011": 8.00,
    "110011001100": 8.00,
}


class CTMTable:
    """Coding Theorem Method table for looking up algorithmic probability.

    Maps short binary strings to their algorithmic complexity K(s),
    estimated via the Coding Theorem: K(s) ~ -log2(m(s)) where m(s) is
    the algorithmic probability (fraction of TMs that produce s).
    """

    def __init__(self) -> None:
        self._table: Dict[str, float] = {}
        self._output_counts: Dict[str, int] = {}
        self._total_halting: int = 0
        self._built = False

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def size(self) -> int:
        return len(self._table)

    def build(
        self,
        max_states: int = 2,
        max_symbols: int = 2,
        max_steps: int = 50,
        block_size: int = 12,
    ) -> None:
        """Build the CTM table by enumerating small Turing machines.

        Args:
            max_states: Maximum number of TM states to enumerate.
            max_symbols: Maximum number of tape symbols.
            max_steps: Maximum steps before declaring non-halting.
            block_size: Maximum string length to store in table.
        """
        output_counts: Counter = Counter()
        total_halting = 0

        for tm in enumerate_tms(max_states, max_symbols):
            output = tm.run(max_steps=max_steps)
            if output is not None and len(output) <= block_size:
                output_counts[output] += 1
                total_halting += 1

        self._output_counts = dict(output_counts)
        self._total_halting = total_halting

        # Compute K(s) = -log2(m(s)) where m(s) = count(s) / total_halting
        for s, count in output_counts.items():
            prob = count / total_halting
            self._table[s] = -math.log2(prob)

        # Merge with precomputed fallback for missing strings
        for s, k in _PRECOMPUTED_FALLBACK.items():
            if s not in self._table:
                self._table[s] = k

        self._built = True

    def lookup(self, binary_string: str) -> float:
        """Look up the algorithmic complexity of a binary string.

        Args:
            binary_string: A string of 0s and 1s.

        Returns:
            Estimated Kolmogorov complexity K(s).
        """
        if binary_string in self._table:
            return self._table[binary_string]

        # Check precomputed fallback
        if binary_string in _PRECOMPUTED_FALLBACK:
            return _PRECOMPUTED_FALLBACK[binary_string]

        # For strings not in table, estimate based on length
        # Upper bound: length of string (description "print this string")
        return float(len(binary_string))

    def algorithmic_probability(self, binary_string: str) -> float:
        """Get the algorithmic probability m(s) = 2^(-K(s)).

        Args:
            binary_string: A string of 0s and 1s.

        Returns:
            Algorithmic probability (higher = more likely to be produced by random TM).
        """
        k = self.lookup(binary_string)
        return 2.0 ** (-k)

    def save(self, path: str) -> None:
        """Save the CTM table to a JSON file.

        Args:
            path: File path to save to.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "table": self._table,
            "output_counts": self._output_counts,
            "total_halting": self._total_halting,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load the CTM table from a JSON file.

        Args:
            path: File path to load from.
        """
        with open(path, "r") as f:
            data = json.load(f)
        self._table = data["table"]
        self._output_counts = data.get("output_counts", {})
        self._total_halting = data.get("total_halting", 0)
        self._built = True

    def load_or_build(
        self,
        path: str,
        max_states: int = 2,
        max_symbols: int = 2,
        max_steps: int = 50,
        block_size: int = 12,
    ) -> None:
        """Load from file if it exists, otherwise build and save.

        Args:
            path: File path for the table.
            max_states: Maximum number of TM states.
            max_symbols: Maximum number of tape symbols.
            max_steps: Maximum steps before non-halting.
            block_size: Maximum string length.
        """
        if os.path.exists(path):
            self.load(path)
        else:
            self.build(max_states, max_symbols, max_steps, block_size)
            self.save(path)

    def get_table(self) -> Dict[str, float]:
        """Return a copy of the internal table."""
        return dict(self._table)

    @classmethod
    def with_fallback_only(cls) -> "CTMTable":
        """Create a CTMTable using only the precomputed fallback values.

        This is fast and suitable for testing.
        """
        instance = cls()
        instance._table = dict(_PRECOMPUTED_FALLBACK)
        instance._built = True
        return instance
