"""Answer checking for financial math tasks."""

from __future__ import annotations

import re
from typing import Optional


class FinancialAnswerChecker:
    """Check if a response contains the correct financial answer.

    Handles various formats: $1,234.56, 1234.56, 12.34%, etc.
    Uses relative tolerance for numeric comparison.
    """

    def __init__(self, tolerance: float = 0.02):
        """
        Args:
            tolerance: Relative tolerance for numeric comparison (default 2%).
        """
        self.tolerance = tolerance

    def check(self, response: str, expected: str) -> bool:
        """Check if the response contains the expected answer.

        Args:
            response: The full LLM response text
            expected: The expected answer (numeric string)

        Returns:
            True if a matching numeric value is found in the response.
        """
        expected_num = self._extract_numeric(expected)
        if expected_num is None:
            # Fall back to string containment
            return expected.strip().lower() in response.lower()

        # Extract all numbers from response and check if any match
        response_nums = self._extract_all_numerics(response)
        for num in response_nums:
            if self._compare_numeric(num, expected_num):
                return True

        return False

    def _extract_numeric(self, text: str) -> Optional[float]:
        """Extract a numeric value from text.

        Handles formats like:
        - $1,234.56
        - 1234.56
        - 12.34%
        - -5.00
        """
        if not text:
            return None

        # Remove common formatting
        cleaned = text.strip()
        cleaned = cleaned.replace("$", "").replace(",", "").replace("%", "")
        cleaned = cleaned.strip()

        try:
            return float(cleaned)
        except ValueError:
            # Try regex for embedded numbers
            match = re.search(r"-?\d+\.?\d*", cleaned)
            if match:
                return float(match.group())
            return None

    def _extract_all_numerics(self, text: str) -> list:
        """Extract all numeric values from text."""
        # Remove dollar signs and commas for extraction
        cleaned = text.replace("$", "").replace(",", "")
        # Find all numeric patterns
        matches = re.findall(r"-?\d+\.?\d*", cleaned)
        results = []
        for m in matches:
            try:
                results.append(float(m))
            except ValueError:
                pass
        return results

    def _compare_numeric(
        self, actual: float, expected: float
    ) -> bool:
        """Compare two numbers with relative tolerance.

        Args:
            actual: The number from the response
            expected: The expected number

        Returns:
            True if within tolerance.
        """
        if expected == 0:
            return abs(actual) < self.tolerance

        relative_error = abs(actual - expected) / abs(expected)
        return relative_error <= self.tolerance
