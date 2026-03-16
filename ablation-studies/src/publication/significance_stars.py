"""Significance star notation for publication tables."""

from __future__ import annotations


def add_stars(p_value: float) -> str:
    """Return significance stars based on p-value.

    Convention:
        p < 0.001: ***
        p < 0.01:  **
        p < 0.05:  *
        p >= 0.05: (empty string)

    Args:
        p_value: The p-value from a statistical test.

    Returns:
        String with 0-3 asterisks.
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""
