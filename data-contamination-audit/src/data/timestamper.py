"""Temporal binning for documents.

Maps a :class:`Document` timestamp to a human-readable bin label such as
``"2023"``, ``"2023-H1"``, or ``"2023-Q3"``.
"""

from __future__ import annotations

import logging

from src.data.common_crawl import Document

logger = logging.getLogger(__name__)

# Supported bin sizes.
_VALID_BIN_SIZES = {"year", "half-year", "quarter"}


def assign_time_bin(doc: Document, bin_size: str = "year") -> str:
    """Map a document's timestamp to a temporal bin label.

    Parameters
    ----------
    doc:
        Document whose ``timestamp`` field will be used.
    bin_size:
        One of ``"year"``, ``"half-year"``, or ``"quarter"``.

    Returns
    -------
    str
        Bin label, e.g. ``"2023"``, ``"2023-H2"``, ``"2023-Q4"``.

    Raises
    ------
    ValueError
        If *bin_size* is not one of the supported values.
    """
    if bin_size not in _VALID_BIN_SIZES:
        raise ValueError(
            f"Invalid bin_size {bin_size!r}. Must be one of {sorted(_VALID_BIN_SIZES)}"
        )

    ts = doc.timestamp
    year = ts.year

    if bin_size == "year":
        return str(year)

    month = ts.month

    if bin_size == "half-year":
        half = "H1" if month <= 6 else "H2"
        return f"{year}-{half}"

    # bin_size == "quarter"
    if month <= 3:
        q = "Q1"
    elif month <= 6:
        q = "Q2"
    elif month <= 9:
        q = "Q3"
    else:
        q = "Q4"
    return f"{year}-{q}"
