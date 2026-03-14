"""Clean data reserve: filtering, quality checks, and export.

Public API
----------
- :func:`filter_to_reserve` — select documents above a human-authorship
  probability threshold.
- :func:`compute_alpha_t` — proportion of authentic data in the corpus.
- :func:`deduplicate` — remove near-duplicate documents.
- :func:`language_filter` — remove non-English documents.
- :func:`length_filter` — remove too-short / too-long documents.
- :func:`apply_quality_filters` — run all quality filters in sequence.
- :func:`export_reserve` — write reserve to Parquet + summary JSON.
"""

from src.reserve.export import export_reserve
from src.reserve.filter import compute_alpha_t, filter_to_reserve
from src.reserve.quality import (
    apply_quality_filters,
    deduplicate,
    language_filter,
    length_filter,
)

__all__ = [
    "filter_to_reserve",
    "compute_alpha_t",
    "deduplicate",
    "language_filter",
    "length_filter",
    "apply_quality_filters",
    "export_reserve",
]
