"""Shared utilities."""

from src.utils.code_similarity import code_similarity, normalize_code
from src.utils.grid_diff import compute_grid_diff, diff_summary
from src.utils.logging import get_logger, setup_logging

__all__ = [
    "code_similarity", "normalize_code",
    "compute_grid_diff", "diff_summary",
    "get_logger", "setup_logging",
]
