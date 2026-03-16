"""Verification cache: memoizes verification results."""
from __future__ import annotations

from typing import Any, Dict, Optional


class VerificationCache:
    """Cache for verification results keyed by candidate_id."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def get(self, candidate_id: str) -> Optional[Any]:
        """Get cached result for a candidate."""
        return self._cache.get(candidate_id)

    def put(self, candidate_id: str, result: Any) -> None:
        """Cache a verification result."""
        self._cache[candidate_id] = result

    def has(self, candidate_id: str) -> bool:
        """Check if a candidate result is cached."""
        return candidate_id in self._cache

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
