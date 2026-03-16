"""ConstraintCache: TTL-based caching for constraint results."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

from src.constraints.base import ConstraintResult


@dataclass
class _CacheEntry:
    result: ConstraintResult
    timestamp: float


class ConstraintCache:
    """Simple TTL-based cache for constraint check results."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._ttl = ttl_seconds
        self._store: Dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Optional[ConstraintResult]:
        """Return cached result if present and not expired, else None."""
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry.timestamp > self._ttl:
            del self._store[key]
            return None
        return entry.result

    def put(self, key: str, result: ConstraintResult) -> None:
        """Store a result with the current timestamp."""
        self._store[key] = _CacheEntry(result=result, timestamp=time.monotonic())

    def invalidate(self, key: Optional[str] = None) -> None:
        """Invalidate a specific key or all keys."""
        if key is None:
            self._store.clear()
        else:
            self._store.pop(key, None)

    def __len__(self) -> int:
        return len(self._store)

    def keys(self) -> list:
        return list(self._store.keys())
