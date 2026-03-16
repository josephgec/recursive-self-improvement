"""Candidate pool for managing and selecting candidates."""
from __future__ import annotations

from typing import List, Optional

from src.outer_loop.strategy_evolver import Candidate


class CandidatePool:
    """Pool of candidates with selection strategies."""

    def __init__(self):
        self._candidates: List[Candidate] = []

    def add(self, candidate: Candidate) -> None:
        """Add a candidate to the pool."""
        self._candidates.append(candidate)

    def add_many(self, candidates: List[Candidate]) -> None:
        """Add multiple candidates to the pool."""
        self._candidates.extend(candidates)

    def get_best(self, n: int = 1) -> List[Candidate]:
        """Get top-n candidates by score."""
        sorted_candidates = sorted(self._candidates, key=lambda c: c.score, reverse=True)
        return sorted_candidates[:n]

    def get_diverse(self, n: int = 3) -> List[Candidate]:
        """Get n diverse candidates (different targets and operators)."""
        seen_keys: set = set()
        result: List[Candidate] = []
        for c in self._candidates:
            key = (c.target, c.operator)
            if key not in seen_keys:
                result.append(c)
                seen_keys.add(key)
                if len(result) >= n:
                    break
        # fill remaining slots if not enough diverse candidates
        if len(result) < n:
            for c in self._candidates:
                if c not in result:
                    result.append(c)
                    if len(result) >= n:
                        break
        return result

    @property
    def size(self) -> int:
        return len(self._candidates)

    def clear(self) -> None:
        """Clear all candidates from the pool."""
        self._candidates.clear()

    @property
    def candidates(self) -> List[Candidate]:
        return list(self._candidates)
