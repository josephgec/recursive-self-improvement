"""ResultAggregator: combine results from sub-queries."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List


class ResultAggregator:
    """Aggregate results from multiple sub-sessions."""

    @staticmethod
    def concatenate(results: List[Any], separator: str = "\n") -> str:
        """Concatenate all results into a single string."""
        return separator.join(str(r) for r in results)

    @staticmethod
    def vote(results: List[Any]) -> Any:
        """Return the most common result (majority vote)."""
        if not results:
            return None
        counter = Counter(str(r) for r in results)
        winner, _ = counter.most_common(1)[0]
        # Return the original object that matches
        for r in results:
            if str(r) == winner:
                return r
        return winner

    @staticmethod
    def merge_structured(results: List[Any]) -> Dict[str, Any]:
        """Merge results that are dicts into a single dict."""
        merged: Dict[str, Any] = {}
        for r in results:
            if isinstance(r, dict):
                merged.update(r)
            elif isinstance(r, str):
                try:
                    parsed = json.loads(r)
                    if isinstance(parsed, dict):
                        merged.update(parsed)
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
                merged[f"result_{len(merged)}"] = r
            else:
                merged[f"result_{len(merged)}"] = r
        return merged

    @staticmethod
    def inject_helpers(repl: Dict[str, Any]) -> None:
        """Inject aggregation helper functions into a REPL namespace."""

        def _aggregate_concat(results: List[Any], separator: str = "\n") -> str:
            return ResultAggregator.concatenate(results, separator)

        def _aggregate_vote(results: List[Any]) -> Any:
            return ResultAggregator.vote(results)

        def _aggregate_merge(results: List[Any]) -> Dict[str, Any]:
            return ResultAggregator.merge_structured(results)

        repl["aggregate_concat"] = _aggregate_concat
        repl["aggregate_vote"] = _aggregate_vote
        repl["aggregate_merge"] = _aggregate_merge
