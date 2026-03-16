"""Result aggregation from multiple REPL executions."""

from collections import Counter
from typing import Any, Dict, List, Optional

from src.protocol.final_functions import FinalResult


class ResultAggregator:
    """Aggregates results from multiple REPL executions.

    Supports collecting, voting, concatenating, and merging
    results from parallel or sequential executions.
    """

    def __init__(self):
        self._results: List[FinalResult] = []

    def collect(self, result: FinalResult) -> None:
        """Add a result to the collection.

        Args:
            result: The FinalResult to add.
        """
        self._results.append(result)

    def concatenate(self) -> str:
        """Concatenate all result values as strings.

        Returns:
            Concatenated string of all results.
        """
        parts = []
        for r in self._results:
            if r.has_result:
                parts.append(str(r.value))
        return "\n".join(parts)

    def vote(self) -> Optional[Any]:
        """Return the most common result value (majority vote).

        Returns:
            The most common result value, or None if no results.
        """
        values = [r.value for r in self._results if r.has_result]
        if not values:
            return None

        # Convert to string representations for counting
        str_counts: Counter = Counter()
        value_map: Dict[str, Any] = {}
        for v in values:
            key = repr(v)
            str_counts[key] += 1
            value_map[key] = v

        most_common = str_counts.most_common(1)[0][0]
        return value_map[most_common]

    def merge_structured(self) -> Dict[str, Any]:
        """Merge structured (dict) results.

        Returns:
            Merged dictionary of all dict-type results.
        """
        merged: Dict[str, Any] = {}
        for r in self._results:
            if r.has_result and isinstance(r.value, dict):
                merged.update(r.value)
        return merged

    def inject_helpers(self, namespace: Dict[str, Any]) -> None:
        """Inject aggregation helper functions into a namespace.

        Args:
            namespace: The namespace to inject into.
        """
        agg = self

        def collect_result(value: Any) -> None:
            agg.collect(FinalResult(value=value, source="collect"))

        namespace["collect_result"] = collect_result

    @property
    def results(self) -> List[FinalResult]:
        """All collected results."""
        return list(self._results)

    @property
    def count(self) -> int:
        """Number of collected results."""
        return len(self._results)

    def clear(self) -> None:
        """Clear all collected results."""
        self._results.clear()
