"""Compare agent snapshots before and after scenarios."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.utils.metrics import (
    compute_code_similarity,
    compute_cyclomatic_complexity,
    count_ast_nodes,
)


class SnapshotComparator:
    """Compare two agent snapshots to measure the impact of a scenario."""

    def compare(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """Compare before/after snapshots.

        Expected snapshot keys:
            - code: str (source code)
            - accuracy: float
            - complexity: int (optional)
            - modification_count: int (optional)
            - functional: bool (optional)

        Returns:
            Comparison dict with deltas and analysis.
        """
        result: Dict[str, Any] = {}

        # Code change analysis
        code_before = before.get("code", "")
        code_after = after.get("code", "")
        result["code_changed"] = code_before != code_after
        result["code_similarity"] = compute_code_similarity(code_before, code_after)

        # Accuracy delta
        acc_before = before.get("accuracy", 0.0)
        acc_after = after.get("accuracy", 0.0)
        result["accuracy_before"] = acc_before
        result["accuracy_after"] = acc_after
        result["accuracy_delta"] = acc_after - acc_before

        # Complexity delta
        comp_before = before.get("complexity")
        comp_after = after.get("complexity")

        if comp_before is None and code_before:
            comp_before = count_ast_nodes(code_before)
        if comp_after is None and code_after:
            comp_after = count_ast_nodes(code_after)

        if comp_before is not None and comp_after is not None:
            result["complexity_before"] = comp_before
            result["complexity_after"] = comp_after
            result["complexity_delta"] = comp_after - comp_before
        else:
            result["complexity_delta"] = 0

        # Cyclomatic complexity
        if code_before and code_after:
            cc_before = compute_cyclomatic_complexity(code_before)
            cc_after = compute_cyclomatic_complexity(code_after)
            result["cyclomatic_before"] = cc_before
            result["cyclomatic_after"] = cc_after
            result["cyclomatic_delta"] = cc_after - cc_before

        # Modification count
        mod_before = before.get("modification_count", 0)
        mod_after = after.get("modification_count", 0)
        result["modifications_during"] = mod_after - mod_before

        # Functional status
        result["functional_before"] = before.get("functional", True)
        result["functional_after"] = after.get("functional", True)
        result["lost_functionality"] = (
            before.get("functional", True) and not after.get("functional", True)
        )

        return result
