"""Grid difference computation utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.arc.grid import Grid, GridDiff, diff_grids


def compute_grid_diff(grid_a: Grid, grid_b: Grid) -> GridDiff:
    """Compute the difference between two grids."""
    return diff_grids(grid_a, grid_b)


def diff_summary(grid_diff: GridDiff) -> str:
    """Generate human-readable diff summary."""
    return grid_diff.summary()


def highlight_changes(grid_a: Grid, grid_b: Grid) -> str:
    """Highlight pixel-level changes between grids."""
    d = diff_grids(grid_a, grid_b)
    lines = []

    if d.shape_changed:
        lines.append(f"Shape changed: {d.old_shape} -> {d.new_shape}")

    if d.num_changes == 0:
        lines.append("No changes")
        return "\n".join(lines)

    lines.append(f"Total changes: {d.num_changes}")
    lines.append(f"Change ratio: {d.change_ratio:.2%}")

    # Group changes by type
    color_changes: Dict[Tuple[int, int], int] = {}
    for r, c, old, new in d.changed_cells:
        key = (old, new)
        color_changes[key] = color_changes.get(key, 0) + 1

    if color_changes:
        lines.append("Color transitions:")
        for (old, new), count in sorted(
            color_changes.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {old} -> {new}: {count} cells")

    return "\n".join(lines)


def compute_accuracy_map(output: Grid, expected: Grid) -> List[List[bool]]:
    """Create a boolean map showing which pixels are correct."""
    if output.shape != expected.shape:
        return []

    accuracy_map = []
    for r in range(output.height):
        row = []
        for c in range(output.width):
            row.append(output.data[r][c] == expected.data[r][c])
        accuracy_map.append(row)

    return accuracy_map


def common_errors(
    outputs: List[Grid],
    expected: List[Grid],
) -> Dict[str, int]:
    """Find common error patterns across multiple examples."""
    error_types: Dict[str, int] = {}

    for out, exp in zip(outputs, expected):
        if out.shape != exp.shape:
            error_types["shape_mismatch"] = error_types.get("shape_mismatch", 0) + 1
            continue

        diff = diff_grids(out, exp)
        if diff.num_changes == 0:
            continue

        ratio = diff.change_ratio
        if ratio > 0.8:
            error_types["mostly_wrong"] = error_types.get("mostly_wrong", 0) + 1
        elif ratio > 0.3:
            error_types["partially_wrong"] = error_types.get("partially_wrong", 0) + 1
        else:
            error_types["minor_errors"] = error_types.get("minor_errors", 0) + 1

    return error_types
