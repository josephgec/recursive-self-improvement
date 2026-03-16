"""Grid, GridDiff, and ARCTask dataclasses for ARC domain."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class Grid:
    """A 2D grid of integer color values (0-9)."""

    data: List[List[int]]

    def __post_init__(self):
        if not self.data:
            raise ValueError("Grid cannot be empty")
        row_len = len(self.data[0])
        for i, row in enumerate(self.data):
            if len(row) != row_len:
                raise ValueError(
                    f"Row {i} has length {len(row)}, expected {row_len}"
                )

    @property
    def height(self) -> int:
        return len(self.data)

    @property
    def width(self) -> int:
        return len(self.data[0]) if self.data else 0

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def to_list(self) -> List[List[int]]:
        """Return a deep copy of the grid data as nested lists."""
        return copy.deepcopy(self.data)

    def to_ascii(self) -> str:
        """Render grid as ASCII art with single-char color codes."""
        lines = []
        for row in self.data:
            lines.append("".join(str(c) for c in row))
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        return self.data == other.data

    def __hash__(self) -> int:
        return hash(tuple(tuple(row) for row in self.data))

    def get(self, row: int, col: int) -> int:
        """Get value at (row, col)."""
        return self.data[row][col]

    def set(self, row: int, col: int, value: int) -> None:
        """Set value at (row, col)."""
        self.data[row][col] = value

    def copy(self) -> Grid:
        """Return a deep copy of this grid."""
        return Grid(copy.deepcopy(self.data))

    def colors_used(self) -> set:
        """Return set of distinct color values in the grid."""
        colors = set()
        for row in self.data:
            for val in row:
                colors.add(val)
        return colors

    def pixel_count(self) -> int:
        """Return total number of pixels."""
        return self.height * self.width

    @classmethod
    def from_list(cls, data: List[List[int]]) -> Grid:
        """Create Grid from nested list."""
        return cls(data=copy.deepcopy(data))

    @classmethod
    def zeros(cls, height: int, width: int) -> Grid:
        """Create a grid filled with zeros."""
        return cls(data=[[0] * width for _ in range(height)])


@dataclass
class GridDiff:
    """Represents differences between two grids."""

    changed_cells: List[Tuple[int, int, int, int]]  # (row, col, old, new)
    shape_changed: bool
    old_shape: Tuple[int, int]
    new_shape: Tuple[int, int]

    @property
    def num_changes(self) -> int:
        return len(self.changed_cells)

    @property
    def change_ratio(self) -> float:
        """Ratio of changed cells to total cells in the larger grid."""
        max_pixels = max(
            self.old_shape[0] * self.old_shape[1],
            self.new_shape[0] * self.new_shape[1],
            1,
        )
        return self.num_changes / max_pixels

    def summary(self) -> str:
        """Human-readable summary of changes."""
        parts = []
        if self.shape_changed:
            parts.append(
                f"Shape: {self.old_shape} -> {self.new_shape}"
            )
        parts.append(f"Changed cells: {self.num_changes}")
        parts.append(f"Change ratio: {self.change_ratio:.2%}")
        return "; ".join(parts)


def diff_grids(grid_a: Grid, grid_b: Grid) -> GridDiff:
    """Compute the difference between two grids."""
    shape_changed = grid_a.shape != grid_b.shape
    changed = []

    min_h = min(grid_a.height, grid_b.height)
    min_w = min(grid_a.width, grid_b.width)

    for r in range(min_h):
        for c in range(min_w):
            if grid_a.data[r][c] != grid_b.data[r][c]:
                changed.append((r, c, grid_a.data[r][c], grid_b.data[r][c]))

    # Cells only in grid_a (if grid_a is larger)
    for r in range(min_h, grid_a.height):
        for c in range(grid_a.width):
            changed.append((r, c, grid_a.data[r][c], -1))

    for r in range(min_h):
        for c in range(min_w, grid_a.width):
            changed.append((r, c, grid_a.data[r][c], -1))

    # Cells only in grid_b (if grid_b is larger)
    for r in range(min_h, grid_b.height):
        for c in range(grid_b.width):
            changed.append((r, c, -1, grid_b.data[r][c]))

    for r in range(min_h):
        for c in range(min_w, grid_b.width):
            changed.append((r, c, -1, grid_b.data[r][c]))

    return GridDiff(
        changed_cells=changed,
        shape_changed=shape_changed,
        old_shape=grid_a.shape,
        new_shape=grid_b.shape,
    )


@dataclass
class ARCExample:
    """A single input-output example pair."""

    input_grid: Grid
    output_grid: Grid


@dataclass
class ARCTask:
    """An ARC task with train and test examples."""

    task_id: str
    train: List[ARCExample]
    test: List[ARCExample]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_train(self) -> int:
        return len(self.train)

    @property
    def num_test(self) -> int:
        return len(self.test)

    def all_examples(self) -> List[ARCExample]:
        """Return all train and test examples."""
        return self.train + self.test

    @classmethod
    def from_dict(cls, task_id: str, data: Dict[str, Any]) -> ARCTask:
        """Create ARCTask from a dictionary (standard ARC JSON format)."""
        train = []
        for ex in data.get("train", []):
            train.append(ARCExample(
                input_grid=Grid.from_list(ex["input"]),
                output_grid=Grid.from_list(ex["output"]),
            ))

        test = []
        for ex in data.get("test", []):
            test.append(ARCExample(
                input_grid=Grid.from_list(ex["input"]),
                output_grid=Grid.from_list(ex["output"]),
            ))

        return cls(
            task_id=task_id,
            train=train,
            test=test,
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            "train": [
                {
                    "input": ex.input_grid.to_list(),
                    "output": ex.output_grid.to_list(),
                }
                for ex in self.train
            ],
            "test": [
                {
                    "input": ex.input_grid.to_list(),
                    "output": ex.output_grid.to_list(),
                }
                for ex in self.test
            ],
        }
