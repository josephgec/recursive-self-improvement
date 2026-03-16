"""Grid visualizer for rendering ARC tasks for LLM consumption."""

from __future__ import annotations

from typing import List

from src.arc.grid import ARCExample, ARCTask, Grid


COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "maroon",
}


class GridVisualizer:
    """Renders ARC grids and tasks for LLM prompts."""

    def __init__(self, use_color_names: bool = False):
        self.use_color_names = use_color_names

    def render_grid(self, grid: Grid) -> str:
        """Render a single grid as a string."""
        lines = [f"Grid ({grid.height}x{grid.width}):"]

        if self.use_color_names:
            for row in grid.data:
                cells = [COLOR_NAMES.get(c, str(c)) for c in row]
                lines.append("  " + " ".join(cells))
        else:
            for row in grid.data:
                lines.append("  " + " ".join(str(c) for c in row))

        return "\n".join(lines)

    def render_example(self, example: ARCExample, index: int = 0) -> str:
        """Render a single input-output example."""
        parts = [f"Example {index + 1}:"]
        parts.append("Input:")
        parts.append(self.render_grid(example.input_grid))
        parts.append("Output:")
        parts.append(self.render_grid(example.output_grid))
        return "\n".join(parts)

    def render_task(self, task: ARCTask) -> str:
        """Render an entire ARC task for LLM consumption."""
        parts = [f"=== ARC Task: {task.task_id} ==="]
        parts.append("")

        parts.append(f"Training Examples ({task.num_train}):")
        parts.append("-" * 40)
        for i, example in enumerate(task.train):
            parts.append(self.render_example(example, i))
            parts.append("")

        parts.append(f"Test Inputs ({task.num_test}):")
        parts.append("-" * 40)
        for i, example in enumerate(task.test):
            parts.append(f"Test {i + 1} Input:")
            parts.append(self.render_grid(example.input_grid))
            parts.append("")

        return "\n".join(parts)

    def render_comparison(
        self,
        input_grid: Grid,
        expected: Grid,
        actual: Grid,
    ) -> str:
        """Render a comparison between expected and actual output."""
        parts = ["Comparison:"]
        parts.append("Input:")
        parts.append(self.render_grid(input_grid))
        parts.append("Expected Output:")
        parts.append(self.render_grid(expected))
        parts.append("Actual Output:")
        parts.append(self.render_grid(actual))

        # Highlight differences
        if expected.shape == actual.shape:
            diffs = []
            for r in range(expected.height):
                for c in range(expected.width):
                    if expected.data[r][c] != actual.data[r][c]:
                        diffs.append(
                            f"  ({r},{c}): expected {expected.data[r][c]}, "
                            f"got {actual.data[r][c]}"
                        )
            if diffs:
                parts.append(f"Differences ({len(diffs)}):")
                parts.extend(diffs[:20])
                if len(diffs) > 20:
                    parts.append(f"  ... and {len(diffs) - 20} more")
            else:
                parts.append("No differences (match!)")
        else:
            parts.append(
                f"Shape mismatch: expected {expected.shape}, "
                f"got {actual.shape}"
            )

        return "\n".join(parts)

    def render_task_compact(self, task: ARCTask) -> str:
        """Render task in compact format for shorter prompts."""
        parts = [f"Task: {task.task_id}"]
        for i, ex in enumerate(task.train):
            parts.append(
                f"Train {i+1}: {ex.input_grid.to_ascii()} -> {ex.output_grid.to_ascii()}"
            )
        for i, ex in enumerate(task.test):
            parts.append(f"Test {i+1}: {ex.input_grid.to_ascii()} -> ?")
        return "\n".join(parts)
