"""Pipeline summary deliverable."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class PipelineSummary:
    """Generate pipeline execution summary."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def record(self, key: str, value: Any) -> None:
        self._data[key] = value

    def generate(
        self,
        iterations: int,
        benchmarks: List[str],
        overall_improvement: float,
        final_accuracy: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        summary = {
            "pipeline": {
                "total_iterations": iterations,
                "benchmarks_evaluated": benchmarks,
                "num_benchmarks": len(benchmarks),
            },
            "results": {
                "overall_improvement": round(overall_improvement, 4),
                "final_accuracy": round(final_accuracy, 4),
            },
            "metadata": dict(kwargs),
        }
        summary.update(self._data)
        return summary

    def to_markdown(
        self,
        iterations: int,
        benchmarks: List[str],
        overall_improvement: float,
        final_accuracy: float,
        **kwargs: Any,
    ) -> str:
        data = self.generate(
            iterations, benchmarks, overall_improvement, final_accuracy, **kwargs
        )
        lines = [
            "# Pipeline Summary",
            "",
            f"- Iterations: {data['pipeline']['total_iterations']}",
            f"- Benchmarks: {', '.join(data['pipeline']['benchmarks_evaluated'])}",
            f"- Overall Improvement: {data['results']['overall_improvement']}",
            f"- Final Accuracy: {data['results']['final_accuracy']}",
            "",
        ]
        return "\n".join(lines)
