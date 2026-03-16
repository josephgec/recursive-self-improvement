"""Benchmark summary deliverable."""

from __future__ import annotations

from typing import Any, Dict, List


class BenchmarkSummary:
    """Generate per-benchmark summary."""

    def generate(
        self,
        benchmark_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"benchmarks": {}}
        for name, data in benchmark_results.items():
            summary["benchmarks"][name] = {
                "final_accuracy": data.get("final_accuracy", 0.0),
                "improvement": data.get("improvement", 0.0),
                "num_tasks": data.get("num_tasks", 0),
                "categories": data.get("categories", []),
            }
        return summary

    def to_markdown(
        self,
        benchmark_results: Dict[str, Dict[str, Any]],
    ) -> str:
        data = self.generate(benchmark_results)
        lines = ["# Benchmark Summary", ""]
        for name, info in data["benchmarks"].items():
            lines.append(f"## {name}")
            lines.append(f"- Final Accuracy: {info['final_accuracy']:.4f}")
            lines.append(f"- Improvement: {info['improvement']:.4f}")
            lines.append(f"- Tasks: {info['num_tasks']}")
            lines.append("")
        return "\n".join(lines)
