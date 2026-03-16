"""Final report deliverable: package all results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.deliverables.pipeline_summary import PipelineSummary
from src.deliverables.benchmark_summary import BenchmarkSummary
from src.deliverables.ablation_summary import AblationSummary
from src.ablation.ablation_study import AblationResult


class FinalReport:
    """Generate final comprehensive report."""

    def __init__(self) -> None:
        self._pipeline_summary = PipelineSummary()
        self._benchmark_summary = BenchmarkSummary()
        self._ablation_summary = AblationSummary()

    def generate(
        self,
        iterations: int,
        benchmarks: List[str],
        overall_improvement: float,
        final_accuracy: float,
        benchmark_results: Dict[str, Dict[str, Any]],
        ablation_result: Optional[AblationResult] = None,
        collapse_prevention_score: float = 0.0,
        sustainability_score: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "pipeline": self._pipeline_summary.generate(
                iterations, benchmarks, overall_improvement, final_accuracy
            ),
            "benchmarks": self._benchmark_summary.generate(benchmark_results),
            "collapse_prevention_score": round(collapse_prevention_score, 4),
            "sustainability_score": round(sustainability_score, 4),
        }

        if ablation_result:
            report["ablation"] = self._ablation_summary.generate(ablation_result)

        report["metadata"] = dict(kwargs)
        return report

    def to_markdown(
        self,
        iterations: int,
        benchmarks: List[str],
        overall_improvement: float,
        final_accuracy: float,
        benchmark_results: Dict[str, Dict[str, Any]],
        ablation_result: Optional[AblationResult] = None,
        collapse_prevention_score: float = 0.0,
        sustainability_score: float = 0.0,
        **kwargs: Any,
    ) -> str:
        lines = [
            "# RSI Benchmark Final Report",
            "",
            self._pipeline_summary.to_markdown(
                iterations, benchmarks, overall_improvement, final_accuracy
            ),
            self._benchmark_summary.to_markdown(benchmark_results),
        ]

        lines.append("# Collapse Analysis")
        lines.append(f"- Collapse Prevention Score: {collapse_prevention_score:.4f}")
        lines.append(f"- Sustainability Score: {sustainability_score:.4f}")
        lines.append("")

        if ablation_result:
            lines.append(self._ablation_summary.to_markdown(ablation_result))

        return "\n".join(lines)
