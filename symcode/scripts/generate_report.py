#!/usr/bin/env python3
"""Generate a markdown report from saved benchmark results.

Usage:
    python scripts/generate_report.py --results data/results/math500_results.json
    python scripts/generate_report.py --results data/results/math500_results.json --output report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.benchmarks.runner import BenchmarkResult
from src.verification.result_types import (
    AttemptRecord,
    CodeExecutionResult,
    CodeError,
    SolveResult,
)
from src.analysis.report import generate_report
from src.utils.logging import setup_logging

app = typer.Typer(help="Generate analysis report from benchmark results.")


def _load_results(path: Path) -> BenchmarkResult:
    """Load BenchmarkResult from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))

    def _parse_solve_results(entries: list[dict]) -> list[SolveResult]:
        results = []
        for entry in entries:
            attempts = []
            for a in entry.get("attempts", []):
                error = None
                error_type = a.get("error_type")
                if error_type:
                    error = CodeError(
                        error_type=error_type,
                        message="",
                    )
                exec_result = CodeExecutionResult(
                    success=error is None,
                    error=error,
                )
                attempts.append(
                    AttemptRecord(
                        attempt_number=a.get("attempt_number", 0),
                        code="",
                        execution_result=exec_result,
                        extracted_answer=a.get("extracted_answer"),
                        answer_correct=a.get("answer_correct"),
                        feedback=a.get("feedback", ""),
                    )
                )

            results.append(
                SolveResult(
                    problem=entry.get("problem", ""),
                    expected_answer=entry.get("expected_answer"),
                    final_answer=entry.get("final_answer"),
                    correct=entry.get("correct", False),
                    num_attempts=entry.get("num_attempts", 0),
                    attempts=attempts,
                    task_type=entry.get("task_type", ""),
                    pipeline=entry.get("pipeline", ""),
                    total_time=entry.get("total_time", 0.0),
                )
            )
        return results

    result = BenchmarkResult(
        symcode_results=_parse_solve_results(data.get("symcode_results", [])),
        prose_results=_parse_solve_results(data.get("prose_results", [])),
        symcode_accuracy=data.get("symcode_accuracy", 0.0),
        prose_accuracy=data.get("prose_accuracy", 0.0),
        total_time=data.get("total_time", 0.0),
        num_problems=data.get("num_problems", 0),
    )
    return result


@app.command()
def main(
    results: str = typer.Option(
        ..., help="Path to benchmark results JSON file"
    ),
    output: str = typer.Option(
        "", help="Output path for markdown report"
    ),
    title: str = typer.Option(
        "SymCode Benchmark Report", help="Report title"
    ),
) -> None:
    """Generate a report from saved results."""
    setup_logging()

    results_path = Path(results)
    if not results_path.exists():
        typer.echo(f"Results file not found: {results_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading results from {results_path}...")
    benchmark_result = _load_results(results_path)

    if not output:
        output = str(results_path.with_suffix(".md"))

    report_text = generate_report(
        benchmark_result, output_path=output, title=title
    )
    typer.echo(f"Report generated: {output}")
    typer.echo(f"\nPreview (first 500 chars):\n{report_text[:500]}")


if __name__ == "__main__":
    app()
