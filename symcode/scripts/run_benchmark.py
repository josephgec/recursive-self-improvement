#!/usr/bin/env python3
"""Run a benchmark evaluation of the SymCode pipeline.

Usage:
    python scripts/run_benchmark.py --benchmark math500 --pipeline both --report
    python scripts/run_benchmark.py --benchmark olympiad --pipeline symcode --max-problems 50
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.benchmarks.math500 import MATH500Loader, BenchmarkProblem
from src.benchmarks.olympiad import OlympiadBenchLoader
from src.benchmarks.runner import BenchmarkRunner
from src.analysis.report import generate_report
from src.utils.logging import setup_logging

import yaml

app = typer.Typer(help="Run SymCode benchmark evaluation.")


def _load_config(config_path: str | None) -> dict:
    """Load YAML config or return defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    default = _PROJECT_ROOT / "configs" / "default.yaml"
    if default.exists():
        with open(default) as f:
            return yaml.safe_load(f)
    return {}


@app.command()
def main(
    benchmark: str = typer.Option(
        "math500", help="Benchmark name: math500 or olympiad"
    ),
    pipeline: str = typer.Option(
        "both", help="Pipeline to run: symcode, prose, or both"
    ),
    model: str = typer.Option(
        "", help="Model name override (e.g. gpt-4o)"
    ),
    config: str = typer.Option(
        "", help="Path to YAML config file"
    ),
    concurrency: int = typer.Option(
        1, help="Number of concurrent workers"
    ),
    max_problems: int = typer.Option(
        0, help="Maximum problems to evaluate (0 = all)"
    ),
    output_dir: str = typer.Option(
        "data/results", help="Directory for output files"
    ),
    report: bool = typer.Option(
        False, help="Generate a markdown report"
    ),
    mock: bool = typer.Option(
        False, help="Use mock LLM for testing"
    ),
) -> None:
    """Run benchmark evaluation."""
    setup_logging()

    cfg = _load_config(config or None)
    if model:
        cfg.setdefault("model", {})["name"] = model

    # Load problems
    typer.echo(f"Loading {benchmark} benchmark...")
    if benchmark == "math500":
        loader = MATH500Loader(
            num_problems=max_problems if max_problems > 0 else 500
        )
    elif benchmark == "olympiad":
        loader = OlympiadBenchLoader(
            num_problems=max_problems if max_problems > 0 else None
        )
    else:
        typer.echo(f"Unknown benchmark: {benchmark}", err=True)
        raise typer.Exit(1)

    problems = loader.load()
    if not problems:
        typer.echo("No problems loaded. Check dataset availability.", err=True)
        raise typer.Exit(1)

    if max_problems > 0:
        problems = problems[:max_problems]

    typer.echo(f"Loaded {len(problems)} problems")

    # Determine pipelines
    if pipeline == "both":
        pipelines = ["symcode", "prose"]
    else:
        pipelines = [pipeline]

    # Run benchmark
    runner = BenchmarkRunner(config=cfg)
    if mock:
        # Override with mock LLM
        from src.pipeline.code_generator import SymCodeGenerator
        from src.pipeline.prose_baseline import ProseBaseline
        runner.generator = SymCodeGenerator(config=cfg, mock=True)
        runner.prose_baseline = ProseBaseline(config=cfg, mock=True)

    typer.echo(f"Running {', '.join(pipelines)} pipeline(s)...")
    result = runner.run(problems, pipelines=pipelines, concurrency=concurrency)

    # Print summary
    typer.echo(f"\n{'='*50}")
    typer.echo(f"Results ({len(problems)} problems):")
    if result.symcode_results:
        typer.echo(f"  SymCode accuracy: {result.symcode_accuracy:.1%}")
    if result.prose_results:
        typer.echo(f"  Prose accuracy:   {result.prose_accuracy:.1%}")
    if result.symcode_results and result.prose_results:
        delta = (result.symcode_accuracy - result.prose_accuracy) * 100
        typer.echo(f"  Delta:            {delta:+.1f}pp")
    typer.echo(f"  Total time:       {result.total_time:.1f}s")
    typer.echo(f"{'='*50}\n")

    # Save results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / f"{benchmark}_results.json"
    runner.save_results(result, results_path)
    typer.echo(f"Results saved to {results_path}")

    # Generate report
    if report:
        report_path = out / f"{benchmark}_report.md"
        generate_report(result, output_path=report_path)
        typer.echo(f"Report saved to {report_path}")


if __name__ == "__main__":
    app()
