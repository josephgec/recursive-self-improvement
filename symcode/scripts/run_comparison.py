#!/usr/bin/env python3
"""Run a head-to-head comparison of SymCode vs prose baseline.

Usage:
    python scripts/run_comparison.py --benchmark math500 --max-problems 100
    python scripts/run_comparison.py --benchmark olympiad --output-dir data/comparison
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.benchmarks.math500 import MATH500Loader
from src.benchmarks.olympiad import OlympiadBenchLoader
from src.benchmarks.runner import BenchmarkRunner
from src.benchmarks.comparison import ComparisonAnalyzer
from src.benchmarks.metrics import MetricsComputer
from src.analysis.report import generate_report
from src.utils.logging import setup_logging

import yaml

app = typer.Typer(help="Head-to-head SymCode vs prose comparison.")


def _load_config(config_path: str | None) -> dict:
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
    benchmark: str = typer.Option("math500", help="Benchmark: math500 or olympiad"),
    config: str = typer.Option("", help="YAML config path"),
    max_problems: int = typer.Option(0, help="Max problems (0 = all)"),
    concurrency: int = typer.Option(1, help="Concurrent workers"),
    output_dir: str = typer.Option("data/comparison", help="Output directory"),
    mock: bool = typer.Option(False, help="Use mock LLM"),
) -> None:
    """Run head-to-head comparison."""
    setup_logging()

    cfg = _load_config(config or None)

    # Load problems
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
    if max_problems > 0:
        problems = problems[:max_problems]

    if not problems:
        typer.echo("No problems loaded.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(problems)} problems")

    # Run both pipelines
    runner = BenchmarkRunner(config=cfg)
    if mock:
        from src.pipeline.code_generator import SymCodeGenerator
        from src.pipeline.prose_baseline import ProseBaseline
        runner.generator = SymCodeGenerator(config=cfg, mock=True)
        runner.prose_baseline = ProseBaseline(config=cfg, mock=True)

    typer.echo("Running SymCode pipeline...")
    result = runner.run(problems, pipelines=["symcode", "prose"], concurrency=concurrency)

    # Comparison analysis
    if result.symcode_results and result.prose_results:
        analyzer = ComparisonAnalyzer(result.symcode_results, result.prose_results)
        summary = analyzer.summary()
        stat = analyzer.statistical_test()
        failure = analyzer.failure_mode_comparison()

        typer.echo(f"\n{'='*60}")
        typer.echo("COMPARISON SUMMARY")
        typer.echo(f"{'='*60}")
        typer.echo(f"  SymCode accuracy: {summary['symcode_accuracy']:.1%}")
        typer.echo(f"  Prose accuracy:   {summary['prose_accuracy']:.1%}")
        typer.echo(f"  Delta:            {summary['delta_pp']:+.1f}pp")
        typer.echo(f"  McNemar p-value:  {stat['p_value']:.4f}" if stat['p_value'] is not None else "  McNemar: N/A")
        typer.echo(f"  Significant:      {'Yes' if stat.get('significant_005') else 'No'}")
        typer.echo(f"  SymCode-only:     {summary['symcode_only_correct']}")
        typer.echo(f"  Prose-only:       {summary['prose_only_correct']}")
        typer.echo(f"{'='*60}\n")

        # Save
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        comparison_data = {
            "summary": summary,
            "statistical_test": stat,
            "failure_modes": failure,
            "per_subject": analyzer.per_subject_comparison(),
        }
        comp_path = out / f"{benchmark}_comparison.json"
        comp_path.write_text(
            json.dumps(comparison_data, indent=2, default=str), encoding="utf-8"
        )
        typer.echo(f"Comparison saved to {comp_path}")

        # Full results
        runner.save_results(result, out / f"{benchmark}_results.json")

        # Report
        report_path = out / f"{benchmark}_report.md"
        generate_report(result, output_path=report_path)
        typer.echo(f"Report saved to {report_path}")


if __name__ == "__main__":
    app()
