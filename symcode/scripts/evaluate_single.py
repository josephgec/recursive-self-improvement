#!/usr/bin/env python3
"""Evaluate a single math problem through the SymCode pipeline.

Usage:
    python scripts/evaluate_single.py --problem "Solve x^2 - 4 = 0"
    python scripts/evaluate_single.py --problem "What is 2+3?" --pipeline prose --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.code_generator import SymCodeGenerator
from src.pipeline.prose_baseline import ProseBaseline
from src.pipeline.router import TaskRouter
from src.verification.answer_checker import AnswerChecker
from src.verification.retry_loop import RetryLoop
from src.utils.logging import setup_logging

import yaml
import logging

app = typer.Typer(help="Evaluate a single math problem.")


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
    problem: str = typer.Option(..., help="Math problem to solve"),
    pipeline: str = typer.Option("symcode", help="Pipeline: symcode or prose"),
    expected: str = typer.Option("", help="Expected answer for verification"),
    config: str = typer.Option("", help="YAML config path"),
    verbose: bool = typer.Option(False, help="Show detailed output"),
    mock: bool = typer.Option(False, help="Use mock LLM"),
) -> None:
    """Evaluate a single problem."""
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    cfg = _load_config(config or None)

    # Route the problem
    router = TaskRouter(cfg)
    routing = router.heuristic_route(problem)
    typer.echo(f"Routing: {routing.task_type.value} ({'SymCode' if routing.use_symcode else 'Prose'})")

    if pipeline == "symcode":
        generator = SymCodeGenerator(config=cfg, mock=mock)
        checker = AnswerChecker()
        retry_loop = RetryLoop(
            generator=generator,
            checker=checker,
            max_retries=cfg.get("verification", {}).get("max_retries", 3),
        )

        result = retry_loop.solve(
            problem=problem,
            task_type=routing.task_type,
            expected_answer=expected or None,
        )

        typer.echo(f"\nPipeline: SymCode")
        typer.echo(f"Answer: {result.final_answer}")
        typer.echo(f"Correct: {result.correct}")
        typer.echo(f"Attempts: {result.num_attempts}")
        typer.echo(f"Time: {result.total_time:.2f}s")

        if verbose:
            for a in result.attempts:
                typer.echo(f"\n--- Attempt {a.attempt_number} ---")
                typer.echo(f"Code:\n{a.code}")
                typer.echo(f"Answer: {a.extracted_answer}")
                typer.echo(f"Correct: {a.answer_correct}")
                if a.feedback:
                    typer.echo(f"Feedback: {a.feedback[:300]}")

    elif pipeline == "prose":
        prose = ProseBaseline(config=cfg, mock=mock)
        answer, response = prose.solve(problem)

        typer.echo(f"\nPipeline: Prose")
        typer.echo(f"Answer: {answer}")

        if expected:
            checker = AnswerChecker()
            correct = checker.check(answer, expected) if answer else False
            typer.echo(f"Correct: {correct}")

        if verbose:
            typer.echo(f"\nFull response:\n{response}")

    else:
        typer.echo(f"Unknown pipeline: {pipeline}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
