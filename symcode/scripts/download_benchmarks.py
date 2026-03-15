#!/usr/bin/env python3
"""Download benchmark datasets for offline use.

Usage:
    python scripts/download_benchmarks.py
    python scripts/download_benchmarks.py --benchmark math500 --cache-dir data/datasets
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logging import setup_logging

app = typer.Typer(help="Download benchmark datasets.")


@app.command()
def main(
    benchmark: str = typer.Option(
        "all", help="Benchmark to download: math500, olympiad, or all"
    ),
    cache_dir: str = typer.Option(
        "data/datasets", help="Directory to cache datasets"
    ),
) -> None:
    """Download benchmark datasets for offline evaluation."""
    setup_logging()

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    if benchmark in ("all", "math500"):
        _download_math500(cache)

    if benchmark in ("all", "olympiad"):
        _download_olympiad(cache)


def _download_math500(cache_dir: Path) -> None:
    """Download the MATH-500 dataset."""
    typer.echo("Downloading MATH-500 (hendrycks/competition_math)...")
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "hendrycks/competition_math",
            split="test",
            cache_dir=str(cache_dir),
            trust_remote_code=True,
        )
        typer.echo(f"  Downloaded {len(ds)} problems")
        typer.echo(f"  Cached to {cache_dir}")
    except ImportError:
        typer.echo(
            "  ERROR: 'datasets' library not installed. "
            "Run: pip install datasets",
            err=True,
        )
    except Exception as e:
        typer.echo(f"  ERROR: Could not download MATH-500: {e}", err=True)


def _download_olympiad(cache_dir: Path) -> None:
    """Download the OlympiadBench dataset."""
    typer.echo("Downloading OlympiadBench...")
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "olympiad_bench",
            split="test",
            cache_dir=str(cache_dir),
            trust_remote_code=True,
        )
        typer.echo(f"  Downloaded {len(ds)} problems")
        typer.echo(f"  Cached to {cache_dir}")
    except ImportError:
        typer.echo(
            "  ERROR: 'datasets' library not installed. "
            "Run: pip install datasets",
            err=True,
        )
    except Exception as e:
        typer.echo(
            f"  WARNING: Could not download OlympiadBench: {e}. "
            "Synthetic problems will be used as fallback.",
            err=True,
        )
        typer.echo("  Using built-in synthetic olympiad problems instead.")


if __name__ == "__main__":
    app()
