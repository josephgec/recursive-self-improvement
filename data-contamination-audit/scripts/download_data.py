#!/usr/bin/env python
"""Simple data acquisition script.

Downloads Wikipedia and Common Crawl data for configured time bins and saves
the resulting temporal corpus to disk.

Usage
-----
    python scripts/download_data.py --config configs/default.yaml
    python scripts/download_data.py --output-dir data/custom/
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.common_crawl import Document
from src.data.sampler import build_temporal_corpus

console = Console()
logger = logging.getLogger("download_data")

app = typer.Typer(help="Download data for the contamination audit pipeline.")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@app.command()
def main(
    config: Path = typer.Option(
        _PROJECT_ROOT / "configs" / "default.yaml",
        "--config",
        help="Path to the pipeline configuration YAML file.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Output directory. Defaults to the project data/ directory.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Download Wikipedia and Common Crawl data for the configured time bins."""
    _setup_logging(verbose)

    cfg = _load_config(config)
    logger.info("Loaded config from %s", config)

    # Resolve output directory
    if output_dir is None:
        output_dir = Path(cfg.get("project", {}).get("output_dir", "data"))
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read sampling configuration
    sampling_cfg = cfg.get("sampling", {})
    sources = sampling_cfg.get("sources", ["wikipedia", "common_crawl"])
    n_per_bin = sampling_cfg.get("n_per_bin", 5000)
    bins = sampling_cfg.get("bins", [])
    bin_size = sampling_cfg.get("bin_size", "year")
    seed = cfg.get("project", {}).get("seed", 42)
    cc_languages = sampling_cfg.get("cc_languages", ["en"])

    console.print(f"[bold]Sources:[/bold] {sources}")
    console.print(f"[bold]Bins:[/bold] {bins}")
    console.print(f"[bold]Documents per bin per source:[/bold] {n_per_bin}")

    # Build temporal corpus
    corpus = build_temporal_corpus(
        sources=sources,
        n_per_bin=n_per_bin,
        bins=bins,
        seed=seed,
        bin_size=bin_size,
        cc_languages=cc_languages,
    )

    # Save corpus
    corpus_path = output_dir / "corpus.pkl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "wb") as f:
        pickle.dump(corpus, f)

    total = sum(len(v) for v in corpus.values())
    console.print(
        f"\n[green]Downloaded {total} documents across {len(corpus)} time bins.[/green]"
    )
    console.print(f"[green]Corpus saved to {corpus_path}[/green]")

    # Print per-bin counts
    for bin_label in sorted(corpus.keys()):
        docs = corpus[bin_label]
        console.print(f"  {bin_label}: {len(docs)} documents")


if __name__ == "__main__":
    app()
