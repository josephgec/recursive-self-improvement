#!/usr/bin/env python
"""Reserve export script.

Exports the filtered clean data reserve to Parquet format with a companion
summary JSON.

Usage
-----
    python scripts/export_reserve.py --reserve-dir data/reserve/
    python scripts/export_reserve.py --config configs/default.yaml --reserve-dir data/reserve/
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
from src.reserve.export import export_reserve

console = Console()
logger = logging.getLogger("export_reserve")

app = typer.Typer(help="Export the clean data reserve.")


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
    reserve_dir: Path = typer.Option(
        ...,
        "--reserve-dir",
        help="Directory containing reserve_documents.pkl, or output directory for export.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Export the reserve documents to Parquet with a summary JSON."""
    _setup_logging(verbose)

    cfg = _load_config(config)
    logger.info("Loaded config from %s", config)

    reserve_cfg = cfg.get("reserve", {})
    threshold = reserve_cfg.get("threshold", 0.90)

    # Resolve the reserve directory
    if not reserve_dir.is_absolute():
        reserve_dir = _PROJECT_ROOT / reserve_dir
    reserve_dir.mkdir(parents=True, exist_ok=True)

    # Load reserve documents
    docs_path = reserve_dir / "reserve_documents.pkl"
    if not docs_path.exists():
        console.print(
            f"[red]Error: reserve documents not found at {docs_path}[/red]"
        )
        console.print(
            "[yellow]Run the pipeline filter step first, or provide the "
            "correct --reserve-dir.[/yellow]"
        )
        raise typer.Exit(code=1)

    with open(docs_path, "rb") as f:
        reserve_docs: list[Document] = pickle.load(f)

    console.print(f"[bold]Loaded {len(reserve_docs)} reserve documents[/bold]")

    # Determine full corpus size for alpha_t computation
    # Try to load from parent directory
    parent_dir = reserve_dir.parent
    all_docs_path = parent_dir / "all_documents.pkl"
    if all_docs_path.exists():
        with open(all_docs_path, "rb") as f:
            all_docs: list[Document] = pickle.load(f)
        full_corpus_size = len(all_docs)
        console.print(f"[bold]Full corpus size:[/bold] {full_corpus_size}")
    else:
        full_corpus_size = len(reserve_docs)
        logger.warning(
            "Could not find all_documents.pkl — using reserve size as corpus size"
        )

    # Export
    export_reserve(
        documents=reserve_docs,
        output_dir=reserve_dir,
        format="parquet",
        full_corpus_size=full_corpus_size,
        threshold=threshold,
    )

    console.print(f"\n[green]Reserve exported to {reserve_dir}[/green]")
    console.print(f"  - {reserve_dir / 'reserve.parquet'}")
    console.print(f"  - {reserve_dir / 'summary.json'}")


if __name__ == "__main__":
    app()
