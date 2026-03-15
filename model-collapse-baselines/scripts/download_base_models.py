#!/usr/bin/env python3
"""Download pretrained base model checkpoints.

Fetches TinyLlama-1.1B and/or Mistral-7B from HuggingFace Hub
so they are available for offline experiments.

Usage::

    python scripts/download_base_models.py --model 1b
    python scripts/download_base_models.py --model 7b
    python scripts/download_base_models.py --model all
    python scripts/download_base_models.py --model all --cache-dir ./models
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Download pretrained base models from HuggingFace Hub.")

logger = logging.getLogger(__name__)

MODEL_IDS = {
    "1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "7b": "mistralai/Mistral-7B-v0.1",
    "debug": "sshleifer/tiny-gpt2",
}


@app.command()
def main(
    model: str = typer.Option(
        "1b", "--model", "-m",
        help="Which model to download: '1b', '7b', 'debug', or 'all'.",
    ),
    cache_dir: Optional[Path] = typer.Option(
        None, "--cache-dir", "-d",
        help="Directory to cache downloaded models. Defaults to HF cache.",
    ),
) -> None:
    """Download one or more pretrained models."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if model == "all":
        models_to_download = list(MODEL_IDS.keys())
    else:
        if model not in MODEL_IDS:
            typer.echo(
                f"Unknown model '{model}'. Choose from: {list(MODEL_IDS.keys())}",
                err=True,
            )
            raise typer.Exit(1)
        models_to_download = [model]

    for model_key in models_to_download:
        model_id = MODEL_IDS[model_key]
        typer.echo(f"Downloading {model_key}: {model_id}")

        try:
            _download_model(model_id, cache_dir)
            typer.echo(f"  Done: {model_id}")
        except Exception as e:
            typer.echo(f"  Failed: {e}", err=True)


def _download_model(model_id: str, cache_dir: Optional[Path]) -> None:
    """Download model and tokenizer weights from HuggingFace Hub."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)

    logger.info("Downloading tokenizer: %s", model_id)
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, **kwargs)

    logger.info("Downloading model: %s", model_id)
    AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, **kwargs,
    )

    logger.info("Successfully downloaded %s", model_id)


if __name__ == "__main__":
    app()
