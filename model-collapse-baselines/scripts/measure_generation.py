#!/usr/bin/env python3
"""Compute all metrics for one generation's model checkpoint.

Usage::

    python scripts/measure_generation.py --model-path data/checkpoints/generation_00/model
    python scripts/measure_generation.py --model-path data/checkpoints/generation_02/model --generation 2
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Compute all metrics for a single generation's model.")

logger = logging.getLogger(__name__)


def _load_config(config_path: Path) -> dict:
    """Load a YAML config file."""
    import yaml

    project_root = Path(__file__).resolve().parent.parent
    default_path = project_root / "configs" / "default.yaml"

    cfg = {}
    if default_path.exists():
        with open(default_path) as f:
            cfg = yaml.safe_load(f) or {}

    if config_path.resolve() != default_path.resolve():
        with open(config_path) as f:
            overlay = yaml.safe_load(f) or {}
        _deep_merge(cfg, overlay)

    return cfg


def _deep_merge(base: dict, overlay: dict) -> None:
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


@app.command()
def main(
    model_path: Path = typer.Option(
        ..., "--model-path", "-m",
        help="Path to the model checkpoint directory.",
    ),
    config: Path = typer.Option(
        "configs/default.yaml", "--config", "-c",
        help="Path to the experiment config YAML.",
    ),
    generation: int = typer.Option(
        0, "--generation", "-g",
        help="Generation index (for labelling results).",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output JSON file for the metrics. Defaults to <model_path>/metrics.json.",
    ),
) -> None:
    """Load a model checkpoint and compute all quality metrics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _load_config(config)

    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model from %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare evaluation texts.
    eval_samples = cfg.get("measurement", {}).get("eval_samples", 100)
    eval_texts = _generate_eval_texts(model, tokenizer, eval_samples, cfg)

    results: dict = {"generation": generation}

    # Entropy.
    try:
        from src.measurement.entropy import EntropyMeasurer

        measurer = EntropyMeasurer(batch_size=8)
        seq_result = measurer.sequence_entropy(eval_texts)
        results["entropy"] = seq_result.unigram
        results["bigram_entropy"] = seq_result.bigram
        results["trigram_entropy"] = seq_result.trigram
        logger.info("Entropy: unigram=%.4f", seq_result.unigram)
    except Exception as e:
        logger.warning("Entropy measurement failed: %s", e)

    # Diversity.
    try:
        from src.measurement.diversity import DiversityMeasurer

        div_measurer = DiversityMeasurer()
        div_result = div_measurer.measure(eval_texts)
        results["distinct_1"] = div_result.distinct_1
        results["distinct_2"] = div_result.distinct_2
        results["distinct_3"] = div_result.distinct_3
        results["distinct_4"] = div_result.distinct_4
        results["self_bleu"] = div_result.self_bleu
        results["vocabulary_usage"] = div_result.vocabulary_usage
        logger.info("Diversity: distinct-2=%.4f", div_result.distinct_2)
    except Exception as e:
        logger.warning("Diversity measurement failed: %s", e)

    # KL divergence (requires a reference distribution).
    try:
        from src.measurement.kl_divergence import KLDivergenceEstimator

        vocab_size = tokenizer.vocab_size
        ref_dist = np.ones(vocab_size) / vocab_size  # uniform as fallback
        kl_estimator = KLDivergenceEstimator(ref_dist, tokenizer)
        kl_result = kl_estimator.estimate_from_generated_text(eval_texts)
        results["kl_divergence"] = kl_result.kl_p_q
        results["js_divergence"] = kl_result.js_divergence
        logger.info("KL(P||Q)=%.4f, JS=%.4f", kl_result.kl_p_q, kl_result.js_divergence)
    except Exception as e:
        logger.warning("KL measurement failed: %s", e)

    # Save results.
    output_path = output or (model_path / "metrics.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Metrics saved to %s", output_path)


def _generate_eval_texts(
    model, tokenizer, n: int, cfg: dict,
) -> list[str]:
    """Generate evaluation texts from the model."""
    import torch

    from src.data.synthetic import GenerationConfig, SyntheticGenerator

    gen_cfg = GenerationConfig(
        num_samples=n,
        max_new_tokens=cfg.get("synthetic_generation", {}).get("max_new_tokens", 64),
        temperature=cfg.get("synthetic_generation", {}).get("temperature", 1.0),
        top_p=cfg.get("synthetic_generation", {}).get("top_p", 0.95),
        batch_size=cfg.get("synthetic_generation", {}).get("batch_size", 8),
    )

    device = next(model.parameters()).device
    model.eval()
    generator = SyntheticGenerator(model, tokenizer, gen_cfg)
    corpus = generator.generate_corpus(n=n, seed=42)
    return corpus["text"]


if __name__ == "__main__":
    app()
