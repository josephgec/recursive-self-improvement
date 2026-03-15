"""Metrics recording and model checkpoint management."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


@dataclass
class GenerationMetrics:
    """Metrics captured for a single generation."""

    generation: int = 0
    alpha: float = 0.0
    train_loss: float | None = None
    perplexity: float | None = None
    kl_divergence: float | None = None
    embedding_variance: float | None = None
    vocab_coverage: float | None = None
    timestamp: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class MetricsRecorder:
    """Append-only metrics store backed by JSON and Parquet files.

    Each call to :meth:`record` persists the metrics to both formats so
    that results survive crashes.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._json_path = self._output_dir / "metrics.json"
        self._parquet_path = self._output_dir / "metrics.parquet"
        self._records: list[dict[str, Any]] = []

        # Load existing records if resuming.
        if self._json_path.exists():
            with open(self._json_path) as f:
                self._records = json.load(f)
            logger.info(
                "Resumed %d existing metric records from %s",
                len(self._records),
                self._json_path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, metrics: GenerationMetrics) -> None:
        """Persist a new metrics snapshot.

        Args:
            metrics: Metrics for one generation.
        """
        if not metrics.timestamp:
            metrics.timestamp = datetime.now(timezone.utc).isoformat()

        row = asdict(metrics)
        self._records.append(row)

        # JSON -- always overwrite for atomicity.
        with open(self._json_path, "w") as f:
            json.dump(self._records, f, indent=2, default=str)

        # Parquet -- best-effort (pandas may not be installed in lean envs).
        self._write_parquet()

        logger.info("Recorded metrics for generation %d", metrics.generation)

    def get_all(self) -> list[dict[str, Any]]:
        """Return all recorded metrics as a list of dicts."""
        return list(self._records)

    def latest_generation(self) -> int:
        """Return the highest recorded generation number, or -1."""
        if not self._records:
            return -1
        return max(r.get("generation", -1) for r in self._records)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_parquet(self) -> None:
        """Best-effort write of metrics to Parquet."""
        try:
            import pandas as pd

            df = pd.DataFrame(self._records)
            df.to_parquet(self._parquet_path, index=False)
        except Exception:
            logger.debug("Parquet write skipped (pandas unavailable or error)")


# ------------------------------------------------------------------
# Checkpoint management
# ------------------------------------------------------------------


class CheckpointManager:
    """Save and load model checkpoints for a lineage experiment.

    Directory layout::

        <root>/
            generation_00/
                model/
                    config.json, model.safetensors, ...
                metadata.json
            generation_01/
                ...
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        generation: int,
        model,
        tokenizer,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save a model checkpoint for *generation*.

        Args:
            generation: Generation index.
            model: HuggingFace model (or peft model).
            tokenizer: Tokenizer to save alongside the model.
            metadata: Optional dict of extra info to persist.

        Returns:
            Path to the checkpoint directory.
        """
        gen_dir = self._generation_dir(generation)
        model_dir = gen_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

        meta = {
            "generation": generation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        with open(gen_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info("Checkpoint saved: %s", gen_dir)
        return gen_dir

    def load(self, generation: int):
        """Load a model and tokenizer from a checkpoint.

        Args:
            generation: Generation index to load.

        Returns:
            Tuple of ``(model, tokenizer)``.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_dir = self._generation_dir(generation) / "model"
        if not model_dir.exists():
            raise FileNotFoundError(
                f"No checkpoint for generation {generation} at {model_dir}"
            )

        # Try loading as a peft model first, fall back to standard.
        adapter_config = model_dir / "adapter_config.json"
        if adapter_config.exists():
            model = self._load_peft_model(model_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir), trust_remote_code=True
            )

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
        logger.info("Checkpoint loaded: generation %d", generation)
        return model, tokenizer

    def latest_generation(self) -> int:
        """Return the highest generation with a saved checkpoint, or -1."""
        best = -1
        for path in self._root.iterdir():
            if path.is_dir() and path.name.startswith("generation_"):
                try:
                    gen = int(path.name.split("_")[1])
                    best = max(best, gen)
                except (IndexError, ValueError):
                    continue
        return best

    def generation_exists(self, generation: int) -> bool:
        """Check whether a checkpoint exists for *generation*."""
        model_dir = self._generation_dir(generation) / "model"
        return model_dir.exists()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generation_dir(self, generation: int) -> Path:
        return self._root / f"generation_{generation:02d}"

    @staticmethod
    def _load_peft_model(model_dir: Path):
        """Load a peft/LoRA model."""
        from peft import AutoPeftModelForCausalLM

        return AutoPeftModelForCausalLM.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
