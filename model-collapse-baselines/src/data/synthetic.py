"""Synthetic text generation from a language model."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Parameters governing synthetic text generation."""

    num_samples: int = 50_000
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    batch_size: int = 32


class SyntheticGenerator:
    """Generate a synthetic corpus from a causal language model.

    Supports both *unconditional* generation (empty / BOS prompt) and
    *prompt-conditioned* generation when explicit prompts are supplied.
    """

    def __init__(self, model, tokenizer, config: GenerationConfig) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_corpus(
        self,
        n: int | None = None,
        prompts: list[str] | None = None,
        seed: int = 0,
    ) -> "Dataset":
        """Generate *n* synthetic documents and return them as a Dataset.

        Args:
            n: Number of documents to generate.  Defaults to
                ``config.num_samples``.
            prompts: Optional list of prompt strings.  If shorter than *n*
                they are cycled; if ``None`` unconditional generation is used.
            seed: Random seed for reproducibility.

        Returns:
            A ``datasets.Dataset`` with a ``"text"`` column.
        """
        import torch
        from datasets import Dataset

        n = n if n is not None else self._config.num_samples

        # Seed for reproducibility.
        torch.manual_seed(seed)

        # Build prompt list.
        if prompts is None or len(prompts) == 0:
            prompt_list = [""] * n
        else:
            # Cycle prompts to reach length n.
            full_cycles = n // len(prompts)
            remainder = n % len(prompts)
            prompt_list = prompts * full_cycles + prompts[:remainder]

        all_texts: list[str] = []
        batch_size = self._config.batch_size

        for start in range(0, n, batch_size):
            batch_prompts = prompt_list[start : start + batch_size]
            batch_texts = self.generate_batch(batch_prompts)
            all_texts.extend(batch_texts)
            if (start // batch_size) % 10 == 0:
                logger.info(
                    "Generated %d / %d documents", len(all_texts), n
                )

        # Trim to exactly n (last batch may have produced extras).
        all_texts = all_texts[:n]
        return Dataset.from_dict({"text": all_texts})

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate one batch of texts from *prompts*.

        Args:
            prompts: List of prompt strings (may be empty strings for
                unconditional generation).

        Returns:
            List of generated text strings (prompt stripped).
        """
        import torch

        device = next(self._model.parameters()).device
        cfg = self._config

        # Tokenize prompts.  For empty/unconditional prompts we feed the
        # BOS token so the model has something to condition on.
        effective_prompts = [
            p if p else (self._tokenizer.bos_token or "") for p in prompts
        ]
        inputs = self._tokenizer(
            effective_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_new_tokens,
        ).to(device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                max_length=None,  # suppress warning when model config has max_length
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the newly-generated tokens (strip the prompt).
        prompt_lengths = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, prompt_lengths:]
        texts = self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Free GPU memory.
        del inputs, outputs, generated_ids
        torch.cuda.empty_cache()

        return texts
