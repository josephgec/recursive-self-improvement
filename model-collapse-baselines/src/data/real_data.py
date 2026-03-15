"""Real data loading and caching from HuggingFace datasets."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class CorpusConfig:
    """Configuration for the real-data corpus."""

    dataset: str = "openwebtext"
    split: str = "train"
    max_documents: int = 100_000
    max_length: int = 512
    seed: int = 42


class RealDataLoader:
    """Loads, tokenizes, and caches a reference corpus from HuggingFace.

    The corpus is tokenized once on first access and reused afterwards.
    """

    def __init__(self, config: CorpusConfig, tokenizer) -> None:
        self._config = config
        self._tokenizer = tokenizer
        self._corpus: Dataset | None = None
        self._token_distribution: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_corpus(self) -> Dataset:
        """Return the full tokenized corpus, loading & caching on first call."""
        if self._corpus is None:
            self._corpus = self._load_and_tokenize()
        return self._corpus

    def sample(self, n: int, seed: int | None = None) -> Dataset:
        """Return a random sample of *n* documents from the corpus.

        Args:
            n: Number of documents to sample.
            seed: Random seed for reproducibility.  Falls back to config seed.
        """
        corpus = self.get_corpus()
        seed = seed if seed is not None else self._config.seed
        n = min(n, len(corpus))
        return corpus.shuffle(seed=seed).select(range(n))

    def get_token_distribution(self) -> np.ndarray:
        """Empirical unigram distribution over the corpus with Laplace smoothing.

        Returns:
            1-D array of length ``vocab_size`` summing to 1.
        """
        if self._token_distribution is not None:
            return self._token_distribution

        corpus = self.get_corpus()
        vocab_size = len(self._tokenizer)
        counts = np.zeros(vocab_size, dtype=np.float64)

        for row in corpus:
            ids = row["input_ids"]
            for token_id in ids:
                if 0 <= token_id < vocab_size:
                    counts[token_id] += 1

        # Laplace smoothing
        counts += 1.0
        self._token_distribution = counts / counts.sum()
        return self._token_distribution

    def get_reference_embeddings(self, encoder) -> np.ndarray:
        """Embed a fixed reference sample using *encoder*.

        Args:
            encoder: A ``sentence_transformers.SentenceTransformer`` or any
                object whose ``encode(list[str])`` method returns an ndarray.

        Returns:
            2-D array of shape ``(n, embedding_dim)``.
        """
        from datasets import Dataset as _  # noqa: F401 – ensure lib available

        sample = self.sample(
            n=min(1000, len(self.get_corpus())),
            seed=self._config.seed,
        )
        texts = sample["text"]
        embeddings: np.ndarray = encoder.encode(texts, show_progress_bar=False)
        return np.asarray(embeddings)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_and_tokenize(self) -> Dataset:
        """Download / stream the dataset, truncate, tokenize, and return."""
        from datasets import load_dataset

        cfg = self._config
        logger.info(
            "Loading dataset=%s split=%s (max_documents=%d)",
            cfg.dataset,
            cfg.split,
            cfg.max_documents,
        )

        # Load only the slice we need.
        ds = load_dataset(
            cfg.dataset,
            split=f"{cfg.split}[:{cfg.max_documents}]",
            trust_remote_code=True,
        )

        # Determine the text column name (handle common variations).
        text_col = "text"
        if text_col not in ds.column_names:
            for candidate in ("content", "document", "passage"):
                if candidate in ds.column_names:
                    text_col = candidate
                    break
            else:
                # Fall back to first string column.
                text_col = ds.column_names[0]

        # Keep only the text column, rename to "text" if needed.
        if text_col != "text":
            ds = ds.rename_column(text_col, "text")
        cols_to_remove = [c for c in ds.column_names if c != "text"]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)

        # Tokenize (truncated to max_length).
        def _tokenize(examples):
            return self._tokenizer(
                examples["text"],
                truncation=True,
                max_length=cfg.max_length,
                padding=False,
                return_attention_mask=False,
            )

        ds = ds.map(_tokenize, batched=True, desc="Tokenizing")
        logger.info("Corpus ready: %d documents", len(ds))
        return ds
