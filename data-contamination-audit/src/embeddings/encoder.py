"""Transformer-based document encoder using sentence-transformers.

Encodes :class:`Document` instances into dense embedding vectors suitable for
cosine-similarity comparison, clustering, and downstream classification.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.data.common_crawl import Document

logger = logging.getLogger(__name__)


class DocumentEncoder:
    """Encode documents into unit-normalised dense embeddings.

    Parameters
    ----------
    model_name:
        Any model identifier accepted by ``sentence-transformers``.
        Defaults to the compact ``all-MiniLM-L6-v2`` (384-dim).
    device:
        ``"auto"`` (default) detects a CUDA / MPS GPU and falls back to CPU.
        Pass ``"cpu"`` or ``"cuda:0"`` to override.
    batch_size:
        Number of texts per forward pass.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 256,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        if device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device

        logger.info(
            "Loading sentence-transformer model %r on %s", model_name, self._device
        )
        self._model = SentenceTransformer(model_name, device=self._device)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate(text: str, max_tokens: int = 512) -> str:
        """Truncate *text* to approximately *max_tokens* word-pieces.

        A cheap whitespace-split proxy is used instead of running the full
        tokeniser, which is acceptable because sentence-transformers itself
        truncates at the model's max-sequence-length anyway.  The goal here
        is to keep memory usage predictable for very long documents.
        """
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])

    @staticmethod
    def _cache_key(model_name: str, doc_ids: list[str]) -> str:
        """Deterministic hash over model name and sorted document IDs."""
        payload = f"{model_name}::" + ",".join(sorted(doc_ids))
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    @staticmethod
    def _normalise(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalise each row to a unit vector."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Guard against zero vectors.
        norms = np.where(norms == 0, 1.0, norms)
        return (embeddings / norms).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode raw strings into unit-normalised embeddings.

        Parameters
        ----------
        texts:
            Arbitrary strings to encode.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(len(texts), embedding_dim)``.
        """
        truncated = [self._truncate(t) for t in texts]
        logger.info(
            "Encoding %d texts (batch_size=%d, device=%s)",
            len(truncated),
            self.batch_size,
            self._device,
        )
        embeddings: np.ndarray = self._model.encode(
            truncated,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return self._normalise(embeddings)

    def encode(
        self,
        documents: list[Document],
        cache_dir: Path | None = None,
    ) -> np.ndarray:
        """Encode :class:`Document` instances, with optional disk caching.

        Parameters
        ----------
        documents:
            Documents whose ``.text`` fields will be embedded.
        cache_dir:
            If provided, embeddings are saved / loaded as ``.npy`` files
            keyed by a hash of ``(model_name, doc_ids)``.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(len(documents), embedding_dim)``.
        """
        if not documents:
            logger.warning("encode() called with an empty document list")
            return np.empty((0, 0), dtype=np.float32)

        doc_ids = [d.doc_id for d in documents]

        # ---- Try cache ------------------------------------------------
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{self._cache_key(self.model_name, doc_ids)}.npy"

            if cache_file.exists():
                logger.info("Cache hit — loading embeddings from %s", cache_file)
                embeddings = np.load(cache_file)
                if embeddings.shape[0] == len(documents):
                    return embeddings
                logger.warning(
                    "Cached array has %d rows but %d documents supplied; "
                    "re-encoding.",
                    embeddings.shape[0],
                    len(documents),
                )

        # ---- Encode ---------------------------------------------------
        texts = [d.text for d in documents]
        embeddings = self.encode_texts(texts)

        # ---- Write cache ---------------------------------------------
        if cache_dir is not None:
            np.save(cache_file, embeddings)  # type: ignore[possibly-undefined]
            logger.info("Saved embeddings to cache: %s", cache_file)

        return embeddings
