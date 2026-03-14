"""Quality filters applied to the reserve corpus.

Provides deduplication (via cosine similarity on embeddings), a heuristic
language filter, and a length filter.  :func:`apply_quality_filters` chains
them in sequence and logs per-filter removal counts.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from src.data.common_crawl import Document

logger = logging.getLogger(__name__)

# Rough list of common English words used by the language heuristic.
_COMMON_ENGLISH_WORDS = frozenset(
    [
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
        "is",
        "are",
        "was",
        "were",
        "been",
        "has",
        "had",
        "did",
        "does",
    ]
)

_WORD_RE = re.compile(r"[a-zA-Z]+")


# ---------------------------------------------------------------------------
# Individual filters
# ---------------------------------------------------------------------------


def deduplicate(
    documents: list[Document],
    embeddings: np.ndarray,
    threshold: float = 0.95,
) -> list[Document]:
    """Remove near-duplicate documents based on cosine similarity.

    For every pair whose cosine similarity meets or exceeds *threshold*, the
    document with the **later** timestamp is removed (the earliest-timestamped
    version is kept).

    Parameters
    ----------
    documents:
        Documents aligned row-wise with *embeddings*.
    embeddings:
        Unit-normalised embedding matrix of shape ``(n, d)``.
    threshold:
        Cosine-similarity threshold above which two documents are considered
        near-duplicates.

    Returns
    -------
    list[Document]
        De-duplicated document list.
    """
    if len(documents) != embeddings.shape[0]:
        raise ValueError(
            f"documents ({len(documents)}) and embeddings "
            f"({embeddings.shape[0]}) must have the same length"
        )

    n = len(documents)
    if n <= 1:
        return list(documents)

    # Compute full similarity matrix (dot product of unit vectors).
    sim_matrix = embeddings @ embeddings.T

    # Track which indices to remove.
    to_remove: set[int] = set()

    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            if sim_matrix[i, j] >= threshold:
                # Remove the document with the later timestamp.
                if documents[j].timestamp <= documents[i].timestamp:
                    to_remove.add(i)
                    break  # i is removed; stop comparing it
                else:
                    to_remove.add(j)

    kept = [doc for idx, doc in enumerate(documents) if idx not in to_remove]
    logger.info(
        "Deduplication: removed %d near-duplicates (threshold=%.2f), %d remain",
        len(to_remove),
        threshold,
        len(kept),
    )
    return kept


def language_filter(
    documents: list[Document],
    target_lang: str = "en",
) -> list[Document]:
    """Remove documents that are unlikely to be in the target language.

    Uses a lightweight heuristic combining:
    1. Fraction of ASCII characters (>= 0.80 required).
    2. Presence of common English words (>= 5% of words must be in the
       common-word list).

    Only ``target_lang="en"`` is currently implemented.

    Parameters
    ----------
    documents:
        Input documents.
    target_lang:
        Target language code (only ``"en"`` supported).

    Returns
    -------
    list[Document]
        Documents that pass the language check.
    """
    if target_lang != "en":
        logger.warning(
            "language_filter only supports 'en'; returning all documents "
            "for target_lang=%r",
            target_lang,
        )
        return list(documents)

    kept: list[Document] = []
    for doc in documents:
        text = doc.text
        if not text:
            continue

        # Criterion 1: ASCII fraction.
        ascii_chars = sum(1 for ch in text if ord(ch) < 128)
        ascii_frac = ascii_chars / len(text)
        if ascii_frac < 0.80:
            continue

        # Criterion 2: common English words.
        words = [w.lower() for w in _WORD_RE.findall(text)]
        if not words:
            continue
        english_count = sum(1 for w in words if w in _COMMON_ENGLISH_WORDS)
        english_frac = english_count / len(words)
        if english_frac < 0.05:
            continue

        kept.append(doc)

    removed = len(documents) - len(kept)
    logger.info(
        "Language filter (%s): removed %d documents, %d remain",
        target_lang,
        removed,
        len(kept),
    )
    return kept


def length_filter(
    documents: list[Document],
    min_chars: int = 500,
    max_chars: int = 500_000,
) -> list[Document]:
    """Remove documents that are too short or too long.

    Parameters
    ----------
    documents:
        Input documents.
    min_chars:
        Minimum character count (inclusive).
    max_chars:
        Maximum character count (inclusive).

    Returns
    -------
    list[Document]
        Documents whose text length falls within ``[min_chars, max_chars]``.
    """
    kept = [doc for doc in documents if min_chars <= len(doc.text) <= max_chars]
    removed = len(documents) - len(kept)
    logger.info(
        "Length filter [%d, %d]: removed %d documents, %d remain",
        min_chars,
        max_chars,
        removed,
        len(kept),
    )
    return kept


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------


def apply_quality_filters(
    documents: list[Document],
    embeddings: np.ndarray,
    config: dict,
) -> list[Document]:
    """Apply all quality filters in sequence.

    The *config* dictionary may contain the following keys (all optional):

    - ``dedup_threshold`` (float): cosine-similarity threshold for
      deduplication (default 0.95).
    - ``target_lang`` (str): target language code (default ``"en"``).
    - ``min_chars`` (int): minimum document length (default 500).
    - ``max_chars`` (int): maximum document length (default 500000).

    Parameters
    ----------
    documents:
        Input documents.
    embeddings:
        Unit-normalised embedding matrix aligned with *documents*.
    config:
        Quality-filter configuration dictionary.

    Returns
    -------
    list[Document]
        Documents surviving all quality filters.
    """
    initial_count = len(documents)
    logger.info("Applying quality filters to %d documents", initial_count)

    # 1. Length filter (applied first so deduplication operates on fewer docs).
    min_chars = config.get("min_chars", 500)
    max_chars = config.get("max_chars", 500_000)
    docs_after_length = length_filter(documents, min_chars=min_chars, max_chars=max_chars)

    # Build a boolean mask so we can slice embeddings in sync.
    length_mask = [
        min_chars <= len(doc.text) <= max_chars for doc in documents
    ]
    embeddings = embeddings[length_mask]

    # 2. Language filter.
    target_lang = config.get("target_lang", "en")
    docs_after_lang = language_filter(docs_after_length, target_lang=target_lang)

    # Sync embeddings after language filter.
    lang_kept_ids = {id(doc) for doc in docs_after_lang}
    lang_mask = [id(doc) in lang_kept_ids for doc in docs_after_length]
    embeddings = embeddings[lang_mask]

    # 3. Deduplication.
    dedup_threshold = config.get("dedup_threshold", 0.95)
    docs_final = deduplicate(
        docs_after_lang, embeddings, threshold=dedup_threshold
    )

    logger.info(
        "Quality filters complete: %d -> %d documents (removed %d)",
        initial_count,
        len(docs_final),
        initial_count - len(docs_final),
    )
    return docs_final
