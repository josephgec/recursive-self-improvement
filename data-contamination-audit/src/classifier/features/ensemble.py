"""Feature assembly: combine perplexity, watermark, and stylometric features.

Provides helpers to extract all features for a single document and to
build a complete feature matrix for a corpus.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.common_crawl import Document

from .perplexity import PerplexityScorer
from .stylometry import compute_stylometric_features
from .watermark import WatermarkDetector

logger = logging.getLogger(__name__)


def extract_all_features(
    doc: Document,
    perplexity_scorer: PerplexityScorer | None,
    watermark_detector: WatermarkDetector,
    tokenizer: Any,
) -> dict[str, float]:
    """Extract all classifier features for a single document.

    Parameters
    ----------
    doc:
        The document to featurise.
    perplexity_scorer:
        Initialised :class:`PerplexityScorer`, or ``None`` to skip
        perplexity scoring and use fallback defaults instead.
    watermark_detector:
        Initialised :class:`WatermarkDetector`.
    tokenizer:
        A HuggingFace-compatible tokenizer with an ``encode`` method
        (used by the watermark detector to obtain token IDs).

    Returns
    -------
    dict[str, float]
        Feature name -> value mapping.  Contains perplexity features,
        watermark features, and stylometric features.
    """
    features: dict[str, float] = {}

    # -- Perplexity features -----------------------------------------------
    if perplexity_scorer is not None:
        ppl_feats = perplexity_scorer.compute_features(doc.text)
    else:
        ppl_feats = dict(PerplexityScorer._FALLBACK_DEFAULTS)
    features.update(ppl_feats)

    # -- Watermark features ------------------------------------------------
    token_ids: list[int] = tokenizer.encode(doc.text)
    wm_feats = watermark_detector.score(token_ids)
    features.update(wm_feats)

    # -- Stylometric features ----------------------------------------------
    style_feats = compute_stylometric_features(doc.text)
    features.update(style_feats)

    return features


def build_feature_matrix(
    documents: list[Document],
    perplexity_scorer: PerplexityScorer | None,
    watermark_detector: WatermarkDetector,
    tokenizer: Any,
    cache_dir: str | Path | None = None,
    cache_tag: str = "features",
) -> pd.DataFrame:
    """Build a feature matrix for a list of documents.

    Parameters
    ----------
    documents:
        Documents to featurise.
    perplexity_scorer:
        Initialised :class:`PerplexityScorer`, or ``None`` to skip
        perplexity scoring and use fallback defaults.
    watermark_detector:
        Initialised :class:`WatermarkDetector`.
    tokenizer:
        HuggingFace-compatible tokenizer.
    cache_dir:
        If provided, the resulting DataFrame is cached to a Parquet file
        in this directory.  On subsequent calls with the same corpus hash,
        the cached version is loaded directly.
    cache_tag:
        A label included in the cache filename.

    Returns
    -------
    pd.DataFrame
        Rows = documents, columns = feature names, plus ``doc_id``.
    """
    # -- Check cache -------------------------------------------------------
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        corpus_hash = _corpus_hash(documents)
        parquet_file = cache_path / f"{cache_tag}_{corpus_hash}.parquet"
        if parquet_file.exists():
            logger.info("Loading cached feature matrix from %s", parquet_file)
            return pd.read_parquet(parquet_file)

    # -- Extract features --------------------------------------------------
    rows: list[dict[str, Any]] = []

    try:
        from rich.progress import Progress
        use_rich = True
    except ImportError:
        use_rich = False

    if use_rich:
        with Progress() as progress:
            task = progress.add_task(
                "Extracting features", total=len(documents)
            )
            for doc in documents:
                feats = extract_all_features(
                    doc, perplexity_scorer, watermark_detector, tokenizer
                )
                feats["doc_id"] = doc.doc_id
                rows.append(feats)
                progress.advance(task)
    else:
        for i, doc in enumerate(documents):
            feats = extract_all_features(
                doc, perplexity_scorer, watermark_detector, tokenizer
            )
            feats["doc_id"] = doc.doc_id
            rows.append(feats)
            if (i + 1) % 50 == 0 or (i + 1) == len(documents):
                logger.info(
                    "Extracted features for %d / %d documents",
                    i + 1,
                    len(documents),
                )

    df = pd.DataFrame(rows)

    # Ensure doc_id is the first column.
    cols = ["doc_id"] + [c for c in df.columns if c != "doc_id"]
    df = df[cols]

    # -- Persist cache -----------------------------------------------------
    if cache_path is not None:
        logger.info("Caching feature matrix to %s", parquet_file)
        df.to_parquet(parquet_file, index=False)  # type: ignore[union-attr]

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _corpus_hash(documents: list[Document]) -> str:
    """Compute a deterministic hash over document IDs for cache keying."""
    h = hashlib.sha256()
    for doc in sorted(documents, key=lambda d: d.doc_id):
        h.update(doc.doc_id.encode("utf-8"))
    return h.hexdigest()[:12]
