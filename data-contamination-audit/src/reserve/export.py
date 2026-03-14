"""Export the clean data reserve to Parquet and a companion summary JSON."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.common_crawl import Document

logger = logging.getLogger(__name__)


def export_reserve(
    documents: list[Document],
    output_dir: Path,
    format: str = "parquet",
    *,
    full_corpus_size: int | None = None,
    threshold: float = 0.90,
) -> None:
    """Write the clean data reserve to disk.

    Creates two files inside *output_dir*:

    1. ``reserve.parquet`` — one row per document with columns:
       ``doc_id``, ``text``, ``source``, ``timestamp``, ``time_bin``,
       ``url``, ``authenticity_score``, ``perplexity_mean``,
       ``watermark_z_score``, ``vocabulary_richness``.
    2. ``summary.json`` — aggregate statistics about the reserve.

    Parameters
    ----------
    documents:
        Reserve documents (should already be filtered and quality-checked).
    output_dir:
        Directory to write output files into (created if necessary).
    format:
        Output format for the data file.  Only ``"parquet"`` is currently
        supported.
    full_corpus_size:
        Total number of documents in the original audited corpus.  Used to
        compute ``alpha_t`` in the summary.  If ``None``, defaults to
        ``len(documents)``.
    threshold:
        The authenticity threshold that was used during filtering.
    """
    if format != "parquet":
        raise ValueError(f"Unsupported export format: {format!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build the Parquet dataframe
    # ------------------------------------------------------------------
    rows: list[dict] = []
    for doc in documents:
        meta = doc.metadata
        rows.append(
            {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "source": doc.source,
                "timestamp": doc.timestamp,
                "time_bin": meta.get("time_bin", ""),
                "url": doc.url or "",
                "authenticity_score": meta.get("authenticity_score", None),
                "perplexity_mean": meta.get("perplexity_mean", None),
                "watermark_z_score": meta.get("watermark_z_score", None),
                "vocabulary_richness": meta.get("vocabulary_richness", None),
            }
        )

    df = pd.DataFrame(rows)
    parquet_path = output_dir / "reserve.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Wrote %d documents to %s", len(df), parquet_path)

    # ------------------------------------------------------------------
    # Build the summary JSON
    # ------------------------------------------------------------------
    reserve_size = len(documents)
    corpus_size = full_corpus_size if full_corpus_size is not None else reserve_size

    alpha_t = reserve_size / corpus_size if corpus_size > 0 else 0.0

    authenticity_scores = [
        doc.metadata.get("authenticity_score", 0.0) for doc in documents
    ]
    mean_auth = (
        sum(authenticity_scores) / len(authenticity_scores)
        if authenticity_scores
        else 0.0
    )

    # Temporal distribution: count documents per time_bin.
    time_bins = [doc.metadata.get("time_bin", "unknown") for doc in documents]
    temporal_distribution = dict(Counter(time_bins))

    # Source distribution.
    sources = [doc.source for doc in documents]
    source_distribution = dict(Counter(sources))

    summary = {
        "total_documents_audited": corpus_size,
        "reserve_size": reserve_size,
        "alpha_t": alpha_t,
        "threshold": threshold,
        "mean_authenticity_score": mean_auth,
        "temporal_distribution": temporal_distribution,
        "source_distribution": source_distribution,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote summary to %s", summary_path)
