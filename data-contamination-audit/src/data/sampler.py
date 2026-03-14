"""Temporal corpus builder.

Assembles a balanced corpus by sampling *n_per_bin* documents from each source
for every requested time bin.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import yaml

from src.data.common_crawl import Document, _load_crawl_config
from src.data.timestamper import assign_time_bin

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CC_CRAWLS_PATH = _PROJECT_ROOT / "configs" / "cc_crawls.yaml"


def _find_crawl_for_bin(bin_label: str, bin_size: str = "year") -> str | None:
    """Find the best Common Crawl crawl ID for a given time bin.

    Returns the crawl whose date falls within the bin, or ``None`` if no
    match is found.
    """
    from datetime import datetime

    crawl_config = _load_crawl_config()

    for crawl_id, date_str in crawl_config.items():
        crawl_date = datetime.fromisoformat(date_str)

        # Build a synthetic Document just to compute its bin label.
        dummy = Document(
            doc_id="tmp",
            text="",
            source="common_crawl",
            timestamp=crawl_date,
            url=None,
        )
        crawl_bin = assign_time_bin(dummy, bin_size)
        if crawl_bin == bin_label:
            return crawl_id

    return None


def build_temporal_corpus(
    sources: list[str],
    n_per_bin: int,
    bins: list[str],
    seed: int,
    *,
    bin_size: str = "year",
    wikipedia_dump_dir: Path | None = None,
    cc_languages: list[str] | None = None,
) -> dict[str, list[Document]]:
    """Build a temporally balanced corpus.

    For each time bin, samples *n_per_bin* documents from **each** requested
    source, merges them, and returns the mapping ``bin_label -> documents``.

    Parameters
    ----------
    sources:
        Data source names.  Supported: ``"wikipedia"``, ``"common_crawl"``.
    n_per_bin:
        Number of documents to sample per source per bin.
    bins:
        List of bin labels (e.g. ``["2019", "2021", "2023"]``).
    seed:
        Master random seed.
    bin_size:
        Temporal bin granularity (``"year"``, ``"half-year"``, ``"quarter"``).
    wikipedia_dump_dir:
        Directory containing pre-downloaded Wikipedia dumps.  Dumps are
        expected at ``{dump_dir}/{YYYYMMDD}/enwiki-{YYYYMMDD}-pages-articles.xml.bz2``.
    cc_languages:
        Optional language filter passed to :func:`sample_cc_warc`.

    Returns
    -------
    dict[str, list[Document]]
        Mapping from bin label to list of sampled documents.
    """
    from src.data.common_crawl import sample_cc_warc
    from src.data.wikipedia import sample_wikipedia

    rng = random.Random(seed)
    corpus: dict[str, list[Document]] = {}

    for bin_label in bins:
        logger.info("Building corpus for bin %s …", bin_label)
        bin_docs: list[Document] = []

        for source in sources:
            # Derive a per-source, per-bin seed for reproducibility.
            source_seed = rng.randint(0, 2**31 - 1)

            if source == "common_crawl":
                crawl_id = _find_crawl_for_bin(bin_label, bin_size)
                if crawl_id is None:
                    logger.warning(
                        "No Common Crawl crawl found for bin %s — skipping",
                        bin_label,
                    )
                    continue

                docs = sample_cc_warc(
                    crawl_id=crawl_id,
                    n=n_per_bin,
                    seed=source_seed,
                    languages=cc_languages,
                )
                bin_docs.extend(docs)

            elif source == "wikipedia":
                # Try to locate a Wikipedia dump for this bin.
                dump_path = _resolve_wikipedia_dump(bin_label, wikipedia_dump_dir)
                if dump_path is None:
                    logger.warning(
                        "No Wikipedia dump found for bin %s — skipping",
                        bin_label,
                    )
                    continue

                docs = sample_wikipedia(
                    dump_path=dump_path,
                    n=n_per_bin,
                    seed=source_seed,
                )
                bin_docs.extend(docs)

            else:
                logger.warning("Unknown source %r — skipping", source)

        corpus[bin_label] = bin_docs
        logger.info(
            "Bin %s: %d documents (%s)",
            bin_label,
            len(bin_docs),
            ", ".join(f"{s}: {sum(1 for d in bin_docs if d.source == s)}" for s in sources),
        )

    total = sum(len(v) for v in corpus.values())
    logger.info("Temporal corpus built: %d bins, %d total documents", len(corpus), total)
    return corpus


def _resolve_wikipedia_dump(
    bin_label: str,
    dump_dir: Path | None,
) -> Path | None:
    """Try to find a Wikipedia dump file matching the given bin label.

    Looks for dumps named ``enwiki-{YYYYMMDD}-pages-articles.xml.bz2`` under
    *dump_dir*, selecting the one whose date falls within the target bin.
    """
    if dump_dir is None:
        dump_dir = _PROJECT_ROOT / "data" / "raw" / "wikipedia"

    if not dump_dir.exists():
        return None

    # Scan subdirectories named by date.
    for date_dir in sorted(dump_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name
        # Look for the dump file.
        candidates = list(date_dir.glob("enwiki-*-pages-articles.xml.bz2"))
        if not candidates:
            candidates = list(date_dir.glob("enwiki-*-pages-articles.xml"))
        if not candidates:
            continue

        dump_path = candidates[0]

        # Check if this dump's year matches the bin label.
        # Simple heuristic: the bin_label starts with the dump year.
        try:
            dump_year = date_str[:4]
        except (IndexError, ValueError):
            continue

        if bin_label.startswith(dump_year):
            return dump_path

    return None
