"""Wikipedia data ingestion.

Downloads Wikimedia database dumps, parses XML with *mwparserfromhell*,
and provides stratified random sampling with stub / disambiguation / redirect
filtering.
"""

from __future__ import annotations

import bz2
import hashlib
import logging
import random
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import mwparserfromhell
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from src.data.common_crawl import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_DUMP_MIRROR = "https://dumps.wikimedia.org"

_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0
_REQUEST_TIMEOUT = 120

# MediaWiki XML namespace
_MW_NS = "http://www.mediawiki.org/xml/export-0.10/"
_MW_NS_ALT = "http://www.mediawiki.org/xml/export-0.11/"

# Patterns for filtering
_DISAMBIGUATION_RE = re.compile(
    r"\{\{[Dd]isambig(uation)?(\|[^}]*)?\}\}|\{\{[Hh]ndis(\|[^}]*)?\}\}"
)
_REDIRECT_RE = re.compile(r"^#REDIRECT\s*\[\[", re.IGNORECASE)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _request_with_retries(
    method: str,
    url: str,
    *,
    max_retries: int = _MAX_RETRIES,
    timeout: int = _REQUEST_TIMEOUT,
    **kwargs: Any,
) -> requests.Response:
    """Issue an HTTP request with exponential-backoff retries."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as exc:
            last_exc = exc
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "Request to %s failed (attempt %d/%d): %s — retrying in %.1fs",
                url,
                attempt,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"All {max_retries} attempts to {method.upper()} {url} failed"
    ) from last_exc


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_wikipedia_dump(date: str, output_dir: Path | None = None) -> Path:
    """Download a Wikipedia articles dump for *date* (``YYYYMMDD``).

    Downloads the ``pages-articles.xml.bz2`` file from Wikimedia mirrors.
    If the file already exists locally it is not re-downloaded.

    Parameters
    ----------
    date:
        Dump date in ``YYYYMMDD`` format, e.g. ``"20230601"``.
    output_dir:
        Where to save the dump.  Defaults to ``data/raw/wikipedia/{date}/``.

    Returns
    -------
    Path
        Path to the downloaded ``.xml.bz2`` file.
    """
    if output_dir is None:
        output_dir = _RAW_DATA_DIR / "wikipedia" / date

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"enwiki-{date}-pages-articles.xml.bz2"
    dest = output_dir / filename

    if dest.exists():
        logger.info("Dump already cached at %s", dest)
        return dest

    url = f"{_DUMP_MIRROR}/enwiki/{date}/{filename}"
    logger.info("Downloading Wikipedia dump from %s …", url)

    # Stream download to avoid loading entire file into memory.
    try:
        resp = requests.get(url, stream=True, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download Wikipedia dump: {exc}") from exc

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MiB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                if downloaded % (50 << 20) < (1 << 20):  # log every ~50 MiB
                    logger.info("  … %.1f%% downloaded", pct)

    logger.info("Saved Wikipedia dump to %s (%d bytes)", dest, dest.stat().st_size)
    return dest


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _detect_namespace(dump_path: Path) -> str:
    """Peek at the first few KB to detect the MediaWiki XML namespace."""
    open_fn = bz2.open if dump_path.suffix == ".bz2" else open
    with open_fn(dump_path, "rt", encoding="utf-8", errors="replace") as f:  # type: ignore[call-overload]
        header = f.read(4096)
    if _MW_NS_ALT in header:
        return _MW_NS_ALT
    return _MW_NS


def _strip_wikitext(wikitext: str) -> str:
    """Convert MediaWiki markup to plain text via *mwparserfromhell*."""
    parsed = mwparserfromhell.parse(wikitext)
    return parsed.strip_code().strip()


def _is_disambiguation(wikitext: str) -> bool:
    return bool(_DISAMBIGUATION_RE.search(wikitext))


def _is_redirect(wikitext: str) -> bool:
    return bool(_REDIRECT_RE.match(wikitext))


def parse_wikipedia_dump(dump_path: Path) -> Iterator[Document]:
    """Stream-parse a Wikipedia XML dump and yield :class:`Document` objects.

    Skips redirect pages.  Extracts plain text from wikitext via
    *mwparserfromhell*.

    Parameters
    ----------
    dump_path:
        Path to the ``.xml.bz2`` (or plain ``.xml``) dump file.

    Yields
    ------
    Document
    """
    ns = _detect_namespace(dump_path)
    open_fn = bz2.open if dump_path.suffix == ".bz2" else open

    count = 0
    logger.info("Parsing Wikipedia dump %s …", dump_path)

    # Use iterparse for memory efficiency.
    with open_fn(dump_path, "rb") as raw:  # type: ignore[call-overload]
        context = ET.iterparse(raw, events=("end",))
        for event, elem in context:
            tag = elem.tag.replace(f"{{{ns}}}", "")

            if tag != "page":
                continue

            # Extract fields from the <page> element.
            title_elem = elem.find(f"{{{ns}}}title")
            redirect_elem = elem.find(f"{{{ns}}}redirect")
            revision_elem = elem.find(f"{{{ns}}}revision")

            title = title_elem.text if title_elem is not None else ""

            # Skip redirects at the XML level.
            if redirect_elem is not None:
                elem.clear()
                continue

            if revision_elem is None:
                elem.clear()
                continue

            text_elem = revision_elem.find(f"{{{ns}}}text")
            ts_elem = revision_elem.find(f"{{{ns}}}timestamp")
            page_id_elem = elem.find(f"{{{ns}}}id")

            wikitext = text_elem.text if text_elem is not None else ""
            if not wikitext:
                elem.clear()
                continue

            # Skip wikitext-level redirects.
            if _is_redirect(wikitext):
                elem.clear()
                continue

            plain_text = _strip_wikitext(wikitext)

            # Parse timestamp.
            if ts_elem is not None and ts_elem.text:
                try:
                    timestamp = datetime.fromisoformat(
                        ts_elem.text.replace("Z", "+00:00")
                    )
                except ValueError:
                    timestamp = datetime(2000, 1, 1)
            else:
                timestamp = datetime(2000, 1, 1)

            page_id = page_id_elem.text if page_id_elem is not None else ""
            doc_id_hash = hashlib.sha256(
                f"wikipedia:{page_id}:{title}".encode()
            ).hexdigest()[:16]

            count += 1
            if count % 10_000 == 0:
                logger.info("  … parsed %d articles", count)

            yield Document(
                doc_id=f"wiki-{doc_id_hash}",
                text=plain_text,
                source="wikipedia",
                timestamp=timestamp,
                url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                metadata={
                    "title": title,
                    "page_id": page_id,
                    "is_disambiguation": _is_disambiguation(wikitext),
                },
            )

            # Free memory.
            elem.clear()

    logger.info("Finished parsing %d articles from %s", count, dump_path)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_wikipedia(
    dump_path: Path,
    n: int,
    seed: int,
) -> list[Document]:
    """Draw a stratified random sample from a Wikipedia dump.

    Filters out:
    - Stubs (text shorter than 500 characters)
    - Disambiguation pages
    - Redirect pages (already filtered during parsing)

    Results are cached as Parquet in ``data/processed/wikipedia/{date}/``.

    Parameters
    ----------
    dump_path:
        Path to the ``.xml.bz2`` dump file.
    n:
        Number of documents to sample.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list[Document]
    """
    # Derive date from filename (e.g. enwiki-20230601-pages-articles.xml.bz2).
    stem = dump_path.stem
    if stem.endswith(".xml"):
        stem = stem[: -len(".xml")]
    parts = stem.split("-")
    dump_date = parts[1] if len(parts) >= 2 else "unknown"

    processed_dir = _PROCESSED_DIR / "wikipedia" / dump_date
    processed_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = processed_dir / "articles.parquet"

    # Build the article pool — either from cache or by parsing.
    if parquet_path.exists():
        logger.info("Loading cached parsed articles from %s", parquet_path)
        df = pd.read_parquet(parquet_path)
        articles = _dataframe_to_documents(df)
    else:
        logger.info("Parsing dump and caching to %s …", parquet_path)
        articles = list(parse_wikipedia_dump(dump_path))
        _save_documents_parquet(articles, parquet_path)

    # Filter.
    eligible = [
        doc
        for doc in articles
        if len(doc.text) >= 500
        and not doc.metadata.get("is_disambiguation", False)
    ]

    logger.info(
        "Eligible articles after filtering: %d / %d total",
        len(eligible),
        len(articles),
    )

    if len(eligible) <= n:
        logger.warning(
            "Requested %d samples but only %d eligible — returning all",
            n,
            len(eligible),
        )
        return eligible

    rng = random.Random(seed)
    sampled = rng.sample(eligible, k=n)
    logger.info("Sampled %d Wikipedia articles (seed=%d)", len(sampled), seed)
    return sampled


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------


def _save_documents_parquet(docs: list[Document], path: Path) -> None:
    """Serialize a list of Documents to a Parquet file."""
    records = [
        {
            "doc_id": d.doc_id,
            "text": d.text,
            "source": d.source,
            "timestamp": d.timestamp.isoformat(),
            "url": d.url,
            "title": d.metadata.get("title", ""),
            "page_id": d.metadata.get("page_id", ""),
            "is_disambiguation": d.metadata.get("is_disambiguation", False),
        }
        for d in docs
    ]
    table = pa.table(pd.DataFrame(records))
    pq.write_table(table, path)
    logger.info("Saved %d documents to %s", len(docs), path)


def _dataframe_to_documents(df: pd.DataFrame) -> list[Document]:
    """Reconstruct Document objects from a cached Parquet dataframe."""
    docs: list[Document] = []
    for _, row in df.iterrows():
        try:
            ts = datetime.fromisoformat(row["timestamp"])
        except (ValueError, TypeError):
            ts = datetime(2000, 1, 1)

        docs.append(
            Document(
                doc_id=row["doc_id"],
                text=row["text"],
                source=row["source"],
                timestamp=ts,
                url=row.get("url"),
                metadata={
                    "title": row.get("title", ""),
                    "page_id": row.get("page_id", ""),
                    "is_disambiguation": row.get("is_disambiguation", False),
                },
            )
        )
    return docs
