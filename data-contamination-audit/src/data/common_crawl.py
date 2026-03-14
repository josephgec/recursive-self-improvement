"""Common Crawl data ingestion and the shared Document dataclass.

This module provides:
- The `Document` dataclass used across the entire pipeline.
- Functions for downloading Common Crawl columnar indices and sampling
  WARC records via byte-range HTTP requests to S3.
"""

from __future__ import annotations

import hashlib
import io
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests
import trafilatura
import yaml
from warcio.archiveiterator import ArchiveIterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared Document dataclass
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single text document ingested from any source."""

    doc_id: str              # Unique identifier
    text: str                # Plain text content
    source: str              # "wikipedia" or "common_crawl"
    timestamp: datetime      # Publication or crawl date
    url: str | None          # Source URL
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            # Derive a deterministic ID from content when not supplied.
            h = hashlib.sha256(self.text.encode("utf-8", errors="replace")).hexdigest()[:16]
            self.doc_id = f"{self.source}-{h}"


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CC_CRAWLS_PATH = _PROJECT_ROOT / "configs" / "cc_crawls.yaml"
_RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw"

CC_S3_BASE = "https://data.commoncrawl.org"
CC_INDEX_BASE = f"{CC_S3_BASE}/cc-index/table/cc-main/warc"


def _load_crawl_config() -> dict[str, str]:
    """Load cc_crawls.yaml and return a mapping of crawl_id -> date string."""
    with open(_CC_CRAWLS_PATH) as f:
        data = yaml.safe_load(f)
    return {entry["id"]: entry["date"] for entry in data["crawls"]}


# ---------------------------------------------------------------------------
# Retry / HTTP helpers
# ---------------------------------------------------------------------------

_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0
_REQUEST_TIMEOUT = 60


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
# Public API
# ---------------------------------------------------------------------------


def download_cc_index(crawl_id: str) -> Path:
    """Fetch the columnar index (Parquet) for a Common Crawl crawl.

    The index is stored under ``data/raw/cc-index/{crawl_id}/``.  If already
    downloaded the cached path is returned immediately.

    Parameters
    ----------
    crawl_id:
        Crawl identifier, e.g. ``"CC-MAIN-2023-23"``.

    Returns
    -------
    Path
        Directory containing the downloaded Parquet index files.
    """
    crawl_config = _load_crawl_config()
    if crawl_id not in crawl_config:
        raise ValueError(
            f"Unknown crawl_id {crawl_id!r}. "
            f"Valid crawls: {list(crawl_config.keys())}"
        )

    index_dir = _RAW_DATA_DIR / "cc-index" / crawl_id
    index_dir.mkdir(parents=True, exist_ok=True)

    # The CC columnar index is published as a set of Parquet files whose
    # listing is available via the cdx-api cluster.idx endpoint.
    cluster_idx_url = (
        f"{CC_S3_BASE}/cc-index/collections/{crawl_id}/indexes/cluster.idx"
    )

    cluster_idx_path = index_dir / "cluster.idx"
    if cluster_idx_path.exists():
        logger.info("Columnar index already cached at %s", index_dir)
        return index_dir

    logger.info("Downloading cluster index for %s …", crawl_id)
    resp = _request_with_retries("GET", cluster_idx_url)
    cluster_idx_path.write_bytes(resp.content)
    logger.info("Saved cluster index to %s (%d bytes)", cluster_idx_path, len(resp.content))

    return index_dir


def _fetch_warc_record(warc_filename: str, offset: int, length: int) -> str | None:
    """Download a single WARC record via a byte-range GET and extract text.

    Returns the extracted plain text, or ``None`` on failure.
    """
    url = f"{CC_S3_BASE}/{warc_filename}"
    end_byte = offset + length - 1
    headers = {"Range": f"bytes={offset}-{end_byte}"}

    try:
        resp = _request_with_retries("GET", url, headers=headers)
    except RuntimeError:
        logger.error("Failed to fetch WARC record from %s", url)
        return None

    # Parse the WARC record.
    stream = io.BytesIO(resp.content)
    try:
        for record in ArchiveIterator(stream):
            if record.rec_type == "response":
                html_bytes = record.content_stream().read()
                html = html_bytes.decode("utf-8", errors="replace")
                text = trafilatura.extract(html)
                if text and len(text.strip()) > 0:
                    return text.strip()
    except Exception:
        logger.exception("Error parsing WARC record from %s", url)

    return None


def sample_cc_warc(
    crawl_id: str,
    n: int,
    seed: int,
    languages: list[str] | None = None,
) -> list[Document]:
    """Sample *n* documents from a Common Crawl WARC crawl.

    1. Reads the columnar index (downloading if necessary).
    2. Selects random record pointers from the index, optionally filtering by
       language.
    3. Fetches the corresponding WARC segments via byte-range requests.
    4. Extracts plain text with *trafilatura*.

    Parameters
    ----------
    crawl_id:
        Crawl identifier, e.g. ``"CC-MAIN-2023-23"``.
    n:
        Number of documents to sample.
    seed:
        Random seed for reproducibility.
    languages:
        If given, restrict to records whose ``content_languages`` field
        contains one of these ISO-639-1 codes (e.g. ``["en"]``).

    Returns
    -------
    list[Document]
        Sampled documents (may be fewer than *n* if extraction fails on some
        records).
    """
    crawl_config = _load_crawl_config()
    if crawl_id not in crawl_config:
        raise ValueError(
            f"Unknown crawl_id {crawl_id!r}. "
            f"Valid crawls: {list(crawl_config.keys())}"
        )

    crawl_date = datetime.fromisoformat(crawl_config[crawl_id])
    index_dir = download_cc_index(crawl_id)

    # Attempt to read any parquet files from the index directory.
    parquet_files = sorted(index_dir.glob("*.parquet"))
    if not parquet_files:
        # Fall back: read the cluster.idx to locate the cdx-api parquet shards.
        # In production, this would involve querying the CC index API.  For
        # robustness we also support pre-downloaded parquet shards.
        logger.warning(
            "No parquet files found in %s.  Attempting CC Index Server API …",
            index_dir,
        )
        return _sample_via_index_api(crawl_id, n, seed, languages, crawl_date)

    # Read parquet index.
    logger.info("Reading %d parquet shard(s) from %s", len(parquet_files), index_dir)
    table = pq.read_table(parquet_files[0])
    df = table.to_pandas()

    # Optional language filter.
    if languages:
        lang_set = set(languages)
        if "content_languages" in df.columns:
            mask = df["content_languages"].apply(
                lambda v: bool(set(str(v).split(",")) & lang_set) if v else False
            )
            df = df[mask]

    if df.empty:
        logger.warning("No records left after language filtering for %s", crawl_id)
        return []

    rng = random.Random(seed)
    sample_size = min(n * 3, len(df))  # oversample to account for extraction failures
    indices = rng.sample(range(len(df)), k=sample_size)
    sampled_rows = df.iloc[indices]

    documents: list[Document] = []
    for _, row in sampled_rows.iterrows():
        if len(documents) >= n:
            break

        warc_filename = row.get("warc_filename", "")
        warc_offset = int(row.get("warc_record_offset", 0))
        warc_length = int(row.get("warc_record_length", 0))
        record_url = row.get("url", None)

        if not warc_filename or warc_length == 0:
            continue

        text = _fetch_warc_record(warc_filename, warc_offset, warc_length)
        if text is None:
            continue

        doc_id = hashlib.sha256(
            f"{crawl_id}:{record_url}:{warc_offset}".encode()
        ).hexdigest()[:16]

        documents.append(
            Document(
                doc_id=f"cc-{doc_id}",
                text=text,
                source="common_crawl",
                timestamp=crawl_date,
                url=str(record_url) if record_url else None,
                metadata={
                    "crawl_id": crawl_id,
                    "warc_filename": warc_filename,
                    "warc_offset": warc_offset,
                    "warc_length": warc_length,
                },
            )
        )
        logger.debug("Sampled CC document %s (%d chars)", documents[-1].doc_id, len(text))

    logger.info(
        "Sampled %d documents from Common Crawl crawl %s", len(documents), crawl_id
    )
    return documents


def _sample_via_index_api(
    crawl_id: str,
    n: int,
    seed: int,
    languages: list[str] | None,
    crawl_date: datetime,
) -> list[Document]:
    """Fall-back sampling via the Common Crawl Index Server (cdx-api).

    Queries the CC Index Server for random URL patterns and fetches matching
    WARC records.
    """
    # Use a set of broadly-popular URL seeds to find diverse content.
    url_seeds = [
        "*.com/*",
        "*.org/*",
        "*.net/*",
        "*.edu/*",
        "*.co.uk/*",
    ]

    rng = random.Random(seed)
    rng.shuffle(url_seeds)

    api_base = f"https://index.commoncrawl.org/{crawl_id}-index"
    documents: list[Document] = []

    for url_pattern in url_seeds:
        if len(documents) >= n:
            break

        params: dict[str, Any] = {
            "url": url_pattern,
            "output": "json",
            "limit": min(n * 2, 500),
        }
        if languages:
            params["filter"] = f"=languages:{languages[0]}"

        try:
            resp = _request_with_retries("GET", api_base, params=params, timeout=120)
        except RuntimeError:
            logger.warning(
                "Index API query failed for pattern %s on %s", url_pattern, crawl_id
            )
            continue

        # Each line is a JSON-encoded record.
        import json

        lines = resp.text.strip().splitlines()
        rng.shuffle(lines)

        for line in lines:
            if len(documents) >= n:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            warc_filename = rec.get("filename", "")
            warc_offset = int(rec.get("offset", 0))
            warc_length = int(rec.get("length", 0))
            record_url = rec.get("url")

            if not warc_filename or warc_length == 0:
                continue

            text = _fetch_warc_record(warc_filename, warc_offset, warc_length)
            if text is None:
                continue

            doc_id = hashlib.sha256(
                f"{crawl_id}:{record_url}:{warc_offset}".encode()
            ).hexdigest()[:16]

            documents.append(
                Document(
                    doc_id=f"cc-{doc_id}",
                    text=text,
                    source="common_crawl",
                    timestamp=crawl_date,
                    url=str(record_url) if record_url else None,
                    metadata={
                        "crawl_id": crawl_id,
                        "warc_filename": warc_filename,
                        "warc_offset": warc_offset,
                        "warc_length": warc_length,
                    },
                )
            )

    logger.info(
        "Sampled %d documents via CC Index API for crawl %s",
        len(documents),
        crawl_id,
    )
    return documents
