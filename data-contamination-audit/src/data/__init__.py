"""Data ingestion package for the data-contamination-audit pipeline.

Re-exports the shared :class:`Document` dataclass and the public API of each
sub-module for convenient top-level imports.
"""

from src.data.common_crawl import Document, download_cc_index, sample_cc_warc
from src.data.sampler import build_temporal_corpus
from src.data.timestamper import assign_time_bin
from src.data.wikipedia import (
    download_wikipedia_dump,
    parse_wikipedia_dump,
    sample_wikipedia,
)

__all__ = [
    "Document",
    "assign_time_bin",
    "build_temporal_corpus",
    "download_cc_index",
    "download_wikipedia_dump",
    "parse_wikipedia_dump",
    "sample_cc_warc",
    "sample_wikipedia",
]
