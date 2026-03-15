"""Comprehensive tests for data ingestion modules.

Covers:
- src/data/timestamper.py  (assign_time_bin)
- src/data/common_crawl.py (Document, _request_with_retries, download_cc_index,
                            _fetch_warc_record, sample_cc_warc, _load_crawl_config)
- src/data/wikipedia.py    (parse_wikipedia_dump, sample_wikipedia,
                            download_wikipedia_dump, filtering helpers)
- src/data/sampler.py      (build_temporal_corpus, _find_crawl_for_bin,
                            _resolve_wikipedia_dump)
"""

from __future__ import annotations

import hashlib
import io
import json
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import requests

from src.data.common_crawl import Document

# ---------------------------------------------------------------------------
# Paths to test fixtures
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
SAMPLE_WIKI_XML = TEST_DATA_DIR / "sample_wiki_articles.xml"


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def jan1_doc():
    """Document timestamped January 1."""
    return Document(
        doc_id="d-jan1",
        text="jan1 text",
        source="test",
        timestamp=datetime(2023, 1, 1),
        url=None,
    )


@pytest.fixture
def jun30_doc():
    """Document timestamped June 30."""
    return Document(
        doc_id="d-jun30",
        text="jun30 text",
        source="test",
        timestamp=datetime(2023, 6, 30),
        url=None,
    )


@pytest.fixture
def jul1_doc():
    """Document timestamped July 1."""
    return Document(
        doc_id="d-jul1",
        text="jul1 text",
        source="test",
        timestamp=datetime(2023, 7, 1),
        url=None,
    )


@pytest.fixture
def dec31_doc():
    """Document timestamped December 31."""
    return Document(
        doc_id="d-dec31",
        text="dec31 text",
        source="test",
        timestamp=datetime(2023, 12, 31),
        url=None,
    )


@pytest.fixture
def crawl_config_data():
    """Minimal crawl config YAML content."""
    return {
        "crawls": [
            {"id": "CC-MAIN-2021-21", "date": "2021-05-01"},
            {"id": "CC-MAIN-2023-23", "date": "2023-06-01"},
            {"id": "CC-MAIN-2024-22", "date": "2024-05-01"},
        ]
    }


@pytest.fixture
def mock_crawl_config():
    """Return a dict as produced by _load_crawl_config."""
    return {
        "CC-MAIN-2021-21": "2021-05-01",
        "CC-MAIN-2023-23": "2023-06-01",
        "CC-MAIN-2024-22": "2024-05-01",
    }


# ===================================================================
# timestamper.py tests
# ===================================================================


class TestAssignTimeBin:
    """Tests for src.data.timestamper.assign_time_bin."""

    # -- year bin -------------------------------------------------------

    def test_year_bin_basic(self, jan1_doc):
        from src.data.timestamper import assign_time_bin

        assert assign_time_bin(jan1_doc, "year") == "2023"

    def test_year_bin_dec31(self, dec31_doc):
        from src.data.timestamper import assign_time_bin

        assert assign_time_bin(dec31_doc, "year") == "2023"

    # -- half-year bin --------------------------------------------------

    def test_half_year_h1_jan1(self, jan1_doc):
        from src.data.timestamper import assign_time_bin

        assert assign_time_bin(jan1_doc, "half-year") == "2023-H1"

    def test_half_year_h1_jun30(self, jun30_doc):
        from src.data.timestamper import assign_time_bin

        # June 30 is still H1 (month <= 6).
        assert assign_time_bin(jun30_doc, "half-year") == "2023-H1"

    def test_half_year_h2_jul1(self, jul1_doc):
        from src.data.timestamper import assign_time_bin

        assert assign_time_bin(jul1_doc, "half-year") == "2023-H2"

    def test_half_year_h2_dec31(self, dec31_doc):
        from src.data.timestamper import assign_time_bin

        assert assign_time_bin(dec31_doc, "half-year") == "2023-H2"

    # -- quarter bin ----------------------------------------------------

    def test_quarter_q1_jan1(self, jan1_doc):
        from src.data.timestamper import assign_time_bin

        assert assign_time_bin(jan1_doc, "quarter") == "2023-Q1"

    def test_quarter_q1_mar31(self):
        from src.data.timestamper import assign_time_bin

        doc = Document(
            doc_id="x", text="t", source="s",
            timestamp=datetime(2023, 3, 31), url=None,
        )
        assert assign_time_bin(doc, "quarter") == "2023-Q1"

    def test_quarter_q2_apr1(self):
        from src.data.timestamper import assign_time_bin

        doc = Document(
            doc_id="x", text="t", source="s",
            timestamp=datetime(2023, 4, 1), url=None,
        )
        assert assign_time_bin(doc, "quarter") == "2023-Q2"

    def test_quarter_q2_jun30(self, jun30_doc):
        from src.data.timestamper import assign_time_bin

        assert assign_time_bin(jun30_doc, "quarter") == "2023-Q2"

    def test_quarter_q3_jul1(self, jul1_doc):
        from src.data.timestamper import assign_time_bin

        assert assign_time_bin(jul1_doc, "quarter") == "2023-Q3"

    def test_quarter_q3_sep30(self):
        from src.data.timestamper import assign_time_bin

        doc = Document(
            doc_id="x", text="t", source="s",
            timestamp=datetime(2023, 9, 30), url=None,
        )
        assert assign_time_bin(doc, "quarter") == "2023-Q3"

    def test_quarter_q4_oct1(self):
        from src.data.timestamper import assign_time_bin

        doc = Document(
            doc_id="x", text="t", source="s",
            timestamp=datetime(2023, 10, 1), url=None,
        )
        assert assign_time_bin(doc, "quarter") == "2023-Q4"

    def test_quarter_q4_dec31(self, dec31_doc):
        from src.data.timestamper import assign_time_bin

        assert assign_time_bin(dec31_doc, "quarter") == "2023-Q4"

    # -- invalid bin_size -----------------------------------------------

    def test_invalid_bin_size_raises(self, jan1_doc):
        from src.data.timestamper import assign_time_bin

        with pytest.raises(ValueError, match="Invalid bin_size"):
            assign_time_bin(jan1_doc, "month")

    def test_invalid_bin_size_empty(self, jan1_doc):
        from src.data.timestamper import assign_time_bin

        with pytest.raises(ValueError):
            assign_time_bin(jan1_doc, "")

    # -- default bin_size is "year" -------------------------------------

    def test_default_bin_size(self, jan1_doc):
        from src.data.timestamper import assign_time_bin

        assert assign_time_bin(jan1_doc) == "2023"


# ===================================================================
# common_crawl.py — Document tests
# ===================================================================


class TestDocument:
    """Tests for the Document dataclass."""

    def test_creation_with_explicit_doc_id(self):
        doc = Document(
            doc_id="explicit-id",
            text="Hello world",
            source="wikipedia",
            timestamp=datetime(2023, 1, 1),
            url="https://example.com",
        )
        assert doc.doc_id == "explicit-id"
        assert doc.text == "Hello world"
        assert doc.source == "wikipedia"
        assert doc.url == "https://example.com"
        assert doc.metadata == {}

    def test_auto_generated_doc_id_when_empty(self):
        doc = Document(
            doc_id="",
            text="Hello world",
            source="wikipedia",
            timestamp=datetime(2023, 1, 1),
            url=None,
        )
        h = hashlib.sha256("Hello world".encode("utf-8")).hexdigest()[:16]
        assert doc.doc_id == f"wikipedia-{h}"

    def test_auto_generated_doc_id_deterministic(self):
        doc1 = Document(
            doc_id="", text="same text", source="common_crawl",
            timestamp=datetime(2023, 1, 1), url=None,
        )
        doc2 = Document(
            doc_id="", text="same text", source="common_crawl",
            timestamp=datetime(2024, 6, 1), url=None,
        )
        assert doc1.doc_id == doc2.doc_id

    def test_different_text_different_auto_id(self):
        doc1 = Document(
            doc_id="", text="text A", source="x",
            timestamp=datetime(2023, 1, 1), url=None,
        )
        doc2 = Document(
            doc_id="", text="text B", source="x",
            timestamp=datetime(2023, 1, 1), url=None,
        )
        assert doc1.doc_id != doc2.doc_id

    def test_metadata_default_factory(self):
        doc1 = Document(
            doc_id="a", text="t", source="s",
            timestamp=datetime(2023, 1, 1), url=None,
        )
        doc2 = Document(
            doc_id="b", text="t", source="s",
            timestamp=datetime(2023, 1, 1), url=None,
        )
        # Ensure each instance gets its own dict.
        doc1.metadata["key"] = "val"
        assert "key" not in doc2.metadata

    def test_metadata_custom(self):
        doc = Document(
            doc_id="a", text="t", source="s",
            timestamp=datetime(2023, 1, 1), url=None,
            metadata={"custom": True},
        )
        assert doc.metadata == {"custom": True}


# ===================================================================
# common_crawl.py — _load_crawl_config
# ===================================================================


class TestLoadCrawlConfig:
    """Tests for loading cc_crawls.yaml."""

    def test_load_crawl_config_returns_dict(self, crawl_config_data):
        yaml_text = (
            "crawls:\n"
            "  - id: CC-MAIN-2023-23\n"
            '    date: "2023-06-01"\n'
        )
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__ = lambda s: io.StringIO(yaml_text)
            mock_open.return_value.__exit__ = Mock(return_value=False)
            from src.data.common_crawl import _load_crawl_config

            with patch("src.data.common_crawl._CC_CRAWLS_PATH", new="dummy.yaml"):
                result = _load_crawl_config()

        assert isinstance(result, dict)
        assert "CC-MAIN-2023-23" in result
        assert result["CC-MAIN-2023-23"] == "2023-06-01"


# ===================================================================
# common_crawl.py — _request_with_retries
# ===================================================================


class TestRequestWithRetries:
    """Tests for _request_with_retries."""

    @patch("src.data.common_crawl.time.sleep")
    @patch("src.data.common_crawl.requests.request")
    def test_success_first_attempt(self, mock_request, mock_sleep):
        from src.data.common_crawl import _request_with_retries

        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_request.return_value = mock_resp

        resp = _request_with_retries("GET", "https://example.com", max_retries=3)

        assert resp is mock_resp
        mock_request.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("src.data.common_crawl.time.sleep")
    @patch("src.data.common_crawl.requests.request")
    def test_success_after_retries(self, mock_request, mock_sleep):
        from src.data.common_crawl import _request_with_retries

        failure = requests.ConnectionError("fail")
        mock_good = Mock()
        mock_good.raise_for_status = Mock()
        mock_request.side_effect = [failure, failure, mock_good]

        resp = _request_with_retries("GET", "https://example.com", max_retries=3)
        assert resp is mock_good
        assert mock_request.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("src.data.common_crawl.time.sleep")
    @patch("src.data.common_crawl.requests.request")
    def test_all_retries_exhausted(self, mock_request, mock_sleep):
        from src.data.common_crawl import _request_with_retries

        mock_request.side_effect = requests.ConnectionError("fail")

        with pytest.raises(RuntimeError, match="All 2 attempts"):
            _request_with_retries("GET", "https://example.com", max_retries=2)

        assert mock_request.call_count == 2
        assert mock_sleep.call_count == 2

    @patch("src.data.common_crawl.time.sleep")
    @patch("src.data.common_crawl.requests.request")
    def test_http_error_triggers_retry(self, mock_request, mock_sleep):
        from src.data.common_crawl import _request_with_retries

        bad_resp = Mock()
        bad_resp.raise_for_status.side_effect = requests.HTTPError("500")
        good_resp = Mock()
        good_resp.raise_for_status = Mock()
        mock_request.side_effect = [bad_resp, good_resp]

        resp = _request_with_retries("GET", "https://example.com", max_retries=3)
        assert resp is good_resp

    @patch("src.data.common_crawl.time.sleep")
    @patch("src.data.common_crawl.requests.request")
    def test_passes_kwargs(self, mock_request, mock_sleep):
        from src.data.common_crawl import _request_with_retries

        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_request.return_value = mock_resp

        _request_with_retries(
            "POST", "https://example.com",
            max_retries=1, headers={"X-Custom": "val"},
        )
        _, kwargs = mock_request.call_args
        assert kwargs["headers"] == {"X-Custom": "val"}


# ===================================================================
# common_crawl.py — download_cc_index
# ===================================================================


class TestDownloadCCIndex:
    """Tests for download_cc_index."""

    @patch("src.data.common_crawl._request_with_retries")
    @patch("src.data.common_crawl._load_crawl_config")
    def test_invalid_crawl_id_raises(self, mock_config, mock_req):
        from src.data.common_crawl import download_cc_index

        mock_config.return_value = {"CC-MAIN-2023-23": "2023-06-01"}

        with pytest.raises(ValueError, match="Unknown crawl_id"):
            download_cc_index("CC-MAIN-FAKE")

    @patch("src.data.common_crawl._request_with_retries")
    @patch("src.data.common_crawl._load_crawl_config")
    def test_downloads_cluster_idx(self, mock_config, mock_req, tmp_path):
        from src.data.common_crawl import download_cc_index

        mock_config.return_value = {"CC-MAIN-2023-23": "2023-06-01"}

        mock_resp = Mock()
        mock_resp.content = b"fake cluster index data"
        mock_req.return_value = mock_resp

        with patch("src.data.common_crawl._RAW_DATA_DIR", tmp_path):
            result = download_cc_index("CC-MAIN-2023-23")

        assert result.exists()
        cluster_file = result / "cluster.idx"
        assert cluster_file.exists()
        assert cluster_file.read_bytes() == b"fake cluster index data"

    @patch("src.data.common_crawl._request_with_retries")
    @patch("src.data.common_crawl._load_crawl_config")
    def test_cached_index_skips_download(self, mock_config, mock_req, tmp_path):
        from src.data.common_crawl import download_cc_index

        mock_config.return_value = {"CC-MAIN-2023-23": "2023-06-01"}

        index_dir = tmp_path / "cc-index" / "CC-MAIN-2023-23"
        index_dir.mkdir(parents=True)
        (index_dir / "cluster.idx").write_text("cached")

        with patch("src.data.common_crawl._RAW_DATA_DIR", tmp_path):
            result = download_cc_index("CC-MAIN-2023-23")

        mock_req.assert_not_called()
        assert result == index_dir


# ===================================================================
# common_crawl.py — _fetch_warc_record
# ===================================================================


class TestFetchWarcRecord:
    """Tests for _fetch_warc_record."""

    @patch("src.data.common_crawl.trafilatura.extract")
    @patch("src.data.common_crawl.ArchiveIterator")
    @patch("src.data.common_crawl._request_with_retries")
    def test_successful_extraction(self, mock_req, mock_archive_iter, mock_extract):
        from src.data.common_crawl import _fetch_warc_record

        mock_resp = Mock()
        mock_resp.content = b"fake warc bytes"
        mock_req.return_value = mock_resp

        mock_record = Mock()
        mock_record.rec_type = "response"
        mock_record.content_stream.return_value.read.return_value = b"<html>hello</html>"
        mock_archive_iter.return_value = [mock_record]

        mock_extract.return_value = (
            "This is a full article with multiple sentences. "
            "It contains enough content to pass quality filters. "
            "The third sentence ensures the minimum threshold is met. "
            "And a fourth for good measure."
        )

        result = _fetch_warc_record("crawl-data/segment/warc.gz", 100, 500)

        assert result is not None
        assert "full article" in result
        mock_req.assert_called_once_with(
            "GET",
            "https://data.commoncrawl.org/crawl-data/segment/warc.gz",
            headers={"Range": "bytes=100-599"},
        )

    @patch("src.data.common_crawl._request_with_retries")
    def test_request_failure_returns_none(self, mock_req):
        from src.data.common_crawl import _fetch_warc_record

        mock_req.side_effect = RuntimeError("All retries failed")
        result = _fetch_warc_record("file.warc.gz", 0, 100)
        assert result is None

    @patch("src.data.common_crawl.trafilatura.extract")
    @patch("src.data.common_crawl.ArchiveIterator")
    @patch("src.data.common_crawl._request_with_retries")
    def test_trafilatura_returns_none(self, mock_req, mock_archive_iter, mock_extract):
        from src.data.common_crawl import _fetch_warc_record

        mock_resp = Mock()
        mock_resp.content = b"data"
        mock_req.return_value = mock_resp

        mock_record = Mock()
        mock_record.rec_type = "response"
        mock_record.content_stream.return_value.read.return_value = b"<html></html>"
        mock_archive_iter.return_value = [mock_record]

        mock_extract.return_value = None

        result = _fetch_warc_record("f.warc.gz", 0, 100)
        assert result is None

    @patch("src.data.common_crawl.trafilatura.extract")
    @patch("src.data.common_crawl.ArchiveIterator")
    @patch("src.data.common_crawl._request_with_retries")
    def test_skips_non_response_records(self, mock_req, mock_archive_iter, mock_extract):
        from src.data.common_crawl import _fetch_warc_record

        mock_resp = Mock()
        mock_resp.content = b"data"
        mock_req.return_value = mock_resp

        # Only a "request" record, no "response"
        mock_record = Mock()
        mock_record.rec_type = "request"
        mock_archive_iter.return_value = [mock_record]

        result = _fetch_warc_record("f.warc.gz", 0, 100)
        assert result is None
        mock_extract.assert_not_called()

    @patch("src.data.common_crawl.ArchiveIterator")
    @patch("src.data.common_crawl._request_with_retries")
    def test_archive_parse_exception(self, mock_req, mock_archive_iter):
        from src.data.common_crawl import _fetch_warc_record

        mock_resp = Mock()
        mock_resp.content = b"corrupt data"
        mock_req.return_value = mock_resp

        mock_archive_iter.side_effect = Exception("corrupt WARC")

        result = _fetch_warc_record("f.warc.gz", 0, 100)
        assert result is None


# ===================================================================
# common_crawl.py — sample_cc_warc
# ===================================================================


class TestSampleCCWarc:
    """Tests for sample_cc_warc."""

    @patch("src.data.common_crawl._fetch_warc_record")
    @patch("src.data.common_crawl.download_cc_index")
    @patch("src.data.common_crawl._load_crawl_config")
    def test_unknown_crawl_id(self, mock_config, mock_download, mock_fetch):
        from src.data.common_crawl import sample_cc_warc

        mock_config.return_value = {"CC-MAIN-2023-23": "2023-06-01"}

        with pytest.raises(ValueError, match="Unknown crawl_id"):
            sample_cc_warc("CC-MAIN-FAKE", n=5, seed=42)

    @patch("src.data.common_crawl._fetch_warc_record")
    @patch("src.data.common_crawl.pq.read_table")
    @patch("src.data.common_crawl.download_cc_index")
    @patch("src.data.common_crawl._load_crawl_config")
    def test_samples_from_parquet(
        self, mock_config, mock_download, mock_read_table, mock_fetch, tmp_path
    ):
        from src.data.common_crawl import sample_cc_warc
        import pandas as pd

        mock_config.return_value = {"CC-MAIN-2023-23": "2023-06-01"}

        # Create a fake parquet directory with a parquet file.
        index_dir = tmp_path / "index"
        index_dir.mkdir()
        (index_dir / "part-00000.parquet").write_text("fake")
        mock_download.return_value = index_dir

        # Build a mock parquet table.
        df = pd.DataFrame(
            {
                "url": [f"https://example.com/page{i}" for i in range(10)],
                "warc_filename": [f"crawl/seg/warc{i}.gz" for i in range(10)],
                "warc_record_offset": list(range(0, 10000, 1000)),
                "warc_record_length": [500] * 10,
                "content_languages": ["en"] * 10,
            }
        )
        mock_table = Mock()
        mock_table.to_pandas.return_value = df
        mock_read_table.return_value = mock_table

        mock_fetch.return_value = "Extracted document text for testing."

        docs = sample_cc_warc("CC-MAIN-2023-23", n=3, seed=42)

        assert len(docs) == 3
        for doc in docs:
            assert doc.source == "common_crawl"
            assert doc.text == "Extracted document text for testing."
            assert doc.doc_id.startswith("cc-")
            assert doc.timestamp == datetime(2023, 6, 1)
            assert doc.metadata["crawl_id"] == "CC-MAIN-2023-23"

    @patch("src.data.common_crawl._fetch_warc_record")
    @patch("src.data.common_crawl.pq.read_table")
    @patch("src.data.common_crawl.download_cc_index")
    @patch("src.data.common_crawl._load_crawl_config")
    def test_handles_extraction_failures(
        self, mock_config, mock_download, mock_read_table, mock_fetch, tmp_path
    ):
        from src.data.common_crawl import sample_cc_warc
        import pandas as pd

        mock_config.return_value = {"CC-MAIN-2023-23": "2023-06-01"}

        index_dir = tmp_path / "index"
        index_dir.mkdir()
        (index_dir / "part-00000.parquet").write_text("fake")
        mock_download.return_value = index_dir

        df = pd.DataFrame(
            {
                "url": [f"https://example.com/{i}" for i in range(10)],
                "warc_filename": [f"seg/warc{i}.gz" for i in range(10)],
                "warc_record_offset": [100] * 10,
                "warc_record_length": [500] * 10,
                "content_languages": ["en"] * 10,
            }
        )
        mock_table = Mock()
        mock_table.to_pandas.return_value = df
        mock_read_table.return_value = mock_table

        # Some extractions fail.
        mock_fetch.side_effect = [None, "text A", None, "text B", None, None, "text C"] + [None] * 20

        docs = sample_cc_warc("CC-MAIN-2023-23", n=3, seed=42)
        assert len(docs) == 3

    @patch("src.data.common_crawl._fetch_warc_record")
    @patch("src.data.common_crawl.pq.read_table")
    @patch("src.data.common_crawl.download_cc_index")
    @patch("src.data.common_crawl._load_crawl_config")
    def test_language_filter(
        self, mock_config, mock_download, mock_read_table, mock_fetch, tmp_path
    ):
        from src.data.common_crawl import sample_cc_warc
        import pandas as pd

        mock_config.return_value = {"CC-MAIN-2023-23": "2023-06-01"}

        index_dir = tmp_path / "index"
        index_dir.mkdir()
        (index_dir / "shard.parquet").write_text("fake")
        mock_download.return_value = index_dir

        df = pd.DataFrame(
            {
                "url": [f"https://example.com/{i}" for i in range(6)],
                "warc_filename": [f"warc{i}.gz" for i in range(6)],
                "warc_record_offset": [0] * 6,
                "warc_record_length": [500] * 6,
                "content_languages": ["en", "de", "en", "fr", "en", "en"],
            }
        )
        mock_table = Mock()
        mock_table.to_pandas.return_value = df
        mock_read_table.return_value = mock_table

        mock_fetch.return_value = "Filtered text"

        docs = sample_cc_warc("CC-MAIN-2023-23", n=2, seed=42, languages=["en"])
        assert len(docs) == 2

    @patch("src.data.common_crawl._sample_via_index_api")
    @patch("src.data.common_crawl.download_cc_index")
    @patch("src.data.common_crawl._load_crawl_config")
    def test_no_parquet_falls_back_to_api(
        self, mock_config, mock_download, mock_api, tmp_path
    ):
        from src.data.common_crawl import sample_cc_warc

        mock_config.return_value = {"CC-MAIN-2023-23": "2023-06-01"}

        # Empty index dir — no parquet files.
        index_dir = tmp_path / "index"
        index_dir.mkdir()
        mock_download.return_value = index_dir

        mock_api.return_value = []

        docs = sample_cc_warc("CC-MAIN-2023-23", n=5, seed=42)
        assert docs == []
        mock_api.assert_called_once()


# ===================================================================
# common_crawl.py — HTML text extraction (via trafilatura)
# ===================================================================


class TestHTMLExtraction:
    """Test that trafilatura extracts text from sample HTML files."""

    def test_extract_from_news_article(self):
        import trafilatura

        html_path = TEST_DATA_DIR / "sample_html_pages" / "news_article.html"
        if not html_path.exists():
            pytest.skip("Sample HTML file not found")

        html = html_path.read_text()
        text = trafilatura.extract(html)
        assert text is not None
        assert "Scientists Discover" in text or "deep-sea" in text.lower() or "ocean" in text.lower()


# ===================================================================
# wikipedia.py — filtering helpers
# ===================================================================


class TestWikipediaFiltering:
    """Tests for _is_redirect, _is_disambiguation, _strip_wikitext."""

    def test_is_redirect_true(self):
        from src.data.wikipedia import _is_redirect

        assert _is_redirect("#REDIRECT [[Some Page]]") is True
        assert _is_redirect("#redirect [[Other]]") is True

    def test_is_redirect_false(self):
        from src.data.wikipedia import _is_redirect

        assert _is_redirect("Normal article text") is False
        assert _is_redirect("See also #REDIRECT") is False

    def test_is_disambiguation_true(self):
        from src.data.wikipedia import _is_disambiguation

        assert _is_disambiguation("Some text\n{{disambiguation}}") is True
        assert _is_disambiguation("{{Disambig|geo}}") is True
        assert _is_disambiguation("{{hndis|Foo}}") is True

    def test_is_disambiguation_false(self):
        from src.data.wikipedia import _is_disambiguation

        assert _is_disambiguation("Normal article text") is False
        assert _is_disambiguation("{{reflist}}") is False

    def test_strip_wikitext(self):
        from src.data.wikipedia import _strip_wikitext

        result = _strip_wikitext("'''Bold text''' and [[link|display]].")
        assert "Bold text" in result
        assert "display" in result
        assert "[[" not in result
        assert "'''" not in result


# ===================================================================
# wikipedia.py — _detect_namespace
# ===================================================================


class TestDetectNamespace:
    """Tests for _detect_namespace."""

    def test_detects_010(self):
        from src.data.wikipedia import _detect_namespace, _MW_NS

        result = _detect_namespace(SAMPLE_WIKI_XML)
        assert result == _MW_NS  # 0.10

    def test_detects_011(self, tmp_path):
        from src.data.wikipedia import _detect_namespace, _MW_NS_ALT

        xml_content = '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">\n</mediawiki>'
        xml_file = tmp_path / "dump.xml"
        xml_file.write_text(xml_content)

        result = _detect_namespace(xml_file)
        assert result == _MW_NS_ALT


# ===================================================================
# wikipedia.py — parse_wikipedia_dump
# ===================================================================


class TestParseWikipediaDump:
    """Tests for parse_wikipedia_dump using sample XML."""

    def test_parse_yields_documents(self):
        from src.data.wikipedia import parse_wikipedia_dump

        docs = list(parse_wikipedia_dump(SAMPLE_WIKI_XML))
        # The sample XML has 8 pages: 2 redirects, 1 disambiguation,
        # 1 stub, 4 substantive articles => 5 non-redirect articles yielded
        # (redirects are skipped at XML level; disambiguation and stub ARE
        # yielded — filtering happens later in sample_wikipedia).
        assert len(docs) >= 4  # at least the 4 substantive + disambiguation + stub

    def test_redirects_are_skipped(self):
        from src.data.wikipedia import parse_wikipedia_dump

        docs = list(parse_wikipedia_dump(SAMPLE_WIKI_XML))
        titles = [d.metadata["title"] for d in docs]
        assert "Redirect Example" not in titles
        assert "Another Redirect" not in titles

    def test_document_fields_populated(self):
        from src.data.wikipedia import parse_wikipedia_dump

        docs = list(parse_wikipedia_dump(SAMPLE_WIKI_XML))
        test_planet = [d for d in docs if d.metadata.get("title") == "Test Planet"]
        assert len(test_planet) == 1
        doc = test_planet[0]

        assert doc.source == "wikipedia"
        assert doc.doc_id.startswith("wiki-")
        assert "planet" in doc.text.lower() or "Test Planet" in doc.text
        assert doc.url == "https://en.wikipedia.org/wiki/Test_Planet"
        assert doc.metadata["page_id"] == "1001"
        assert doc.timestamp.year == 2023

    def test_disambiguation_marked(self):
        from src.data.wikipedia import parse_wikipedia_dump

        docs = list(parse_wikipedia_dump(SAMPLE_WIKI_XML))
        disambig = [
            d for d in docs
            if d.metadata.get("title") == "Foo (disambiguation)"
        ]
        assert len(disambig) == 1
        assert disambig[0].metadata["is_disambiguation"] is True

    def test_non_disambiguation_marked_false(self):
        from src.data.wikipedia import parse_wikipedia_dump

        docs = list(parse_wikipedia_dump(SAMPLE_WIKI_XML))
        test_planet = [d for d in docs if d.metadata.get("title") == "Test Planet"]
        assert test_planet[0].metadata["is_disambiguation"] is False

    def test_timestamps_parsed(self):
        from src.data.wikipedia import parse_wikipedia_dump

        docs = list(parse_wikipedia_dump(SAMPLE_WIKI_XML))
        tea = [d for d in docs if d.metadata.get("title") == "History of Tea"]
        assert len(tea) == 1
        assert tea[0].timestamp.year == 2021
        assert tea[0].timestamp.month == 9

    def test_wikitext_stripped(self):
        from src.data.wikipedia import parse_wikipedia_dump

        docs = list(parse_wikipedia_dump(SAMPLE_WIKI_XML))
        for doc in docs:
            # No raw wikitext markup should survive.
            assert "[[" not in doc.text
            assert "{{" not in doc.text
            assert "'''" not in doc.text


# ===================================================================
# wikipedia.py — sample_wikipedia
# ===================================================================


class TestSampleWikipedia:
    """Tests for sample_wikipedia."""

    @patch("src.data.wikipedia._save_documents_parquet")
    @patch("src.data.wikipedia.parse_wikipedia_dump")
    def test_filters_stubs_and_disambiguation(self, mock_parse, mock_save, tmp_path):
        from src.data.wikipedia import sample_wikipedia

        docs = [
            Document(
                doc_id="wiki-001", text="A" * 600, source="wikipedia",
                timestamp=datetime(2023, 1, 1), url="https://en.wikipedia.org/wiki/A",
                metadata={"title": "Real Article", "page_id": "1", "is_disambiguation": False},
            ),
            Document(
                doc_id="wiki-002", text="B" * 600, source="wikipedia",
                timestamp=datetime(2023, 1, 1), url="https://en.wikipedia.org/wiki/B",
                metadata={"title": "Another Article", "page_id": "2", "is_disambiguation": False},
            ),
            Document(
                doc_id="wiki-stub", text="Short", source="wikipedia",
                timestamp=datetime(2023, 1, 1), url="https://en.wikipedia.org/wiki/S",
                metadata={"title": "Stub", "page_id": "3", "is_disambiguation": False},
            ),
            Document(
                doc_id="wiki-disambig", text="D" * 600, source="wikipedia",
                timestamp=datetime(2023, 1, 1), url="https://en.wikipedia.org/wiki/D",
                metadata={"title": "Disambig", "page_id": "4", "is_disambiguation": True},
            ),
        ]
        mock_parse.return_value = docs

        dump_path = tmp_path / "enwiki-20230101-pages-articles.xml.bz2"
        dump_path.touch()

        with patch("src.data.wikipedia._PROCESSED_DIR", tmp_path / "processed"):
            result = sample_wikipedia(dump_path, n=2, seed=42)

        # Stub and disambiguation are filtered out.
        assert len(result) == 2
        ids = {d.doc_id for d in result}
        assert "wiki-stub" not in ids
        assert "wiki-disambig" not in ids

    @patch("src.data.wikipedia._save_documents_parquet")
    @patch("src.data.wikipedia.parse_wikipedia_dump")
    def test_returns_all_when_fewer_eligible(self, mock_parse, mock_save, tmp_path):
        from src.data.wikipedia import sample_wikipedia

        docs = [
            Document(
                doc_id="wiki-only", text="X" * 600, source="wikipedia",
                timestamp=datetime(2023, 1, 1), url=None,
                metadata={"title": "Only", "page_id": "1", "is_disambiguation": False},
            ),
        ]
        mock_parse.return_value = docs

        dump_path = tmp_path / "enwiki-20230101-pages-articles.xml.bz2"
        dump_path.touch()

        with patch("src.data.wikipedia._PROCESSED_DIR", tmp_path / "processed"):
            result = sample_wikipedia(dump_path, n=10, seed=42)

        assert len(result) == 1

    @patch("src.data.wikipedia.pd.read_parquet")
    def test_loads_from_cache(self, mock_read_pq, tmp_path):
        from src.data.wikipedia import sample_wikipedia
        import pandas as pd

        # Pre-create the parquet cache file.
        processed_dir = tmp_path / "processed" / "wikipedia" / "20230101"
        processed_dir.mkdir(parents=True)
        parquet_path = processed_dir / "articles.parquet"
        parquet_path.touch()

        # Build a dataframe that _dataframe_to_documents can parse.
        df = pd.DataFrame(
            [
                {
                    "doc_id": "wiki-cached",
                    "text": "Y" * 600,
                    "source": "wikipedia",
                    "timestamp": "2023-01-01T00:00:00",
                    "url": "https://en.wikipedia.org/wiki/Cached",
                    "title": "Cached Article",
                    "page_id": "99",
                    "is_disambiguation": False,
                },
            ]
        )
        mock_read_pq.return_value = df

        dump_path = tmp_path / "enwiki-20230101-pages-articles.xml.bz2"
        dump_path.touch()

        with patch("src.data.wikipedia._PROCESSED_DIR", tmp_path / "processed"):
            result = sample_wikipedia(dump_path, n=1, seed=42)

        assert len(result) == 1
        assert result[0].doc_id == "wiki-cached"


# ===================================================================
# wikipedia.py — download_wikipedia_dump
# ===================================================================


class TestDownloadWikipediaDump:
    """Tests for download_wikipedia_dump."""

    @patch("src.data.wikipedia.requests.get")
    def test_constructs_correct_url(self, mock_get, tmp_path):
        from src.data.wikipedia import download_wikipedia_dump

        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_resp.headers = {"content-length": "100"}
        mock_resp.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_resp

        result = download_wikipedia_dump("20230601", output_dir=tmp_path)

        expected_url = (
            "https://dumps.wikimedia.org/enwiki/20230601/"
            "enwiki-20230601-pages-articles.xml.bz2"
        )
        mock_get.assert_called_once_with(
            expected_url, stream=True, timeout=120,
        )
        assert result == tmp_path / "enwiki-20230601-pages-articles.xml.bz2"
        assert result.exists()

    @patch("src.data.wikipedia.requests.get")
    def test_cached_dump_skips_download(self, mock_get, tmp_path):
        from src.data.wikipedia import download_wikipedia_dump

        dest = tmp_path / "enwiki-20230601-pages-articles.xml.bz2"
        dest.write_text("cached")

        result = download_wikipedia_dump("20230601", output_dir=tmp_path)

        mock_get.assert_not_called()
        assert result == dest

    @patch("src.data.wikipedia.requests.get")
    def test_download_failure_raises(self, mock_get, tmp_path):
        from src.data.wikipedia import download_wikipedia_dump

        mock_get.side_effect = requests.ConnectionError("network down")

        with pytest.raises(RuntimeError, match="Failed to download"):
            download_wikipedia_dump("20230601", output_dir=tmp_path)


# ===================================================================
# wikipedia.py — _save_documents_parquet / _dataframe_to_documents
# ===================================================================


class TestParquetRoundTrip:
    """Test serialization/deserialization of Documents to/from Parquet."""

    def test_roundtrip(self, tmp_path):
        from src.data.wikipedia import _save_documents_parquet, _dataframe_to_documents
        import pandas as pd

        docs = [
            Document(
                doc_id="wiki-rt1", text="Round-trip article one." * 30,
                source="wikipedia", timestamp=datetime(2023, 3, 15),
                url="https://en.wikipedia.org/wiki/RoundTrip1",
                metadata={"title": "RoundTrip1", "page_id": "10", "is_disambiguation": False},
            ),
            Document(
                doc_id="wiki-rt2", text="Round-trip article two." * 30,
                source="wikipedia", timestamp=datetime(2021, 9, 12),
                url="https://en.wikipedia.org/wiki/RoundTrip2",
                metadata={"title": "RoundTrip2", "page_id": "20", "is_disambiguation": True},
            ),
        ]

        path = tmp_path / "test.parquet"
        _save_documents_parquet(docs, path)
        assert path.exists()

        df = pd.read_parquet(path)
        restored = _dataframe_to_documents(df)

        assert len(restored) == 2
        assert restored[0].doc_id == "wiki-rt1"
        assert restored[1].doc_id == "wiki-rt2"
        assert restored[0].metadata["title"] == "RoundTrip1"
        assert restored[1].metadata["is_disambiguation"] is True
        assert restored[0].timestamp.year == 2023


# ===================================================================
# sampler.py — _find_crawl_for_bin
# ===================================================================


class TestFindCrawlForBin:
    """Tests for sampler._find_crawl_for_bin."""

    @patch("src.data.sampler._load_crawl_config")
    def test_finds_matching_crawl_year(self, mock_config):
        from src.data.sampler import _find_crawl_for_bin

        mock_config.return_value = {
            "CC-MAIN-2021-21": "2021-05-01",
            "CC-MAIN-2023-23": "2023-06-01",
        }

        result = _find_crawl_for_bin("2023", bin_size="year")
        assert result == "CC-MAIN-2023-23"

    @patch("src.data.sampler._load_crawl_config")
    def test_finds_matching_crawl_half_year(self, mock_config):
        from src.data.sampler import _find_crawl_for_bin

        mock_config.return_value = {
            "CC-MAIN-2023-23": "2023-06-01",
        }

        result = _find_crawl_for_bin("2023-H1", bin_size="half-year")
        assert result == "CC-MAIN-2023-23"

    @patch("src.data.sampler._load_crawl_config")
    def test_finds_matching_crawl_quarter(self, mock_config):
        from src.data.sampler import _find_crawl_for_bin

        mock_config.return_value = {
            "CC-MAIN-2023-23": "2023-06-01",
        }

        # June -> Q2
        result = _find_crawl_for_bin("2023-Q2", bin_size="quarter")
        assert result == "CC-MAIN-2023-23"

    @patch("src.data.sampler._load_crawl_config")
    def test_returns_none_when_no_match(self, mock_config):
        from src.data.sampler import _find_crawl_for_bin

        mock_config.return_value = {
            "CC-MAIN-2023-23": "2023-06-01",
        }

        result = _find_crawl_for_bin("2020", bin_size="year")
        assert result is None


# ===================================================================
# sampler.py — _resolve_wikipedia_dump
# ===================================================================


class TestResolveWikipediaDump:
    """Tests for sampler._resolve_wikipedia_dump."""

    def test_finds_dump_matching_bin(self, tmp_path):
        from src.data.sampler import _resolve_wikipedia_dump

        dump_dir = tmp_path / "wikipedia"
        date_dir = dump_dir / "20230601"
        date_dir.mkdir(parents=True)
        dump_file = date_dir / "enwiki-20230601-pages-articles.xml.bz2"
        dump_file.touch()

        result = _resolve_wikipedia_dump("2023", dump_dir)
        assert result == dump_file

    def test_returns_none_when_no_dir(self, tmp_path):
        from src.data.sampler import _resolve_wikipedia_dump

        result = _resolve_wikipedia_dump("2023", tmp_path / "nonexistent")
        assert result is None

    def test_returns_none_when_no_matching_year(self, tmp_path):
        from src.data.sampler import _resolve_wikipedia_dump

        dump_dir = tmp_path / "wikipedia"
        date_dir = dump_dir / "20210601"
        date_dir.mkdir(parents=True)
        (date_dir / "enwiki-20210601-pages-articles.xml.bz2").touch()

        result = _resolve_wikipedia_dump("2023", dump_dir)
        assert result is None

    def test_finds_plain_xml(self, tmp_path):
        from src.data.sampler import _resolve_wikipedia_dump

        dump_dir = tmp_path / "wikipedia"
        date_dir = dump_dir / "20230601"
        date_dir.mkdir(parents=True)
        dump_file = date_dir / "enwiki-20230601-pages-articles.xml"
        dump_file.touch()

        result = _resolve_wikipedia_dump("2023", dump_dir)
        assert result == dump_file


# ===================================================================
# sampler.py — build_temporal_corpus
# ===================================================================


class TestBuildTemporalCorpus:
    """Tests for build_temporal_corpus."""

    @patch("src.data.wikipedia.sample_wikipedia")
    @patch("src.data.sampler._resolve_wikipedia_dump")
    @patch("src.data.common_crawl.sample_cc_warc")
    @patch("src.data.sampler._find_crawl_for_bin")
    def test_basic_build(
        self, mock_find_crawl, mock_cc_sample, mock_resolve_wiki, mock_wiki_sample
    ):
        from src.data.sampler import build_temporal_corpus

        mock_find_crawl.return_value = "CC-MAIN-2023-23"
        mock_cc_sample.return_value = [
            Document(
                doc_id="cc-1", text="CC doc", source="common_crawl",
                timestamp=datetime(2023, 6, 1), url=None,
            ),
        ]

        mock_resolve_wiki.return_value = Path("/fake/dump.xml.bz2")
        mock_wiki_sample.return_value = [
            Document(
                doc_id="wiki-1", text="Wiki doc", source="wikipedia",
                timestamp=datetime(2023, 3, 15), url=None,
            ),
        ]

        result = build_temporal_corpus(
            sources=["common_crawl", "wikipedia"],
            n_per_bin=1,
            bins=["2023"],
            seed=42,
            bin_size="year",
        )

        assert "2023" in result
        assert len(result["2023"]) == 2
        sources = {d.source for d in result["2023"]}
        assert sources == {"common_crawl", "wikipedia"}

    @patch("src.data.common_crawl.sample_cc_warc")
    @patch("src.data.sampler._find_crawl_for_bin")
    def test_missing_crawl_skips_bin_source(self, mock_find_crawl, mock_cc_sample):
        from src.data.sampler import build_temporal_corpus

        mock_find_crawl.return_value = None  # No crawl found

        result = build_temporal_corpus(
            sources=["common_crawl"],
            n_per_bin=5,
            bins=["2020"],
            seed=42,
        )

        assert "2020" in result
        assert len(result["2020"]) == 0
        mock_cc_sample.assert_not_called()

    @patch("src.data.common_crawl.sample_cc_warc")
    @patch("src.data.sampler._find_crawl_for_bin")
    def test_multiple_bins(self, mock_find_crawl, mock_cc_sample):
        from src.data.sampler import build_temporal_corpus

        def find_crawl_side_effect(bin_label, bin_size="year"):
            mapping = {"2021": "CC-MAIN-2021-21", "2023": "CC-MAIN-2023-23"}
            return mapping.get(bin_label)

        mock_find_crawl.side_effect = find_crawl_side_effect

        def cc_sample_side_effect(crawl_id, n, seed, languages=None):
            return [
                Document(
                    doc_id=f"cc-{crawl_id}-{i}",
                    text=f"Doc from {crawl_id}",
                    source="common_crawl",
                    timestamp=datetime(2023, 1, 1),
                    url=None,
                )
                for i in range(n)
            ]

        mock_cc_sample.side_effect = cc_sample_side_effect

        result = build_temporal_corpus(
            sources=["common_crawl"],
            n_per_bin=2,
            bins=["2021", "2023"],
            seed=42,
        )

        assert len(result) == 2
        assert len(result["2021"]) == 2
        assert len(result["2023"]) == 2

    @patch("src.data.common_crawl.sample_cc_warc")
    @patch("src.data.sampler._find_crawl_for_bin")
    def test_deterministic_with_fixed_seed(self, mock_find_crawl, mock_cc_sample):
        from src.data.sampler import build_temporal_corpus

        mock_find_crawl.return_value = "CC-MAIN-2023-23"

        call_seeds = []

        def capture_seed(crawl_id, n, seed, languages=None):
            call_seeds.append(seed)
            return [
                Document(
                    doc_id=f"cc-{seed}", text="t", source="common_crawl",
                    timestamp=datetime(2023, 1, 1), url=None,
                )
            ]

        mock_cc_sample.side_effect = capture_seed

        build_temporal_corpus(
            sources=["common_crawl"], n_per_bin=1, bins=["2023"], seed=42,
        )
        first_seeds = list(call_seeds)

        call_seeds.clear()
        build_temporal_corpus(
            sources=["common_crawl"], n_per_bin=1, bins=["2023"], seed=42,
        )
        second_seeds = list(call_seeds)

        assert first_seeds == second_seeds

    def test_unknown_source_skipped(self):
        from src.data.sampler import build_temporal_corpus

        result = build_temporal_corpus(
            sources=["unknown_source"],
            n_per_bin=5,
            bins=["2023"],
            seed=42,
        )

        assert "2023" in result
        assert len(result["2023"]) == 0

    @patch("src.data.wikipedia.sample_wikipedia")
    @patch("src.data.sampler._resolve_wikipedia_dump")
    def test_missing_wiki_dump_skips(self, mock_resolve, mock_wiki_sample):
        from src.data.sampler import build_temporal_corpus

        mock_resolve.return_value = None

        result = build_temporal_corpus(
            sources=["wikipedia"],
            n_per_bin=5,
            bins=["2023"],
            seed=42,
        )

        assert len(result["2023"]) == 0
        mock_wiki_sample.assert_not_called()

    @patch("src.data.common_crawl.sample_cc_warc")
    @patch("src.data.sampler._find_crawl_for_bin")
    def test_balanced_sampling_across_bins(self, mock_find_crawl, mock_cc_sample):
        from src.data.sampler import build_temporal_corpus

        mock_find_crawl.side_effect = lambda bl, bs="year": f"CC-{bl}"

        def make_docs(crawl_id, n, seed, languages=None):
            return [
                Document(
                    doc_id=f"{crawl_id}-{i}", text="text", source="common_crawl",
                    timestamp=datetime(2023, 1, 1), url=None,
                )
                for i in range(n)
            ]

        mock_cc_sample.side_effect = make_docs

        result = build_temporal_corpus(
            sources=["common_crawl"],
            n_per_bin=3,
            bins=["2021", "2022", "2023"],
            seed=99,
        )

        # Each bin should have 3 documents.
        for bin_label in ["2021", "2022", "2023"]:
            assert len(result[bin_label]) == 3


# ===================================================================
# Integration-style test: timestamper + Document
# ===================================================================


class TestTimestamperIntegration:
    """Verify assign_time_bin works across a range of months."""

    @pytest.mark.parametrize(
        "month, expected_quarter",
        [
            (1, "Q1"), (2, "Q1"), (3, "Q1"),
            (4, "Q2"), (5, "Q2"), (6, "Q2"),
            (7, "Q3"), (8, "Q3"), (9, "Q3"),
            (10, "Q4"), (11, "Q4"), (12, "Q4"),
        ],
    )
    def test_all_months_to_quarters(self, month, expected_quarter):
        from src.data.timestamper import assign_time_bin

        doc = Document(
            doc_id="m", text="t", source="s",
            timestamp=datetime(2024, month, 15), url=None,
        )
        assert assign_time_bin(doc, "quarter") == f"2024-{expected_quarter}"

    @pytest.mark.parametrize(
        "month, expected_half",
        [
            (1, "H1"), (3, "H1"), (6, "H1"),
            (7, "H2"), (9, "H2"), (12, "H2"),
        ],
    )
    def test_months_to_half_year(self, month, expected_half):
        from src.data.timestamper import assign_time_bin

        doc = Document(
            doc_id="m", text="t", source="s",
            timestamp=datetime(2024, month, 15), url=None,
        )
        assert assign_time_bin(doc, "half-year") == f"2024-{expected_half}"
