"""Shared pytest fixtures for the data-contamination-audit test suite."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.data.common_crawl import Document

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
SAMPLE_WIKI_XML = TEST_DATA_DIR / "sample_wiki_articles.xml"
SAMPLE_HTML_DIR = TEST_DATA_DIR / "sample_html_pages"


# ---------------------------------------------------------------------------
# Document fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_document() -> Document:
    """A single, minimal Document for unit tests."""
    return Document(
        doc_id="test-001",
        text="The quick brown fox jumps over the lazy dog. " * 20,
        source="wikipedia",
        timestamp=datetime(2023, 6, 15),
        url="https://en.wikipedia.org/wiki/Test_Article",
        metadata={"title": "Test Article", "page_id": "12345"},
    )


@pytest.fixture
def sample_documents() -> list[Document]:
    """A diverse list of Documents spanning multiple sources and timestamps."""
    return [
        Document(
            doc_id="wiki-aaa111",
            text=(
                "Artificial intelligence (AI) is the simulation of human "
                "intelligence processes by computer systems. These processes "
                "include learning, reasoning, and self-correction. AI has "
                "applications in healthcare, finance, transportation, and "
                "many other industries. Machine learning, a subset of AI, "
                "involves training algorithms on data to make predictions."
            ),
            source="wikipedia",
            timestamp=datetime(2019, 3, 10),
            url="https://en.wikipedia.org/wiki/Artificial_intelligence",
            metadata={"title": "Artificial intelligence", "page_id": "100"},
        ),
        Document(
            doc_id="wiki-bbb222",
            text=(
                "Python is a high-level, general-purpose programming language. "
                "Its design philosophy emphasizes code readability with the "
                "use of significant indentation. Python is dynamically typed "
                "and garbage-collected. It supports multiple programming "
                "paradigms, including structured, object-oriented, and "
                "functional programming."
            ),
            source="wikipedia",
            timestamp=datetime(2021, 7, 22),
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            metadata={"title": "Python (programming language)", "page_id": "200"},
        ),
        Document(
            doc_id="cc-ccc333",
            text=(
                "Today we announce the release of our new product that "
                "revolutionizes the way people interact with technology. "
                "Our team has been working tirelessly for the past two years "
                "to bring this innovative solution to market."
            ),
            source="common_crawl",
            timestamp=datetime(2023, 6, 1),
            url="https://example.com/press-release",
            metadata={"crawl_id": "CC-MAIN-2023-23"},
        ),
        Document(
            doc_id="cc-ddd444",
            text=(
                "Climate change is one of the most pressing issues of our "
                "time. The global average temperature has increased by about "
                "1.1 degrees Celsius since the pre-industrial era. Urgent "
                "action is needed to reduce greenhouse gas emissions and "
                "transition to renewable energy sources."
            ),
            source="common_crawl",
            timestamp=datetime(2024, 5, 1),
            url="https://example.org/climate-report",
            metadata={"crawl_id": "CC-MAIN-2024-22"},
        ),
        Document(
            doc_id="wiki-eee555",
            text=(
                "The Great Wall of China is a series of fortifications that "
                "were built across the historical northern borders of ancient "
                "Chinese states and Imperial China as protection against "
                "various nomadic groups. Several walls were built from as "
                "early as the 7th century BC, with selective stretches later "
                "joined by Qin Shi Huang. Little of the Qin wall remains."
            ),
            source="wikipedia",
            timestamp=datetime(2013, 4, 5),
            url="https://en.wikipedia.org/wiki/Great_Wall_of_China",
            metadata={"title": "Great Wall of China", "page_id": "300"},
        ),
    ]


@pytest.fixture
def stub_document() -> Document:
    """A stub article (fewer than 500 chars) for filtering tests."""
    return Document(
        doc_id="wiki-stub01",
        text="This is a very short stub article about nothing in particular.",
        source="wikipedia",
        timestamp=datetime(2022, 1, 1),
        url="https://en.wikipedia.org/wiki/Stub_Example",
        metadata={
            "title": "Stub Example",
            "page_id": "999",
            "is_disambiguation": False,
        },
    )


@pytest.fixture
def disambiguation_document() -> Document:
    """A disambiguation page for filtering tests."""
    return Document(
        doc_id="wiki-disambig01",
        text=(
            "Mercury may refer to: Mercury (planet), the innermost planet "
            "in the Solar System. Mercury (element), a chemical element. "
            "Mercury (mythology), a Roman deity." + " Extra text." * 100
        ),
        source="wikipedia",
        timestamp=datetime(2022, 1, 1),
        url="https://en.wikipedia.org/wiki/Mercury_(disambiguation)",
        metadata={
            "title": "Mercury (disambiguation)",
            "page_id": "888",
            "is_disambiguation": True,
        },
    )


@pytest.fixture
def sample_wiki_xml_path() -> Path:
    """Path to the sample Wikipedia XML test file."""
    return SAMPLE_WIKI_XML


@pytest.fixture
def sample_html_dir() -> Path:
    """Path to the directory of sample HTML pages."""
    return SAMPLE_HTML_DIR
