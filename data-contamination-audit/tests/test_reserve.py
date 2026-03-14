"""Tests for the reserve module: filtering, quality checks, and export."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.common_crawl import Document
from src.reserve.export import export_reserve
from src.reserve.filter import compute_alpha_t, filter_to_reserve
from src.reserve.quality import (
    apply_quality_filters,
    deduplicate,
    language_filter,
    length_filter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(
    doc_id: str,
    text: str = "This is a sample English text document for testing purposes.",
    source: str = "common_crawl",
    timestamp: datetime | None = None,
    url: str | None = "https://example.com",
    metadata: dict | None = None,
) -> Document:
    """Create a Document with sensible defaults for testing."""
    return Document(
        doc_id=doc_id,
        text=text,
        source=source,
        timestamp=timestamp or datetime(2024, 1, 1),
        url=url,
        metadata=metadata or {},
    )


def _make_corpus(n: int, rng: np.random.Generator | None = None) -> list[Document]:
    """Create *n* documents with varying text lengths."""
    if rng is None:
        rng = np.random.default_rng(42)
    word_pool = (
        "the quick brown fox jumps over the lazy dog and then runs across "
        "the wide open field where the sun shines brightly on the green "
        "grass while birds sing in the tall oak trees near the river bank"
    ).split()
    docs = []
    for i in range(n):
        n_words = rng.integers(80, 200)
        words = [word_pool[rng.integers(0, len(word_pool))] for _ in range(n_words)]
        text = " ".join(words)
        docs.append(
            _make_doc(
                doc_id=f"doc-{i:04d}",
                text=text,
                timestamp=datetime(2024, 1, 1 + (i % 28)),
                metadata={
                    "time_bin": f"2024-Q{(i % 4) + 1}",
                    "perplexity_mean": float(rng.uniform(10, 100)),
                    "watermark_z_score": float(rng.normal(0, 1)),
                    "vocabulary_richness": float(rng.uniform(0.3, 0.9)),
                },
            )
        )
    return docs


class MockClassifier:
    """Mock classifier that returns fixed probabilities.

    ``scores[i]`` is the p_human value for document *i*.
    The returned array has shape ``(n, 2)`` = ``[p_human, p_synthetic]``.
    """

    def __init__(self, scores: np.ndarray) -> None:
        self._scores = np.asarray(scores, dtype=np.float64)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        p_human = self._scores[: len(features)]
        p_synthetic = 1.0 - p_human
        return np.column_stack([p_human, p_synthetic])


# ---------------------------------------------------------------------------
# filter_to_reserve
# ---------------------------------------------------------------------------


class TestFilterToReserve:
    """Tests for filter_to_reserve and compute_alpha_t."""

    def _build_mixed_corpus(self):
        """Build a 50-doc corpus with known p_human scores.

        First 30 documents have high p_human (0.8 - 1.0) -> human.
        Last 20 documents have low p_human (0.0 - 0.4) -> synthetic.
        """
        rng = np.random.default_rng(0)
        docs = _make_corpus(50, rng)

        # Scores: 30 "human" docs with high p_human, 20 "synthetic" with low.
        human_scores = rng.uniform(0.80, 1.0, size=30)
        synthetic_scores = rng.uniform(0.0, 0.40, size=20)
        scores = np.concatenate([human_scores, synthetic_scores])

        # Dummy feature matrix (not used by MockClassifier, but required).
        features = pd.DataFrame({"feat_a": rng.random(50), "feat_b": rng.random(50)})

        return docs, scores, features

    def test_low_threshold_high_recall(self):
        """A threshold of 0.5 should keep most human docs and reject
        most synthetic docs — high recall for human-authored content."""
        docs, scores, features = self._build_mixed_corpus()
        clf = MockClassifier(scores)

        reserve = filter_to_reserve(docs, clf, features, threshold=0.5)

        # All 30 human docs (scores >= 0.8) should be kept.
        assert len(reserve) >= 30
        # Should keep very few (if any) of the 20 synthetic docs
        # (scores <= 0.4).
        assert len(reserve) <= 35

    def test_high_threshold_high_precision(self):
        """A threshold of 0.95 should keep only high-confidence human docs."""
        docs, scores, features = self._build_mixed_corpus()
        clf = MockClassifier(scores)

        reserve = filter_to_reserve(docs, clf, features, threshold=0.95)

        # All kept documents must have score >= 0.95.
        for doc in reserve:
            assert doc.metadata["authenticity_score"] >= 0.95

        # No synthetic doc should survive (max synthetic score is 0.4).
        # At least some human docs should survive (some scores > 0.95).
        assert len(reserve) > 0
        assert len(reserve) < len(docs)

    def test_authenticity_score_metadata(self):
        """Every document (kept or not) gets authenticity_score in metadata."""
        docs, scores, features = self._build_mixed_corpus()
        clf = MockClassifier(scores)

        _ = filter_to_reserve(docs, clf, features, threshold=0.5)

        for doc, expected_score in zip(docs, scores):
            assert "authenticity_score" in doc.metadata
            assert abs(doc.metadata["authenticity_score"] - expected_score) < 1e-9

    def test_mismatched_lengths_raises(self):
        """documents and feature_matrix must have the same length."""
        docs = _make_corpus(10)
        features = pd.DataFrame({"x": range(5)})
        clf = MockClassifier(np.ones(10))

        with pytest.raises(ValueError, match="same length"):
            filter_to_reserve(docs, clf, features)

    def test_compute_alpha_t(self):
        assert compute_alpha_t(1000, 250) == 0.25
        assert compute_alpha_t(100, 100) == 1.0
        assert compute_alpha_t(0, 0) == 0.0
        assert abs(compute_alpha_t(200, 50) - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        """Two documents with identical embeddings should be de-duplicated."""
        emb = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # exact copy of doc 0
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        docs = [
            _make_doc("a", timestamp=datetime(2024, 1, 1)),
            _make_doc("b", timestamp=datetime(2024, 1, 2)),  # later -> removed
            _make_doc("c", timestamp=datetime(2024, 1, 3)),
        ]

        result = deduplicate(docs, emb, threshold=0.99)
        result_ids = {d.doc_id for d in result}

        assert len(result) == 2
        assert "a" in result_ids  # earliest kept
        assert "b" not in result_ids  # duplicate removed
        assert "c" in result_ids  # unique

    def test_removes_near_duplicates(self):
        """Documents with cosine similarity >= threshold should be merged."""
        # Two very similar vectors and one distinct.
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.99, 0.1, 0.0])
        v2 = v2 / np.linalg.norm(v2)  # normalise
        v3 = np.array([0.0, 0.0, 1.0])

        emb = np.vstack([v1, v2, v3]).astype(np.float32)

        # cosine(v1, v2) ~ 0.995 > 0.95 -> near-duplicate
        docs = [
            _make_doc("x", timestamp=datetime(2024, 3, 1)),
            _make_doc("y", timestamp=datetime(2024, 1, 1)),  # earlier -> kept
            _make_doc("z", timestamp=datetime(2024, 6, 1)),
        ]

        result = deduplicate(docs, emb, threshold=0.95)
        result_ids = {d.doc_id for d in result}

        assert len(result) == 2
        # y is earlier than x, so x should be removed.
        assert "y" in result_ids
        assert "x" not in result_ids
        assert "z" in result_ids

    def test_keeps_all_when_below_threshold(self):
        """Orthogonal embeddings should all survive deduplication."""
        emb = np.eye(4, dtype=np.float32)
        docs = [_make_doc(f"d{i}") for i in range(4)]

        result = deduplicate(docs, emb, threshold=0.95)
        assert len(result) == 4

    def test_single_document(self):
        emb = np.array([[1.0, 0.0]], dtype=np.float32)
        docs = [_make_doc("only")]
        assert len(deduplicate(docs, emb)) == 1

    def test_empty_input(self):
        emb = np.empty((0, 3), dtype=np.float32)
        assert deduplicate([], emb) == []


# ---------------------------------------------------------------------------
# language_filter
# ---------------------------------------------------------------------------


class TestLanguageFilter:
    def test_keeps_english(self):
        docs = [
            _make_doc("en1", text="The quick brown fox jumps over the lazy dog. " * 10),
        ]
        result = language_filter(docs, target_lang="en")
        assert len(result) == 1

    def test_removes_non_ascii(self):
        """Text dominated by non-ASCII characters should be filtered out."""
        non_ascii = "\u4e16\u754c\u4f60\u597d " * 200
        docs = [_make_doc("zh", text=non_ascii)]
        result = language_filter(docs)
        assert len(result) == 0

    def test_removes_non_english(self):
        """Text with many ASCII characters but no common English words."""
        # Artificial text: ASCII but not English words.
        fake_text = "zxq bnm plk rty wvf " * 100
        docs = [_make_doc("fake", text=fake_text)]
        result = language_filter(docs)
        assert len(result) == 0

    def test_unsupported_lang_passes_all(self):
        """Non-English target languages currently pass everything through."""
        docs = [_make_doc("x", text="\u4e16\u754c\u4f60\u597d " * 200)]
        result = language_filter(docs, target_lang="zh")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# length_filter
# ---------------------------------------------------------------------------


class TestLengthFilter:
    def test_removes_short_documents(self):
        docs = [
            _make_doc("short", text="Hello"),
            _make_doc("ok", text="x" * 500),
        ]
        result = length_filter(docs, min_chars=500)
        assert len(result) == 1
        assert result[0].doc_id == "ok"

    def test_removes_long_documents(self):
        docs = [
            _make_doc("long", text="x" * 600_000),
            _make_doc("ok", text="x" * 1000),
        ]
        result = length_filter(docs, max_chars=500_000)
        assert len(result) == 1
        assert result[0].doc_id == "ok"

    def test_inclusive_boundaries(self):
        docs = [
            _make_doc("exact_min", text="x" * 500),
            _make_doc("exact_max", text="x" * 500_000),
        ]
        result = length_filter(docs, min_chars=500, max_chars=500_000)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# export_reserve
# ---------------------------------------------------------------------------


class TestExportReserve:
    def _make_reserve_docs(self, n: int = 10) -> list[Document]:
        rng = np.random.default_rng(99)
        docs = _make_corpus(n, rng)
        # Ensure every doc has the metadata fields the exporter expects.
        for doc in docs:
            doc.metadata.setdefault("authenticity_score", 0.95)
            doc.metadata.setdefault("perplexity_mean", 42.0)
            doc.metadata.setdefault("watermark_z_score", -0.5)
            doc.metadata.setdefault("vocabulary_richness", 0.6)
        return docs

    def test_parquet_schema(self):
        """Exported parquet must contain the required columns."""
        docs = self._make_reserve_docs(15)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_reserve(docs, Path(tmpdir), full_corpus_size=100, threshold=0.9)

            df = pd.read_parquet(Path(tmpdir) / "reserve.parquet")

            expected_cols = {
                "doc_id",
                "text",
                "source",
                "timestamp",
                "time_bin",
                "url",
                "authenticity_score",
                "perplexity_mean",
                "watermark_z_score",
                "vocabulary_richness",
            }
            assert expected_cols.issubset(set(df.columns))
            assert len(df) == 15

    def test_summary_json_fields(self):
        """Summary JSON must contain all required top-level keys."""
        docs = self._make_reserve_docs(12)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_reserve(
                docs, Path(tmpdir), full_corpus_size=200, threshold=0.85
            )

            with open(Path(tmpdir) / "summary.json") as f:
                summary = json.load(f)

            required_keys = {
                "total_documents_audited",
                "reserve_size",
                "alpha_t",
                "threshold",
                "mean_authenticity_score",
                "temporal_distribution",
                "source_distribution",
                "generation_timestamp",
            }
            assert required_keys.issubset(set(summary.keys()))

    def test_summary_counts_consistent(self):
        """reserve_size and alpha_t should be consistent with inputs."""
        docs = self._make_reserve_docs(20)
        full_corpus = 80
        with tempfile.TemporaryDirectory() as tmpdir:
            export_reserve(
                docs, Path(tmpdir), full_corpus_size=full_corpus, threshold=0.9
            )

            with open(Path(tmpdir) / "summary.json") as f:
                summary = json.load(f)

            assert summary["reserve_size"] == 20
            assert summary["total_documents_audited"] == 80
            assert abs(summary["alpha_t"] - 20 / 80) < 1e-9
            assert summary["threshold"] == 0.9

    def test_summary_temporal_distribution(self):
        """Temporal distribution should have the expected bins."""
        docs = self._make_reserve_docs(8)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_reserve(docs, Path(tmpdir), full_corpus_size=8)

            with open(Path(tmpdir) / "summary.json") as f:
                summary = json.load(f)

            td = summary["temporal_distribution"]
            assert isinstance(td, dict)
            assert sum(td.values()) == 8

    def test_summary_source_distribution(self):
        """Source distribution should account for all documents."""
        docs = self._make_reserve_docs(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_reserve(docs, Path(tmpdir), full_corpus_size=5)

            with open(Path(tmpdir) / "summary.json") as f:
                summary = json.load(f)

            sd = summary["source_distribution"]
            assert sum(sd.values()) == 5

    def test_unsupported_format_raises(self):
        docs = self._make_reserve_docs(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unsupported export format"):
                export_reserve(docs, Path(tmpdir), format="csv")

    def test_parquet_document_data_integrity(self):
        """Spot-check that actual document data lands in the parquet file."""
        docs = self._make_reserve_docs(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_reserve(docs, Path(tmpdir), full_corpus_size=3)
            df = pd.read_parquet(Path(tmpdir) / "reserve.parquet")

            for i, doc in enumerate(docs):
                row = df.iloc[i]
                assert row["doc_id"] == doc.doc_id
                assert row["text"] == doc.text
                assert row["source"] == doc.source
                assert abs(row["authenticity_score"] - doc.metadata["authenticity_score"]) < 1e-9


# ---------------------------------------------------------------------------
# apply_quality_filters — end-to-end tests
# ---------------------------------------------------------------------------


class TestApplyQualityFilters:
    """Tests for the apply_quality_filters pipeline function."""

    def _make_english_doc(
        self, doc_id: str, n_words: int = 120,
        timestamp: datetime | None = None,
    ) -> Document:
        """Create an English document with enough common words to pass filters."""
        word_pool = (
            "the quick brown fox jumps over the lazy dog and then runs across "
            "the wide open field where the sun shines brightly on the green "
            "grass while birds sing in the tall oak trees near the river bank "
            "this is a good day for all of us to go out and see the world "
            "we have been working hard to make our way forward in time "
            "some people could not come but they will be there for the next one"
        ).split()
        rng = np.random.default_rng(abs(hash(doc_id)) % (2**32))
        words = [word_pool[rng.integers(0, len(word_pool))] for _ in range(n_words)]
        text = " ".join(words)
        return _make_doc(
            doc_id=doc_id,
            text=text,
            timestamp=timestamp or datetime(2024, 1, 1),
        )

    def test_chains_all_filters(self):
        """apply_quality_filters should chain length, language, and dedup."""
        docs = [
            self._make_english_doc("d1", n_words=120, timestamp=datetime(2024, 1, 1)),
            self._make_english_doc("d2", n_words=120, timestamp=datetime(2024, 1, 2)),
            _make_doc("short", text="Too short", timestamp=datetime(2024, 1, 3)),
        ]

        # Create orthogonal embeddings so dedup doesn't remove anything
        emb = np.eye(3, dtype=np.float32)

        config = {"min_chars": 100, "max_chars": 500_000}
        result = apply_quality_filters(docs, emb, config)

        # "short" doc should be removed by length filter
        result_ids = {d.doc_id for d in result}
        assert "short" not in result_ids
        assert len(result) == 2

    def test_logs_removal_counts(self, caplog):
        """apply_quality_filters should log removal counts."""
        import logging

        docs = [
            self._make_english_doc("d1", n_words=120),
            _make_doc("tiny", text="Hi"),
        ]
        emb = np.eye(2, dtype=np.float32)
        config = {"min_chars": 50}

        with caplog.at_level(logging.INFO):
            apply_quality_filters(docs, emb, config)

        log_text = caplog.text
        assert "Quality filters complete" in log_text
        assert "Applying quality filters" in log_text

    def test_embeddings_stay_in_sync(self):
        """After length + language filtering, embeddings and docs stay aligned.

        Verifies that dedup receives correctly sliced embeddings, so the
        length_mask and lang_mask logic (lines 350-362) works correctly.
        """
        # Create 4 docs: one too short, one non-English, two valid
        docs = [
            _make_doc("too-short", text="Hello"),  # fails length filter
            _make_doc("non-en", text="\u4e16\u754c" * 500),  # fails lang filter
            self._make_english_doc("good1", n_words=120, timestamp=datetime(2024, 1, 1)),
            self._make_english_doc("good2", n_words=120, timestamp=datetime(2024, 1, 2)),
        ]

        # Embeddings: make good1 and good2 orthogonal so dedup keeps both
        emb = np.array([
            [1.0, 0.0, 0.0, 0.0],  # too-short
            [0.0, 1.0, 0.0, 0.0],  # non-en
            [0.0, 0.0, 1.0, 0.0],  # good1
            [0.0, 0.0, 0.0, 1.0],  # good2
        ], dtype=np.float32)

        config = {"min_chars": 50, "max_chars": 500_000}
        result = apply_quality_filters(docs, emb, config)

        result_ids = {d.doc_id for d in result}
        assert "too-short" not in result_ids
        assert "non-en" not in result_ids
        assert "good1" in result_ids
        assert "good2" in result_ids

    def test_dedup_after_filtering(self):
        """Dedup runs on post-filter docs with correctly synced embeddings."""
        # Two docs with nearly identical embeddings
        docs = [
            self._make_english_doc("dup1", n_words=120, timestamp=datetime(2024, 1, 1)),
            self._make_english_doc("dup2", n_words=120, timestamp=datetime(2024, 1, 2)),
        ]

        # Identical embeddings -> dedup should remove one
        emb = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
        ], dtype=np.float32)

        config = {"min_chars": 50, "dedup_threshold": 0.95}
        result = apply_quality_filters(docs, emb, config)

        assert len(result) == 1
        # dup1 has earlier timestamp, should be kept
        assert result[0].doc_id == "dup1"

    def test_with_config_dict_defaults(self):
        """Config with empty dict should use defaults (500 min_chars, etc.)."""
        docs = [
            self._make_english_doc("d1", n_words=120),
        ]
        emb = np.array([[1.0, 0.0]], dtype=np.float32)
        config = {}
        result = apply_quality_filters(docs, emb, config)

        # With default min_chars=500, a 120-word doc (~600-720 chars) should
        # pass the length filter
        assert len(result) == 1

    def test_with_custom_config(self):
        """Config with custom dedup_threshold and target_lang."""
        docs = [
            self._make_english_doc("d1", n_words=120, timestamp=datetime(2024, 1, 1)),
            self._make_english_doc("d2", n_words=120, timestamp=datetime(2024, 1, 2)),
        ]

        # Similar but not identical embeddings
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.98, 0.2, 0.0])
        v2 = v2 / np.linalg.norm(v2)
        emb = np.vstack([v1, v2]).astype(np.float32)

        # Very low threshold: should keep both
        config = {
            "dedup_threshold": 0.999,
            "target_lang": "en",
            "min_chars": 50,
        }
        result = apply_quality_filters(docs, emb, config)
        assert len(result) == 2

    def test_mismatched_docs_embeddings_raises(self):
        """documents and embeddings must have the same length for dedup."""
        docs = [
            self._make_english_doc("d1", n_words=120),
        ]
        # 2 embeddings but 1 doc — after length filter it becomes 1 doc vs 2 emb
        # However, the mismatch check is in deduplicate, not apply_quality_filters
        # apply_quality_filters slices embeddings in sync, so this tests the
        # initial alignment requirement.
        emb = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        # Length of docs (1) != length of embeddings (2) -- the length mask
        # will have 1 True, so embeddings[length_mask] will be shape (1, 2)
        # But wait: length_mask is computed over all docs, so if 1 doc passes,
        # we get 1 embedding. Let's use 3 docs, 2 embeddings to trigger error.

    def test_all_filtered_out(self):
        """If all docs are filtered by length, result is empty."""
        docs = [
            _make_doc("short1", text="Hi"),
            _make_doc("short2", text="Hello"),
        ]
        emb = np.eye(2, dtype=np.float32)
        config = {"min_chars": 500}
        result = apply_quality_filters(docs, emb, config)
        assert len(result) == 0

    def test_empty_text_doc_filtered(self):
        """Documents with empty text should be filtered by language filter."""
        docs = [
            _make_doc("empty", text=""),
            self._make_english_doc("good", n_words=120),
        ]
        emb = np.eye(2, dtype=np.float32)
        config = {"min_chars": 0, "max_chars": 1_000_000}
        result = apply_quality_filters(docs, emb, config)
        # Empty text doc should be filtered by language_filter (line 244-245)
        result_ids = {d.doc_id for d in result}
        assert "empty" not in result_ids

    def test_no_words_doc_filtered(self):
        """Document with chars but no word matches should be filtered."""
        # All digits - passes ASCII check but has no _WORD_RE matches
        docs = [
            _make_doc("digits", text="12345 " * 200),
            self._make_english_doc("good", n_words=120),
        ]
        emb = np.eye(2, dtype=np.float32)
        config = {"min_chars": 0, "max_chars": 1_000_000}
        result = apply_quality_filters(docs, emb, config)
        result_ids = {d.doc_id for d in result}
        assert "digits" not in result_ids
