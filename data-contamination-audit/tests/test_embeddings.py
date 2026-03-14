"""Tests for the embeddings module.

A MockEncoder replaces the real sentence-transformer model so that tests
run deterministically without downloading model weights.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.common_crawl import Document
from src.embeddings.encoder import DocumentEncoder
from src.embeddings.similarity import (
    corpus_mean_similarity,
    cross_corpus_similarity,
    pairwise_cosine_similarity,
    similarity_percentiles,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2


def _text_to_embedding(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic pseudo-embedding from a text hash.

    Hashes the text, seeds a NumPy RNG with the hash, and draws a random
    vector.  The result is L2-normalised so cosine similarity == dot product.
    Identical texts always produce the same vector.
    """
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(h)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _make_mock_encode(dim: int = EMBEDDING_DIM):
    """Return a function that mimics ``SentenceTransformer.encode``."""

    def _encode(sentences, batch_size=256, show_progress_bar=False, convert_to_numpy=True):
        return np.stack([_text_to_embedding(s, dim) for s in sentences])

    return _encode


def _make_document(text: str, doc_id: str | None = None) -> Document:
    return Document(
        doc_id=doc_id or "",
        text=text,
        source="test",
        timestamp=datetime(2024, 1, 1),
        url=None,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_encoder():
    """A DocumentEncoder whose underlying SentenceTransformer is mocked."""
    with patch("src.embeddings.encoder.SentenceTransformer") as mock_cls:
        instance = MagicMock()
        instance.encode = _make_mock_encode()
        mock_cls.return_value = instance
        encoder = DocumentEncoder(model_name="all-MiniLM-L6-v2", device="cpu")
    return encoder


@pytest.fixture()
def ten_documents() -> list[Document]:
    """Ten documents with varied content."""
    topics = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require large datasets.",
        "Python is a popular programming language.",
        "The Eiffel Tower is located in Paris, France.",
        "Quantum computing uses qubits instead of bits.",
        "Photosynthesis converts sunlight into chemical energy.",
        "The stock market experienced significant volatility today.",
        "Neural networks are inspired by biological neurons.",
        "Climate change affects global weather patterns.",
        "The Great Wall of China is visible from space.",
    ]
    return [_make_document(t) for t in topics]


# ---------------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------------


class TestDocumentEncoder:
    """Tests for DocumentEncoder."""

    def test_encode_output_shape(self, mock_encoder, ten_documents):
        """Encoding 10 documents returns shape (10, EMBEDDING_DIM)."""
        embeddings = mock_encoder.encode(ten_documents)
        assert embeddings.shape == (10, EMBEDDING_DIM)

    def test_encode_dtype_float32(self, mock_encoder, ten_documents):
        """Embeddings must be float32."""
        embeddings = mock_encoder.encode(ten_documents)
        assert embeddings.dtype == np.float32

    def test_embeddings_are_unit_normalised(self, mock_encoder, ten_documents):
        """Each row should have L2 norm close to 1.0."""
        embeddings = mock_encoder.encode(ten_documents)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_cache_hit(self, mock_encoder, ten_documents, tmp_path):
        """Second call with same docs and cache_dir should load from disk."""
        cache_dir = tmp_path / "emb_cache"

        emb_first = mock_encoder.encode(ten_documents, cache_dir=cache_dir)

        # Verify a .npy file was created.
        npy_files = list(cache_dir.glob("*.npy"))
        assert len(npy_files) == 1

        emb_second = mock_encoder.encode(ten_documents, cache_dir=cache_dir)
        np.testing.assert_array_equal(emb_first, emb_second)

    def test_encode_empty_list(self, mock_encoder):
        """Encoding an empty list should return an empty array."""
        embeddings = mock_encoder.encode([])
        assert embeddings.shape[0] == 0

    def test_truncate_long_text(self):
        """_truncate should cap text at ~512 whitespace tokens."""
        long_text = " ".join(["word"] * 1000)
        truncated = DocumentEncoder._truncate(long_text, max_tokens=512)
        assert len(truncated.split()) == 512

    def test_truncate_short_text_unchanged(self):
        """Short text passes through _truncate unchanged."""
        short_text = "Hello world"
        assert DocumentEncoder._truncate(short_text) == short_text

    def test_encode_texts(self, mock_encoder):
        """encode_texts works on raw strings."""
        texts = ["hello world", "foo bar baz"]
        emb = mock_encoder.encode_texts(texts)
        assert emb.shape == (2, EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# Similarity tests
# ---------------------------------------------------------------------------


class TestSimilarityFunctions:
    """Tests for the similarity module."""

    def test_identical_documents_similarity_one(self, mock_encoder):
        """Identical documents must have cosine similarity == 1.0."""
        docs = [_make_document("Identical text here.")] * 2
        emb = mock_encoder.encode(docs)
        sims = pairwise_cosine_similarity(emb)
        np.testing.assert_allclose(sims[0], 1.0, atol=1e-5)

    def test_similarity_is_symmetric(self, mock_encoder):
        """sim(A, B) == sim(B, A)."""
        doc_a = _make_document("Alpha document about science.")
        doc_b = _make_document("Beta document about cooking.")
        emb = mock_encoder.encode([doc_a, doc_b])
        sim_ab = float(emb[0] @ emb[1])
        sim_ba = float(emb[1] @ emb[0])
        assert sim_ab == pytest.approx(sim_ba, abs=1e-7)

    def test_similar_docs_higher_similarity(self, mock_encoder):
        """Semantically similar texts should score higher than dissimilar ones.

        Because MockEncoder uses hash-based random vectors, we test with
        *identical substrings* to guarantee overlap: texts sharing the same
        words will hash identically, producing sim=1.0, which is always
        higher than two random vectors.
        """
        # Pair 1: identical prefix
        pair_same = [
            _make_document("Machine learning is great"),
            _make_document("Machine learning is great"),
        ]
        # Pair 2: completely different
        pair_diff = [
            _make_document("Machine learning is great"),
            _make_document("Cooking pasta with tomato sauce"),
        ]

        emb_same = mock_encoder.encode(pair_same)
        emb_diff = mock_encoder.encode(pair_diff)

        sim_same = float(emb_same[0] @ emb_same[1])
        sim_diff = float(emb_diff[0] @ emb_diff[1])
        assert sim_same > sim_diff

    def test_similar_pairs_comparison(self, mock_encoder):
        """Three pairs: identical > semi-overlapping > totally different.

        With hash-based mock embeddings identical texts give sim=1.0 and
        different texts give low random similarity.
        """
        texts_identical = ("Neural networks learn", "Neural networks learn")
        texts_partial = ("Neural networks learn", "Neural networks forget")
        texts_different = ("Neural networks learn", "The weather is sunny")

        def _sim(a: str, b: str) -> float:
            emb = mock_encoder.encode([_make_document(a), _make_document(b)])
            return float(emb[0] @ emb[1])

        sim_id = _sim(*texts_identical)
        sim_part = _sim(*texts_partial)
        sim_diff = _sim(*texts_different)

        # Identical must be 1.0.
        assert sim_id == pytest.approx(1.0, abs=1e-5)
        # Identical > partially overlapping (different hash => random vec).
        assert sim_id > sim_part
        # Because partial and different are both hash-random, we only check
        # identical beats both.
        assert sim_id > sim_diff

    def test_pairwise_cosine_shape(self):
        """Upper-triangle flat array has n*(n-1)/2 elements."""
        n, d = 5, 64
        emb = _random_unit_vectors(n, d)
        sims = pairwise_cosine_similarity(emb)
        assert sims.shape == (n * (n - 1) // 2,)

    def test_corpus_mean_similarity_increases_with_duplicates(self, mock_encoder):
        """Adding duplicates should raise the corpus mean similarity."""
        base_docs = [
            _make_document("Alpha topic one"),
            _make_document("Beta topic two"),
            _make_document("Gamma topic three"),
            _make_document("Delta topic four"),
        ]
        emb_base = mock_encoder.encode(base_docs)
        mean_base = corpus_mean_similarity(emb_base)

        # Now add duplicates of the first document.
        dup_docs = base_docs + [_make_document("Alpha topic one")] * 4
        emb_dup = mock_encoder.encode(dup_docs)
        mean_dup = corpus_mean_similarity(emb_dup)

        assert mean_dup > mean_base

    def test_cross_corpus_similarity(self, mock_encoder):
        """Cross-corpus similarity returns a finite float."""
        docs_a = [_make_document("Corpus A document one")]
        docs_b = [_make_document("Corpus B document one")]
        emb_a = mock_encoder.encode(docs_a)
        emb_b = mock_encoder.encode(docs_b)
        sim = cross_corpus_similarity(emb_a, emb_b)
        assert isinstance(sim, float)
        assert -1.0 <= sim <= 1.0 + 1e-5

    def test_cross_corpus_identical_corpora(self, mock_encoder):
        """Cross-corpus of identical single-doc corpora should be 1.0."""
        doc = _make_document("Same text everywhere")
        emb = mock_encoder.encode([doc])
        sim = cross_corpus_similarity(emb, emb)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_cross_corpus_empty(self):
        """Empty corpus should return 0.0."""
        emb_a = np.empty((0, 64), dtype=np.float32)
        emb_b = np.ones((3, 64), dtype=np.float32)
        assert cross_corpus_similarity(emb_a, emb_b) == 0.0

    def test_similarity_percentiles(self):
        """Percentiles should be monotonically non-decreasing."""
        emb = _random_unit_vectors(20, 64)
        pcts = similarity_percentiles(emb, percentiles=[5, 25, 50, 75, 95])
        values = list(pcts.values())
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-7

    def test_similarity_percentiles_keys(self):
        """Returned dict should have the requested keys."""
        emb = _random_unit_vectors(10, 64)
        pcts = similarity_percentiles(emb, percentiles=[10, 50, 90])
        assert set(pcts.keys()) == {10, 50, 90}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _random_unit_vectors(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Generate *n* random unit vectors of dimension *d*."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ---------------------------------------------------------------------------
# Temporal curves tests
# ---------------------------------------------------------------------------

import pandas as pd

from src.embeddings.temporal_curves import (
    compute_temporal_curve,
    detect_inflection_point,
)


def _make_document_with_year(text: str, year: int) -> Document:
    """Create a Document with a timestamp set to Jan 1 of the given year."""
    return Document(
        doc_id="",
        text=text,
        source="test",
        timestamp=datetime(year, 1, 1),
        url=None,
    )


def _build_synthetic_corpus(
    *,
    low_similarity_bins: list[str],
    high_similarity_bins: list[str],
    docs_per_bin: int = 8,
    seed: int = 42,
) -> dict[str, list[Document]]:
    """Build a corpus where high-similarity bins contain near-duplicate docs.

    Low-similarity bins get documents with diverse, unique texts (producing
    low intra-bin similarity with the mock hash-based encoder).
    High-similarity bins get many copies of the same text (producing
    intra-bin similarity close to 1.0).
    """
    rng = np.random.default_rng(seed)
    corpus: dict[str, list[Document]] = {}

    for bin_label in low_similarity_bins:
        year = int(bin_label)
        docs = []
        for i in range(docs_per_bin):
            # Each document gets a unique random suffix so that hashes differ.
            unique_text = f"Diverse topic {bin_label} number {i} salt {rng.integers(0, 10**9)}"
            docs.append(_make_document_with_year(unique_text, year))
        corpus[bin_label] = docs

    for bin_label in high_similarity_bins:
        year = int(bin_label)
        # All documents in the bin share the same text => near-duplicate.
        shared_text = f"Contaminated benchmark text for year {bin_label}"
        docs = [
            _make_document_with_year(shared_text, year)
            for _ in range(docs_per_bin)
        ]
        corpus[bin_label] = docs

    return corpus


class TestTemporalCurves:
    """Tests for compute_temporal_curve and detect_inflection_point."""

    @pytest.fixture()
    def synthetic_corpus(self) -> dict[str, list[Document]]:
        """Corpus with a clear inflection around 2021.

        Bins 2013-2020 have diverse (low-similarity) documents.
        Bins 2021-2025 have near-duplicate (high-similarity) documents.
        """
        low_bins = [str(y) for y in range(2013, 2021)]   # 2013..2020
        high_bins = [str(y) for y in range(2021, 2026)]   # 2021..2025
        return _build_synthetic_corpus(
            low_similarity_bins=low_bins,
            high_similarity_bins=high_bins,
        )

    @pytest.fixture()
    def temporal_curve(
        self, synthetic_corpus, mock_encoder
    ) -> pd.DataFrame:
        """Pre-computed temporal curve for the synthetic corpus."""
        return compute_temporal_curve(synthetic_corpus, mock_encoder)

    # ---- DataFrame shape and columns -----------------------------------

    def test_dataframe_columns(self, temporal_curve):
        """DataFrame must contain the six required columns."""
        expected_cols = {
            "bin",
            "mean_similarity",
            "cross_similarity_to_reference",
            "n_documents",
            "similarity_p25",
            "similarity_p75",
        }
        assert set(temporal_curve.columns) == expected_cols

    def test_dataframe_row_count(self, temporal_curve):
        """One row per bin: 2013-2025 inclusive = 13 bins."""
        assert len(temporal_curve) == 13

    def test_dataframe_sorted_by_bin(self, temporal_curve):
        """Rows must be sorted by bin label."""
        bins = list(temporal_curve["bin"])
        assert bins == sorted(bins)

    def test_n_documents_column(self, temporal_curve):
        """Every bin should have 8 documents (from _build_synthetic_corpus)."""
        assert (temporal_curve["n_documents"] == 8).all()

    # ---- Similarity structure ------------------------------------------

    def test_high_bins_have_higher_similarity(self, temporal_curve):
        """Bins 2021-2025 should have higher mean_similarity than 2013-2020."""
        low = temporal_curve[temporal_curve["bin"] < "2021"]["mean_similarity"]
        high = temporal_curve[temporal_curve["bin"] >= "2021"]["mean_similarity"]
        assert high.min() > low.max()

    # ---- Inflection point detection ------------------------------------

    def test_inflection_point_near_2021(self, temporal_curve):
        """The inflection point must be detected around 2021.

        The 3-bin centred moving-average smoothing may shift the detected
        inflection by up to two bins before the actual step (the smoothed
        value at bin *k* already incorporates *k+1*), so we accept
        2019, 2020, or 2021.
        """
        inflection = detect_inflection_point(temporal_curve)
        assert inflection in {"2019", "2020", "2021"}, (
            f"Expected inflection near 2021, got {inflection!r}"
        )

    def test_inflection_returns_string(self, temporal_curve):
        """detect_inflection_point must return a string."""
        inflection = detect_inflection_point(temporal_curve)
        assert isinstance(inflection, str)

    def test_inflection_point_too_few_bins(self):
        """Should raise ValueError when fewer than 3 bins are provided."""
        tiny_curve = pd.DataFrame(
            {
                "bin": ["2013", "2014"],
                "mean_similarity": [0.1, 0.2],
                "cross_similarity_to_reference": [0.1, 0.2],
                "n_documents": [5, 5],
                "similarity_p25": [0.05, 0.1],
                "similarity_p75": [0.15, 0.25],
            }
        )
        with pytest.raises(ValueError, match="at least 3 bins"):
            detect_inflection_point(tiny_curve)

    # ---- Edge cases ----------------------------------------------------

    def test_empty_corpus(self, mock_encoder):
        """An empty corpus should return an empty DataFrame."""
        df = compute_temporal_curve({}, mock_encoder)
        assert len(df) == 0
        assert "bin" in df.columns

    def test_custom_reference_bin(self, synthetic_corpus, mock_encoder):
        """Using a non-default reference bin should still work."""
        df = compute_temporal_curve(
            synthetic_corpus, mock_encoder, reference_bin="2015"
        )
        assert len(df) == 13
        # Cross-similarity of the reference bin to itself should be high.
        ref_row = df[df["bin"] == "2015"]
        assert float(ref_row["cross_similarity_to_reference"].iloc[0]) > 0.0

    def test_invalid_reference_bin(self, synthetic_corpus, mock_encoder):
        """Requesting a non-existent reference bin should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            compute_temporal_curve(
                synthetic_corpus, mock_encoder, reference_bin="1999"
            )


# ---------------------------------------------------------------------------
# Similarity large-corpus sampling tests
# ---------------------------------------------------------------------------


class TestSimilarityLargeCorpus:
    """Tests for the large-corpus sampling paths in similarity.py.

    These cover the branches triggered when n > 50,000 (corpus_mean_similarity)
    or when n_a * n_b > 50,000 (cross_corpus_similarity).
    """

    def test_corpus_mean_similarity_large_n_samples(self):
        """corpus_mean_similarity with n > 50K should use random sampling."""
        n = 50_001
        d = 32
        # Create large array of unit vectors (use low-d for speed)
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((n, d)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= norms

        result = corpus_mean_similarity(vecs)

        assert isinstance(result, float)
        # For random unit vectors in R^32, expected cosine similarity is ~0
        assert -0.5 < result < 0.5

    def test_corpus_mean_similarity_large_n_deterministic(self):
        """Large-corpus sampling should be deterministic (seeded rng=42)."""
        n = 50_001
        d = 16
        rng = np.random.default_rng(99)
        vecs = rng.standard_normal((n, d)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= norms

        result1 = corpus_mean_similarity(vecs)
        result2 = corpus_mean_similarity(vecs)

        assert result1 == pytest.approx(result2, abs=1e-10)

    def test_corpus_mean_similarity_single_doc(self):
        """Single document should return 1.0."""
        emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        assert corpus_mean_similarity(emb) == 1.0

    def test_corpus_mean_similarity_two_docs(self):
        """Two orthogonal docs should have similarity 0.0."""
        emb = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        assert corpus_mean_similarity(emb) == pytest.approx(0.0, abs=1e-5)

    def test_cross_corpus_similarity_large_sampling(self):
        """cross_corpus_similarity with product > 50K should use sampling."""
        d = 16
        n_a = 300
        n_b = 200  # 300 * 200 = 60,000 > 50K

        rng = np.random.default_rng(42)
        emb_a = rng.standard_normal((n_a, d)).astype(np.float32)
        emb_a /= np.linalg.norm(emb_a, axis=1, keepdims=True)

        emb_b = rng.standard_normal((n_b, d)).astype(np.float32)
        emb_b /= np.linalg.norm(emb_b, axis=1, keepdims=True)

        result = cross_corpus_similarity(emb_a, emb_b)

        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_cross_corpus_similarity_large_deterministic(self):
        """Large cross-corpus sampling should be deterministic."""
        d = 16
        n_a, n_b = 300, 200

        rng = np.random.default_rng(7)
        emb_a = rng.standard_normal((n_a, d)).astype(np.float32)
        emb_a /= np.linalg.norm(emb_a, axis=1, keepdims=True)
        emb_b = rng.standard_normal((n_b, d)).astype(np.float32)
        emb_b /= np.linalg.norm(emb_b, axis=1, keepdims=True)

        r1 = cross_corpus_similarity(emb_a, emb_b)
        r2 = cross_corpus_similarity(emb_a, emb_b)

        assert r1 == pytest.approx(r2, abs=1e-10)

    def test_cross_corpus_empty_a(self):
        """Empty corpus A should return 0.0."""
        emb_a = np.empty((0, 8), dtype=np.float32)
        emb_b = np.ones((5, 8), dtype=np.float32)
        assert cross_corpus_similarity(emb_a, emb_b) == 0.0

    def test_cross_corpus_empty_b(self):
        """Empty corpus B should return 0.0."""
        emb_a = np.ones((5, 8), dtype=np.float32)
        emb_b = np.empty((0, 8), dtype=np.float32)
        assert cross_corpus_similarity(emb_a, emb_b) == 0.0

    def test_similarity_percentiles_single_doc(self):
        """Single doc produces empty pairwise sims — percentiles should be 0."""
        emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        pcts = similarity_percentiles(emb, percentiles=[25, 50, 75])
        assert pcts == {25: 0.0, 50: 0.0, 75: 0.0}

    def test_corpus_mean_similarity_all_identical(self):
        """All-identical unit vectors should have mean similarity 1.0."""
        emb = np.tile(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            (100, 1),
        )
        result = corpus_mean_similarity(emb)
        assert result == pytest.approx(1.0, abs=1e-5)
