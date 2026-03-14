"""Comprehensive integration test for the data-contamination-audit pipeline.

Creates a synthetic test corpus of 100 documents (50 human-like, 50 AI-like),
runs all pipeline steps programmatically, and verifies end-to-end correctness.

All model-dependent components (SentenceTransformer, GPT-2) are mocked to
avoid network access and model downloads.
"""

from __future__ import annotations

import json
import math
import random
import re
import string
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.classifier.features.ensemble import build_feature_matrix
from src.classifier.features.stylometry import compute_stylometric_features
from src.classifier.features.watermark import WatermarkDetector
from src.classifier.model import ContaminationClassifier
from src.data.common_crawl import Document
from src.data.timestamper import assign_time_bin
from src.embeddings.temporal_curves import compute_temporal_curve
from src.reserve.export import export_reserve
from src.reserve.filter import compute_alpha_t, filter_to_reserve
from src.reserve.quality import apply_quality_filters


# ======================================================================
# Synthetic text generators
# ======================================================================

# Rich vocabulary pools for human-like text
_HUMAN_NOUNS = [
    "river", "mountain", "philosophy", "architecture", "thunder", "village",
    "cathedral", "horizon", "manuscript", "twilight", "labyrinth", "quarry",
    "symphony", "peninsula", "meadow", "glacier", "chimney", "butterfly",
    "telescope", "vineyard", "fortress", "archipelago", "canyon", "ember",
    "pilgrimage", "mosaic", "cobblestone", "dragonfly", "hemisphere",
    "avalanche", "tapestry", "blacksmith", "carousel", "whirlpool",
    "constellation", "wildflower", "thunderstorm", "sandcastle", "waterfall",
    "crossroad", "lighthouse", "earthquake", "wilderness", "quicksand",
]

_HUMAN_VERBS = [
    "wandered", "discovered", "illuminated", "whispered", "crumbled",
    "flourished", "echoed", "carved", "shattered", "emerged", "drifted",
    "constructed", "unveiled", "transformed", "ignited", "reclaimed",
    "scattered", "woven", "unearthed", "traversed", "vanished", "kindled",
    "forged", "unraveled", "contemplated", "surrendered", "grasped",
    "demolished", "orchestrated", "envisioned",
]

_HUMAN_ADJECTIVES = [
    "ancient", "peculiar", "magnificent", "treacherous", "ethereal",
    "bewildering", "rugged", "serene", "formidable", "translucent",
    "desolate", "vibrant", "mysterious", "colossal", "fragile",
    "luminous", "uncharted", "elaborate", "pristine", "turbulent",
    "whimsical", "ferocious", "melancholic", "resilient", "arcane",
]

_HUMAN_ADVERBS = [
    "silently", "abruptly", "gracefully", "reluctantly", "fiercely",
    "cautiously", "deliberately", "majestically", "sorrowfully",
    "triumphantly", "haphazardly", "methodically", "wistfully",
]

# Limited vocabulary for AI-like text (repetitive, uniform)
_AI_NOUNS = [
    "system", "process", "approach", "framework", "methodology",
    "implementation", "solution", "technology", "application", "platform",
]

_AI_VERBS = [
    "utilizes", "provides", "enables", "facilitates", "implements",
    "demonstrates", "represents", "incorporates", "leverages", "ensures",
]

_AI_ADJECTIVES = [
    "efficient", "comprehensive", "robust", "innovative", "effective",
    "significant", "advanced", "optimal", "scalable", "reliable",
]


def _generate_human_sentence(rng: random.Random) -> str:
    """Generate a single sentence with natural variation in structure."""
    patterns = [
        # Simple: The ADJ NOUN VERB ADV.
        lambda: (
            f"The {rng.choice(_HUMAN_ADJECTIVES)} {rng.choice(_HUMAN_NOUNS)} "
            f"{rng.choice(_HUMAN_VERBS)} {rng.choice(_HUMAN_ADVERBS)}."
        ),
        # Compound: NOUN VERB, and the NOUN VERB.
        lambda: (
            f"The {rng.choice(_HUMAN_NOUNS)} {rng.choice(_HUMAN_VERBS)}, "
            f"and the {rng.choice(_HUMAN_ADJECTIVES)} {rng.choice(_HUMAN_NOUNS)} "
            f"{rng.choice(_HUMAN_VERBS)} {rng.choice(_HUMAN_ADVERBS)}."
        ),
        # Question-like: How the NOUN VERB remains a ADJ mystery.
        lambda: (
            f"How the {rng.choice(_HUMAN_ADJECTIVES)} {rng.choice(_HUMAN_NOUNS)} "
            f"{rng.choice(_HUMAN_VERBS)} remains a {rng.choice(_HUMAN_ADJECTIVES)} "
            f"mystery to this day."
        ),
        # Descriptive: Across the ADJ NOUN, one could see...
        lambda: (
            f"Across the {rng.choice(_HUMAN_ADJECTIVES)} {rng.choice(_HUMAN_NOUNS)}, "
            f"one could see the {rng.choice(_HUMAN_NOUNS)} that had "
            f"{rng.choice(_HUMAN_VERBS)} over centuries of "
            f"{rng.choice(_HUMAN_ADJECTIVES)} {rng.choice(_HUMAN_NOUNS)}."
        ),
        # Short exclamation
        lambda: (
            f"What a {rng.choice(_HUMAN_ADJECTIVES)} {rng.choice(_HUMAN_NOUNS)}!"
        ),
        # Parenthetical aside
        lambda: (
            f"The {rng.choice(_HUMAN_NOUNS)} — {rng.choice(_HUMAN_ADJECTIVES)} "
            f"and {rng.choice(_HUMAN_ADJECTIVES)} — {rng.choice(_HUMAN_VERBS)} "
            f"before anyone noticed."
        ),
    ]
    return rng.choice(patterns)()


def _generate_ai_sentence(rng: random.Random) -> str:
    """Generate a uniform AI-like sentence: always the same structure/length."""
    # Always use: "The ADJ NOUN VERB a ADJ NOUN for the NOUN."
    return (
        f"The {rng.choice(_AI_ADJECTIVES)} {rng.choice(_AI_NOUNS)} "
        f"{rng.choice(_AI_VERBS)} a {rng.choice(_AI_ADJECTIVES)} "
        f"{rng.choice(_AI_NOUNS)} for the {rng.choice(_AI_NOUNS)}."
    )


def generate_human_text(rng: random.Random, min_sentences: int = 15, max_sentences: int = 40) -> str:
    """Generate human-like text with varied sentence lengths and rich vocabulary.

    Human-like characteristics:
    - High vocabulary richness (type-token ratio)
    - High sentence length standard deviation
    - Varied sentence structures
    - Multiple paragraphs with different lengths
    """
    n_sentences = rng.randint(min_sentences, max_sentences)
    sentences = [_generate_human_sentence(rng) for _ in range(n_sentences)]

    # Group into paragraphs of varying sizes (2-7 sentences)
    paragraphs: list[str] = []
    idx = 0
    while idx < len(sentences):
        remaining = len(sentences) - idx
        if remaining <= 2:
            para_size = remaining
        else:
            para_size = rng.randint(2, min(7, remaining))
        paragraph = " ".join(sentences[idx : idx + para_size])
        paragraphs.append(paragraph)
        idx += para_size

    return "\n\n".join(paragraphs)


def generate_ai_text(rng: random.Random, n_sentences: int = 25) -> str:
    """Generate AI-like text with uniform sentence lengths and repetitive vocabulary.

    AI-like characteristics:
    - Low vocabulary richness (small, repetitive word set)
    - Low sentence length standard deviation (all same length)
    - Formulaic structure
    - Uniform paragraph sizes
    """
    sentences = [_generate_ai_sentence(rng) for _ in range(n_sentences)]

    # Group into uniform paragraphs of exactly 5 sentences
    paragraphs: list[str] = []
    for i in range(0, len(sentences), 5):
        paragraph = " ".join(sentences[i : i + 5])
        paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)


# ======================================================================
# Corpus builder
# ======================================================================


def build_synthetic_corpus(
    n_total: int = 100,
    seed: int = 42,
) -> tuple[list[Document], list[int]]:
    """Create a synthetic corpus with n_total/2 human and n_total/2 AI docs.

    Returns (documents, labels) where labels are 0=human, 1=synthetic.
    """
    rng = random.Random(seed)
    n_human = n_total // 2
    n_ai = n_total - n_human

    documents: list[Document] = []
    labels: list[int] = []

    # Spread documents across multiple time bins
    years = [2019, 2020, 2021, 2022, 2023]

    # Human documents
    for i in range(n_human):
        year = rng.choice(years)
        month = rng.randint(1, 12)
        text = generate_human_text(rng)
        doc = Document(
            doc_id=f"human-{i:04d}",
            text=text,
            source="wikipedia",
            timestamp=datetime(year, month, 15),
            url=f"https://en.wikipedia.org/wiki/Article_{i}",
            metadata={"label": "human"},
        )
        documents.append(doc)
        labels.append(0)

    # AI documents
    for i in range(n_ai):
        year = rng.choice(years)
        month = rng.randint(1, 12)
        text = generate_ai_text(rng)
        doc = Document(
            doc_id=f"ai-{i:04d}",
            text=text,
            source="common_crawl",
            timestamp=datetime(year, month, 15),
            url=f"https://example.com/page_{i}",
            metadata={"label": "ai"},
        )
        documents.append(doc)
        labels.append(1)

    # Shuffle deterministically
    combined = list(zip(documents, labels))
    rng.shuffle(combined)
    documents, labels = zip(*combined)  # type: ignore[assignment]
    return list(documents), list(labels)


# ======================================================================
# Mock helpers
# ======================================================================


class MockPerplexityScorer:
    """Mock PerplexityScorer that returns heuristic perplexity features
    without loading GPT-2.

    Human-like text gets higher perplexity (less predictable).
    AI-like text gets lower perplexity (more predictable).
    """

    def __init__(self, **kwargs):
        self.tokenizer = MockTokenizer()

    def compute_features(self, text: str) -> dict[str, float]:
        """Compute mock perplexity features based on text statistics."""
        words = text.split()
        n_words = len(words)
        unique_words = len(set(w.lower() for w in words))

        # Higher vocabulary richness -> higher perplexity (more human-like)
        richness = unique_words / max(n_words, 1)

        # Sentence length variation
        sentences = re.split(r"[.!?]+\s*", text)
        sentences = [s for s in sentences if s.strip()]
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            std_len = math.sqrt(
                sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            )
        else:
            mean_len = n_words
            std_len = 0.0

        # Base perplexity: human text ~80-200, AI text ~30-60
        base_ppl = 30.0 + richness * 200.0 + std_len * 2.0

        # Add some noise
        noise = hash(text[:50]) % 20 - 10
        base_ppl += noise

        burstiness = std_len / max(mean_len, 1.0) * 0.5

        return {
            "perplexity_mean": max(base_ppl, 10.0),
            "perplexity_std": std_len * 3.0,
            "perplexity_burstiness": burstiness,
        }


class MockTokenizer:
    """Mock tokenizer that provides a simple word-level encode method."""

    def encode(self, text: str) -> list[int]:
        """Encode text to pseudo token IDs."""
        words = text.split()
        return [hash(w) % 50257 for w in words]


class MockEncoder:
    """Mock DocumentEncoder that returns random but deterministic embeddings.

    Human-like docs get embeddings in one region, AI-like docs in another,
    so cosine similarity within each group is higher than across groups.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(
        self,
        documents: list[Document],
        cache_dir: Path | None = None,
    ) -> np.ndarray:
        """Generate mock embeddings based on document content."""
        embeddings = np.zeros((len(documents), self.dim), dtype=np.float32)
        for i, doc in enumerate(documents):
            # Use doc_id as seed for reproducibility
            rng = np.random.RandomState(hash(doc.doc_id) % (2**31))
            base = rng.randn(self.dim).astype(np.float32)

            # Add a class-specific bias so human and AI docs cluster differently
            if doc.doc_id.startswith("human"):
                bias = np.ones(self.dim, dtype=np.float32) * 0.3
            else:
                bias = -np.ones(self.dim, dtype=np.float32) * 0.3
            vec = base + bias

            # L2 normalise
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings[i] = vec

        return embeddings

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode raw texts (for compatibility)."""
        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.RandomState(hash(text[:50]) % (2**31))
            vec = rng.randn(self.dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings[i] = vec
        return embeddings


# ======================================================================
# Integration test
# ======================================================================


class TestIntegrationPipeline:
    """End-to-end integration test for the contamination audit pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path):
        """Build the synthetic corpus and set up temp directories."""
        self.tmp_dir = tmp_path
        self.output_dir = tmp_path / "output"
        self.output_dir.mkdir()
        self.reserve_dir = self.output_dir / "reserve"
        self.reserve_dir.mkdir()

        self.documents, self.labels = build_synthetic_corpus(n_total=100, seed=42)
        self.n_human = sum(1 for l in self.labels if l == 0)
        self.n_ai = sum(1 for l in self.labels if l == 1)
        assert self.n_human == 50
        assert self.n_ai == 50

    def _assign_time_bins(self) -> dict[str, list[Document]]:
        """Assign documents to temporal bins and build a corpus dict."""
        corpus: dict[str, list[Document]] = {}
        for doc in self.documents:
            bin_label = assign_time_bin(doc, bin_size="year")
            doc.metadata["time_bin"] = bin_label
            corpus.setdefault(bin_label, []).append(doc)
        return corpus

    def _build_feature_matrix(self) -> pd.DataFrame:
        """Build feature matrix using mocked perplexity scorer."""
        mock_ppl = MockPerplexityScorer()
        wm_detector = WatermarkDetector()
        mock_tokenizer = MockTokenizer()

        feature_matrix = build_feature_matrix(
            documents=self.documents,
            perplexity_scorer=mock_ppl,
            watermark_detector=wm_detector,
            tokenizer=mock_tokenizer,
        )
        return feature_matrix

    def test_full_pipeline(self):
        """Run all pipeline steps and verify correctness."""
        # ------------------------------------------------------------------
        # Step 1: Assign time bins
        # ------------------------------------------------------------------
        corpus = self._assign_time_bins()

        # Verify all expected years are present
        years_present = set(corpus.keys())
        assert len(years_present) >= 3, (
            f"Expected at least 3 year bins, got {len(years_present)}"
        )

        total_docs_in_corpus = sum(len(v) for v in corpus.values())
        assert total_docs_in_corpus == 100

        # ------------------------------------------------------------------
        # Step 2: Encode documents (mocked)
        # ------------------------------------------------------------------
        mock_encoder = MockEncoder(dim=384)
        embeddings = mock_encoder.encode(self.documents)

        assert embeddings.shape == (100, 384)
        # Verify L2 normalisation
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

        # ------------------------------------------------------------------
        # Step 3: Compute temporal curve (mocked encoder)
        # ------------------------------------------------------------------
        curve_df = compute_temporal_curve(
            corpus, mock_encoder, reference_bin=min(corpus.keys())
        )

        assert len(curve_df) == len(corpus), (
            f"Temporal curve should have {len(corpus)} bins, got {len(curve_df)}"
        )
        assert "bin" in curve_df.columns
        assert "mean_similarity" in curve_df.columns
        assert "cross_similarity_to_reference" in curve_df.columns
        assert "n_documents" in curve_df.columns
        assert "similarity_p25" in curve_df.columns
        assert "similarity_p75" in curve_df.columns

        # All similarity values should be in valid range
        for col in ["mean_similarity", "cross_similarity_to_reference"]:
            values = curve_df[col].values
            assert np.all(values >= -1.0) and np.all(values <= 1.0), (
                f"{col} values out of range: {values}"
            )

        # ------------------------------------------------------------------
        # Step 4: Extract features (mocked perplexity)
        # ------------------------------------------------------------------
        feature_matrix = self._build_feature_matrix()

        assert len(feature_matrix) == 100
        assert "doc_id" in feature_matrix.columns

        # Verify all expected feature columns exist
        expected_features = {
            "perplexity_mean", "perplexity_std", "perplexity_burstiness",
            "watermark_z_score", "watermark_green_fraction",
            "vocabulary_richness", "hapax_ratio",
            "sentence_length_std", "sentence_length_mean",
            "paragraph_length_std", "yules_k",
            "function_word_ratio", "punctuation_ratio",
            "avg_word_length", "conjunction_rate",
            "passive_voice_ratio", "repetition_score",
        }
        actual_features = set(feature_matrix.columns) - {"doc_id"}
        assert expected_features.issubset(actual_features), (
            f"Missing features: {expected_features - actual_features}"
        )

        # ------------------------------------------------------------------
        # Step 5: Train classifier
        # ------------------------------------------------------------------
        labels_series = pd.Series(self.labels)
        classifier = ContaminationClassifier(
            model_type="xgboost",
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
        )
        metrics = classifier.train(feature_matrix, labels_series, val_split=0.2)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auroc" in metrics
        assert "auprc" in metrics

        # Classifier should achieve > 75% accuracy on synthetic data
        assert metrics["accuracy"] > 0.75, (
            f"Classifier accuracy {metrics['accuracy']:.4f} is below 0.75"
        )

        # Save and reload the classifier to test persistence
        model_path = self.output_dir / "classifier.joblib"
        classifier.save(model_path)
        assert model_path.exists()

        loaded_classifier = ContaminationClassifier()
        loaded_classifier.load(model_path)

        # Predictions from loaded model should match
        probas_original = classifier.predict_proba(feature_matrix)
        probas_loaded = loaded_classifier.predict_proba(feature_matrix)
        np.testing.assert_array_almost_equal(probas_original, probas_loaded)

        # ------------------------------------------------------------------
        # Step 6: Filter to reserve
        # ------------------------------------------------------------------
        threshold = 0.5  # Use 0.5 for balanced 50/50 corpus
        reserve_docs = filter_to_reserve(
            documents=self.documents,
            classifier=classifier,
            feature_matrix=feature_matrix,
            threshold=threshold,
        )

        # Reserve should contain mostly human documents
        n_human_in_reserve = sum(
            1 for doc in reserve_docs
            if doc.metadata.get("label") == "human"
        )
        n_ai_in_reserve = sum(
            1 for doc in reserve_docs
            if doc.metadata.get("label") == "ai"
        )

        assert n_human_in_reserve > n_ai_in_reserve, (
            f"Reserve should contain more human docs ({n_human_in_reserve}) "
            f"than AI docs ({n_ai_in_reserve})"
        )

        # Apply quality filters (skip dedup with high threshold to keep more)
        reserve_embeddings = mock_encoder.encode(reserve_docs)
        quality_config = {
            "dedup_threshold": 0.999,  # Very high to avoid dropping unique docs
            "target_lang": "en",
            "min_chars": 50,  # Low threshold for synthetic text
            "max_chars": 500_000,
        }
        reserve_after_quality = apply_quality_filters(
            reserve_docs, reserve_embeddings, quality_config
        )

        # Quality filters should not remove too many docs from the reserve
        assert len(reserve_after_quality) > 0

        # ------------------------------------------------------------------
        # Step 7: Compute alpha_t
        # ------------------------------------------------------------------
        alpha_t = compute_alpha_t(len(self.documents), len(reserve_docs))

        # alpha_t should be approximately 0.5 for balanced 50/50 input
        assert 0.2 <= alpha_t <= 0.8, (
            f"alpha_t = {alpha_t:.4f} is outside the expected range [0.2, 0.8] "
            f"for a 50/50 corpus"
        )

        # ------------------------------------------------------------------
        # Step 8: Export reserve
        # ------------------------------------------------------------------
        export_reserve(
            documents=reserve_after_quality,
            output_dir=self.reserve_dir,
            format="parquet",
            full_corpus_size=len(self.documents),
            threshold=threshold,
        )

        # Verify export artifacts exist
        assert (self.reserve_dir / "reserve.parquet").exists()
        assert (self.reserve_dir / "summary.json").exists()

        # Verify summary content
        with open(self.reserve_dir / "summary.json") as f:
            summary = json.load(f)

        assert summary["total_documents_audited"] == 100
        assert summary["reserve_size"] == len(reserve_after_quality)
        assert summary["threshold"] == threshold
        assert "alpha_t" in summary
        assert "temporal_distribution" in summary
        assert "source_distribution" in summary

        # Verify Parquet content
        reserve_df = pd.read_parquet(self.reserve_dir / "reserve.parquet")
        assert len(reserve_df) == len(reserve_after_quality)
        assert "doc_id" in reserve_df.columns
        assert "text" in reserve_df.columns
        assert "source" in reserve_df.columns

    def test_temporal_curve_bin_count(self):
        """Verify temporal curve has the correct number of bins."""
        corpus = self._assign_time_bins()
        mock_encoder = MockEncoder(dim=384)

        curve_df = compute_temporal_curve(
            corpus, mock_encoder, reference_bin=min(corpus.keys())
        )

        assert len(curve_df) == len(corpus)
        # Each bin should have at least 1 document
        assert all(curve_df["n_documents"] > 0)

    def test_classifier_accuracy_threshold(self):
        """Verify classifier achieves > 75% accuracy on synthetic data."""
        feature_matrix = self._build_feature_matrix()
        labels_series = pd.Series(self.labels)

        classifier = ContaminationClassifier(
            model_type="xgboost",
            n_estimators=100,
            max_depth=4,
        )
        metrics = classifier.train(feature_matrix, labels_series, val_split=0.2)
        assert metrics["accuracy"] > 0.75

    def test_reserve_mostly_human(self):
        """Verify that the reserve contains mostly human documents."""
        feature_matrix = self._build_feature_matrix()
        labels_series = pd.Series(self.labels)

        classifier = ContaminationClassifier(
            model_type="xgboost",
            n_estimators=100,
            max_depth=4,
        )
        classifier.train(feature_matrix, labels_series, val_split=0.2)

        reserve_docs = filter_to_reserve(
            documents=self.documents,
            classifier=classifier,
            feature_matrix=feature_matrix,
            threshold=0.5,
        )

        n_human = sum(
            1 for d in reserve_docs if d.metadata.get("label") == "human"
        )
        n_ai = sum(
            1 for d in reserve_docs if d.metadata.get("label") == "ai"
        )
        assert n_human > n_ai

    def test_alpha_t_approximately_half(self):
        """Verify alpha_t is approximately 0.5 for 50/50 corpus."""
        feature_matrix = self._build_feature_matrix()
        labels_series = pd.Series(self.labels)

        classifier = ContaminationClassifier(
            model_type="xgboost",
            n_estimators=100,
            max_depth=4,
        )
        classifier.train(feature_matrix, labels_series, val_split=0.2)

        reserve_docs = filter_to_reserve(
            documents=self.documents,
            classifier=classifier,
            feature_matrix=feature_matrix,
            threshold=0.5,
        )

        alpha_t = compute_alpha_t(len(self.documents), len(reserve_docs))
        # With 50/50 input and threshold=0.5, alpha_t should be
        # in the neighbourhood of 0.5
        assert 0.2 <= alpha_t <= 0.8, (
            f"alpha_t = {alpha_t:.4f} is outside expected range for 50/50 corpus"
        )

    def test_export_artifacts_exist(self):
        """Verify all export artifacts are created."""
        feature_matrix = self._build_feature_matrix()
        labels_series = pd.Series(self.labels)

        classifier = ContaminationClassifier(
            model_type="xgboost",
            n_estimators=100,
            max_depth=4,
        )
        classifier.train(feature_matrix, labels_series, val_split=0.2)

        reserve_docs = filter_to_reserve(
            documents=self.documents,
            classifier=classifier,
            feature_matrix=feature_matrix,
            threshold=0.5,
        )

        export_dir = self.tmp_dir / "export_test"
        export_reserve(
            documents=reserve_docs,
            output_dir=export_dir,
            format="parquet",
            full_corpus_size=len(self.documents),
            threshold=0.5,
        )

        assert (export_dir / "reserve.parquet").exists()
        assert (export_dir / "summary.json").exists()

        # Verify Parquet is readable and has the right number of rows
        df = pd.read_parquet(export_dir / "reserve.parquet")
        assert len(df) == len(reserve_docs)

        # Verify summary JSON is valid
        with open(export_dir / "summary.json") as f:
            summary = json.load(f)
        assert isinstance(summary, dict)
        assert "alpha_t" in summary
        assert "total_documents_audited" in summary

    def test_stylometric_features_differentiate(self):
        """Verify stylometric features can differentiate human vs AI text."""
        rng = random.Random(99)
        human_features = [
            compute_stylometric_features(generate_human_text(rng))
            for _ in range(20)
        ]
        ai_features = [
            compute_stylometric_features(generate_ai_text(rng))
            for _ in range(20)
        ]

        # Human text should have higher vocabulary richness on average
        human_richness = np.mean([f["vocabulary_richness"] for f in human_features])
        ai_richness = np.mean([f["vocabulary_richness"] for f in ai_features])
        assert human_richness > ai_richness, (
            f"Human vocab richness ({human_richness:.4f}) should be > "
            f"AI vocab richness ({ai_richness:.4f})"
        )

        # Human text should have higher sentence length std
        human_std = np.mean([f["sentence_length_std"] for f in human_features])
        ai_std = np.mean([f["sentence_length_std"] for f in ai_features])
        assert human_std > ai_std, (
            f"Human sentence_length_std ({human_std:.4f}) should be > "
            f"AI sentence_length_std ({ai_std:.4f})"
        )

    def test_synthetic_text_generators(self):
        """Sanity-check the text generators produce distinct text profiles."""
        rng = random.Random(123)

        human_text = generate_human_text(rng)
        ai_text = generate_ai_text(rng)

        # Both should be non-empty
        assert len(human_text) > 100
        assert len(ai_text) > 100

        # Human text should have more unique words relative to total
        human_words = human_text.lower().split()
        ai_words = ai_text.lower().split()
        human_ttr = len(set(human_words)) / len(human_words)
        ai_ttr = len(set(ai_words)) / len(ai_words)
        assert human_ttr > ai_ttr

    def test_pipeline_with_feature_caching(self):
        """Verify feature matrix caching works correctly."""
        cache_dir = self.tmp_dir / "feature_cache"
        cache_dir.mkdir()

        mock_ppl = MockPerplexityScorer()
        wm_detector = WatermarkDetector()
        mock_tokenizer = MockTokenizer()

        # First call: compute and cache
        fm1 = build_feature_matrix(
            documents=self.documents,
            perplexity_scorer=mock_ppl,
            watermark_detector=wm_detector,
            tokenizer=mock_tokenizer,
            cache_dir=cache_dir,
            cache_tag="test",
        )

        # Verify cache file was created
        cache_files = list(cache_dir.glob("test_*.parquet"))
        assert len(cache_files) == 1

        # Second call: should load from cache
        fm2 = build_feature_matrix(
            documents=self.documents,
            perplexity_scorer=mock_ppl,
            watermark_detector=wm_detector,
            tokenizer=mock_tokenizer,
            cache_dir=cache_dir,
            cache_tag="test",
        )

        pd.testing.assert_frame_equal(fm1, fm2)

    def test_classifier_persistence_roundtrip(self):
        """Verify classifier can be saved and loaded with identical results."""
        feature_matrix = self._build_feature_matrix()
        labels_series = pd.Series(self.labels)

        classifier = ContaminationClassifier(
            model_type="xgboost",
            n_estimators=50,
            max_depth=3,
        )
        classifier.train(feature_matrix, labels_series, val_split=0.2)

        # Save
        path = self.tmp_dir / "model_roundtrip.joblib"
        classifier.save(path)

        # Load
        loaded = ContaminationClassifier()
        loaded.load(path)

        # Predictions must match
        p1 = classifier.predict_proba(feature_matrix)
        p2 = loaded.predict_proba(feature_matrix)
        np.testing.assert_array_almost_equal(p1, p2)

        # Feature importance must match
        imp1 = classifier.feature_importance()
        imp2 = loaded.feature_importance()
        pd.testing.assert_frame_equal(imp1, imp2)
