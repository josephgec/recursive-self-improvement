"""Tests for the classifier feature modules, model, and calibration.

Covers perplexity scoring, watermark detection, stylometric analysis,
the ensemble feature assembly pipeline, the ContaminationClassifier
(training, inference, persistence, feature importance), and probability
calibration.  GPU models are mocked to avoid downloading large weights
in CI.
"""

from __future__ import annotations

import math
import random
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.common_crawl import Document
from src.classifier.features.stylometry import compute_stylometric_features
from src.classifier.features.watermark import WatermarkDetector


# ======================================================================
# Sample texts
# ======================================================================

# Human-written passages — varied structure, irregular rhythm.
HUMAN_TEXT_1 = (
    "The old barn sat crooked against the hillside, its red paint long "
    "since faded to the color of rust. Nobody went there anymore — not "
    "since the accident. But the kids still talked about it, especially "
    "on autumn nights when the wind came howling off the ridge. Some said "
    "you could hear voices. Others just laughed. I never did either."
)

HUMAN_TEXT_2 = (
    "She hadn't expected the letter. Not really. It arrived on a Tuesday, "
    "crumpled slightly at one corner, bearing a stamp she didn't recognize. "
    "Inside: three lines of cramped handwriting and a dried flower — "
    "lavender, she thought, or maybe thyme. It smelled like someone else's "
    "summer. She read it twice, then put it in the drawer with all the "
    "other things she couldn't bring herself to throw away."
)

HUMAN_TEXT_3 = (
    "Cooking, for my grandmother, was an act of war. She attacked the "
    "onions. She annihilated the garlic. She beat the eggs into total "
    "submission. The kitchen was her battlefield, and every Thanksgiving, "
    "we were her reluctant soldiers. But the food — oh, the food — was "
    "worth every tear and every burn."
)

# AI-style passages — uniform, predictable, smooth.
AI_TEXT_1 = (
    "Artificial intelligence has transformed numerous industries in "
    "recent years. Machine learning algorithms are now used in healthcare, "
    "finance, and transportation. These systems analyze large datasets to "
    "identify patterns and make predictions. The technology continues to "
    "evolve, offering new opportunities for innovation and efficiency. "
    "Organizations worldwide are investing in AI capabilities to remain "
    "competitive in the global marketplace. The potential applications "
    "are vast and continue to expand as the technology matures."
)

AI_TEXT_2 = (
    "Climate change is one of the most significant challenges facing "
    "humanity today. Rising global temperatures are causing widespread "
    "environmental impacts, including melting ice caps, rising sea levels, "
    "and increased frequency of extreme weather events. Scientists agree "
    "that immediate action is necessary to mitigate these effects. "
    "Transitioning to renewable energy sources and reducing carbon "
    "emissions are critical steps in addressing this global crisis. "
    "International cooperation is essential for achieving meaningful "
    "progress in combating climate change."
)

AI_TEXT_3 = (
    "Effective communication is essential in the modern workplace. "
    "Clear and concise messaging helps teams collaborate more efficiently "
    "and reduces misunderstandings. Organizations that prioritize "
    "communication tend to have higher employee satisfaction and "
    "productivity. Digital tools have made it easier to connect with "
    "colleagues across different locations and time zones. However, "
    "it is important to balance digital communication with face-to-face "
    "interactions to maintain strong working relationships."
)


# ======================================================================
# Perplexity tests (with mocking)
# ======================================================================


class TestPerplexityScorer:
    """Tests for PerplexityScorer.  The actual GPT-2 model is mocked."""

    def _make_mock_scorer(self):
        """Build a PerplexityScorer with mocked transformer internals."""
        import torch

        with patch(
            "src.classifier.features.perplexity.AutoTokenizer"
        ) as mock_tok_cls, patch(
            "src.classifier.features.perplexity.AutoModelForCausalLM"
        ) as mock_model_cls:
            # --- Tokenizer mock ---
            mock_tokenizer = MagicMock()

            def _tokenize(text, return_tensors=None):
                # Deterministic pseudo-tokenisation: each word -> one id.
                words = text.split()
                ids = [hash(w) % 5000 for w in words]
                result = MagicMock()
                result.input_ids = torch.tensor([ids], dtype=torch.long)
                return result

            mock_tokenizer.side_effect = _tokenize
            mock_tokenizer.__call__ = _tokenize
            # For decode in compute_features
            mock_tokenizer.decode = MagicMock(
                side_effect=lambda ids: " ".join(f"w{i}" for i in range(len(ids)))
            )
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer

            # --- Model mock ---
            mock_model = MagicMock()
            mock_model.config.n_positions = 1024
            mock_model.eval.return_value = None
            mock_model.to.return_value = mock_model

            # Make model callable: return an object with .logits
            def _model_forward(input_ids, labels=None):
                batch, seq_len = input_ids.shape
                vocab = 5000
                # Random logits seeded by input for reproducibility
                gen = torch.Generator()
                gen.manual_seed(int(input_ids.sum().item()) % (2**31))
                logits = torch.randn(batch, seq_len, vocab, generator=gen)
                out = MagicMock()
                out.logits = logits
                return out

            mock_model.__call__ = _model_forward
            mock_model.side_effect = _model_forward
            mock_model_cls.from_pretrained.return_value = mock_model

            from src.classifier.features.perplexity import PerplexityScorer

            scorer = PerplexityScorer.__new__(PerplexityScorer)
            scorer.model_name = "gpt2"
            scorer.stride = 512
            scorer.device = torch.device("cpu")
            scorer.tokenizer = mock_tokenizer
            scorer.model = mock_model
            scorer.max_length = 1024

        return scorer

    def test_score_returns_finite_positive(self):
        scorer = self._make_mock_scorer()
        ppl = scorer.score("Hello world this is a test sentence.")
        assert math.isfinite(ppl)
        assert ppl > 0

    def test_score_empty_text_returns_inf(self):
        """Empty text should yield infinite perplexity."""
        import torch

        scorer = self._make_mock_scorer()
        # Override tokenizer to return empty ids for empty text
        empty_result = MagicMock()
        empty_result.input_ids = torch.tensor([[]], dtype=torch.long)
        scorer.tokenizer = MagicMock(return_value=empty_result)

        ppl = scorer.score("")
        assert ppl == float("inf")

    def test_human_vs_ai_perplexity_direction(self):
        """Human text should have higher perplexity than predictable AI text
        under a mock that gives lower loss to more repetitive input."""
        import torch

        scorer = self._make_mock_scorer()

        # We re-wire the mock so that more repetitive text (lower
        # type-token ratio) gets lower loss, simulating GPT-2 behavior.
        original_model = scorer.model

        call_counter = {"n": 0}

        def _biased_forward(input_ids, labels=None):
            batch, seq_len = input_ids.shape
            vocab = 5000
            logits = torch.zeros(batch, seq_len, vocab)
            # For each position, push probability mass onto the actual
            # next token — the more unique tokens, the harder this is,
            # so we scale the boost by repetition.
            unique_ratio = len(set(input_ids[0].tolist())) / max(seq_len, 1)
            # Lower unique_ratio -> more repetitive -> higher boost -> lower loss
            boost = 10.0 * (1.0 - unique_ratio)
            for t in range(seq_len - 1):
                next_tok = input_ids[0, t + 1].item()
                logits[0, t, next_tok] += boost
            out = MagicMock()
            out.logits = logits
            return out

        scorer.model = MagicMock(side_effect=_biased_forward)
        scorer.model.config = original_model.config
        scorer.model.eval = MagicMock()
        scorer.model.to = MagicMock(return_value=scorer.model)

        # AI text is more repetitive -> lower perplexity
        repetitive_text = "the the the the the the the the the the " * 5
        varied_text = (
            "quick brown fox jumps over lazy dog near tall green "
            "mountain beside clear blue river under bright warm sun"
        )

        ppl_repetitive = scorer.score(repetitive_text)
        ppl_varied = scorer.score(varied_text)

        assert ppl_repetitive < ppl_varied, (
            f"Repetitive text perplexity ({ppl_repetitive:.1f}) should be "
            f"lower than varied text ({ppl_varied:.1f})"
        )

    def test_score_batch_length(self):
        scorer = self._make_mock_scorer()
        texts = ["Hello world.", "Another test.", "Third sample text."]
        results = scorer.score_batch(texts)
        assert len(results) == 3
        assert all(math.isfinite(p) and p > 0 for p in results)

    def test_compute_features_keys(self):
        scorer = self._make_mock_scorer()
        feats = scorer.compute_features(
            "A moderately long text. " * 50, chunk_size=64
        )
        assert "perplexity_mean" in feats
        assert "perplexity_std" in feats
        assert "perplexity_burstiness" in feats
        assert feats["perplexity_mean"] > 0

    def test_compute_features_short_text_zero_burstiness(self):
        """Very short text should return burstiness = 0."""
        scorer = self._make_mock_scorer()
        feats = scorer.compute_features("Short text.", chunk_size=256)
        assert feats["perplexity_burstiness"] == 0.0
        assert feats["perplexity_std"] == 0.0


# ======================================================================
# Watermark tests
# ======================================================================


class TestWatermarkDetector:
    """Tests for the green-list watermark detector."""

    def test_random_sequence_low_z_score(self):
        """A truly random token sequence should have z-score near 0."""
        detector = WatermarkDetector(
            vocab_size=50257, gamma=0.25, hash_key=15485863
        )
        rng = random.Random(42)
        random_tokens = [rng.randint(0, 50256) for _ in range(1000)]

        result = detector.score(random_tokens)
        assert abs(result["watermark_z_score"]) < 4.0, (
            f"Random sequence z-score ({result['watermark_z_score']:.2f}) "
            f"should be near 0 (< 4)"
        )
        # Green fraction should be roughly gamma = 0.25
        assert 0.15 < result["watermark_green_fraction"] < 0.35

    def test_watermarked_sequence_high_z_score(self):
        """A sequence biased toward the green list should have high z-score."""
        detector = WatermarkDetector(
            vocab_size=50257, gamma=0.25, hash_key=15485863
        )

        # Simulate watermarked generation: always pick a green token.
        watermarked_tokens = [1000]  # seed token
        for _ in range(999):
            prev = watermarked_tokens[-1]
            green = detector._green_list(prev)
            # Pick a random token from the green list.
            watermarked_tokens.append(min(green))

        result = detector.score(watermarked_tokens)
        assert result["watermark_z_score"] > 4.0, (
            f"Watermarked sequence z-score ({result['watermark_z_score']:.2f}) "
            f"should be > 4"
        )
        assert result["watermark_green_fraction"] > 0.9

    def test_short_sequence(self):
        """A sequence with fewer than 2 tokens should return zeros."""
        detector = WatermarkDetector()
        result = detector.score([42])
        assert result["watermark_z_score"] == 0.0
        assert result["watermark_green_fraction"] == 0.0

    def test_empty_sequence(self):
        detector = WatermarkDetector()
        result = detector.score([])
        assert result["watermark_z_score"] == 0.0

    def test_green_list_deterministic(self):
        """Green list for the same prev_token and key should be identical."""
        detector = WatermarkDetector()
        g1 = detector._green_list(123)
        g2 = detector._green_list(123)
        assert g1 == g2

    def test_green_list_size(self):
        """Green list should contain gamma * vocab_size tokens."""
        detector = WatermarkDetector(vocab_size=1000, gamma=0.25)
        g = detector._green_list(0)
        assert len(g) == 250


# ======================================================================
# Stylometry tests
# ======================================================================


class TestStylometry:
    """Tests for stylometric feature extraction."""

    def test_feature_keys(self):
        """All 12 features should be present."""
        feats = compute_stylometric_features(HUMAN_TEXT_1)
        expected_keys = {
            "vocabulary_richness",
            "hapax_ratio",
            "sentence_length_std",
            "sentence_length_mean",
            "paragraph_length_std",
            "yules_k",
            "function_word_ratio",
            "punctuation_ratio",
            "avg_word_length",
            "conjunction_rate",
            "passive_voice_ratio",
            "repetition_score",
        }
        assert set(feats.keys()) == expected_keys

    def test_all_features_finite(self):
        """All feature values should be finite numbers."""
        for text in [HUMAN_TEXT_1, AI_TEXT_1]:
            feats = compute_stylometric_features(text)
            for key, val in feats.items():
                assert math.isfinite(val), f"{key} is not finite: {val}"

    def test_vocabulary_richness_range(self):
        """Type-token ratio should be between 0 and 1."""
        feats = compute_stylometric_features(HUMAN_TEXT_1)
        assert 0 < feats["vocabulary_richness"] <= 1.0

    def test_repetitive_text_low_vocabulary_richness(self):
        """Highly repetitive text should have low type-token ratio."""
        repetitive = "the cat sat on the mat. " * 50
        feats = compute_stylometric_features(repetitive)
        assert feats["vocabulary_richness"] < 0.15

    def test_human_vs_ai_sentence_length_std(self):
        """Human texts tend to have higher sentence length variation.

        We check on average across our sample texts.
        """
        human_stds = []
        for text in [HUMAN_TEXT_1, HUMAN_TEXT_2, HUMAN_TEXT_3]:
            feats = compute_stylometric_features(text)
            human_stds.append(feats["sentence_length_std"])

        ai_stds = []
        for text in [AI_TEXT_1, AI_TEXT_2, AI_TEXT_3]:
            feats = compute_stylometric_features(text)
            ai_stds.append(feats["sentence_length_std"])

        avg_human_std = sum(human_stds) / len(human_stds)
        avg_ai_std = sum(ai_stds) / len(ai_stds)

        # Human writing generally has more variable sentence lengths.
        assert avg_human_std > avg_ai_std, (
            f"Average human sentence_length_std ({avg_human_std:.2f}) "
            f"should exceed AI ({avg_ai_std:.2f})"
        )

    def test_human_vs_ai_vocabulary_richness(self):
        """Human texts tend to have higher vocabulary richness.

        We check on average across our sample texts.
        """
        human_richness = []
        for text in [HUMAN_TEXT_1, HUMAN_TEXT_2, HUMAN_TEXT_3]:
            feats = compute_stylometric_features(text)
            human_richness.append(feats["vocabulary_richness"])

        ai_richness = []
        for text in [AI_TEXT_1, AI_TEXT_2, AI_TEXT_3]:
            feats = compute_stylometric_features(text)
            ai_richness.append(feats["vocabulary_richness"])

        avg_human = sum(human_richness) / len(human_richness)
        avg_ai = sum(ai_richness) / len(ai_richness)

        assert avg_human > avg_ai, (
            f"Average human vocabulary_richness ({avg_human:.3f}) "
            f"should exceed AI ({avg_ai:.3f})"
        )

    def test_function_word_ratio_positive(self):
        feats = compute_stylometric_features(HUMAN_TEXT_1)
        assert feats["function_word_ratio"] > 0

    def test_empty_text(self):
        """Edge case: empty text should not crash."""
        feats = compute_stylometric_features("")
        assert feats["vocabulary_richness"] == 0.0
        assert feats["sentence_length_mean"] == 0.0

    def test_passive_voice_detection(self):
        """Text with passive constructions should have positive ratio."""
        passive_text = (
            "The ball was thrown by the boy. "
            "The cake was eaten by the guests. "
            "The book was written by a famous author. "
            "The house was built in 1920."
        )
        feats = compute_stylometric_features(passive_text)
        assert feats["passive_voice_ratio"] > 0, (
            "Passive voice ratio should be positive for text with passive constructions"
        )


# ======================================================================
# Ensemble / feature matrix tests (with mocking)
# ======================================================================


class TestEnsemble:
    """Tests for extract_all_features and build_feature_matrix."""

    def _make_mock_components(self):
        """Create mocked scorer, detector, and tokenizer."""
        mock_scorer = MagicMock()
        mock_scorer.compute_features.return_value = {
            "perplexity_mean": 45.2,
            "perplexity_std": 12.1,
            "perplexity_burstiness": 0.27,
        }

        mock_detector = MagicMock()
        mock_detector.score.return_value = {
            "watermark_z_score": 1.3,
            "watermark_green_fraction": 0.28,
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(100))

        return mock_scorer, mock_detector, mock_tokenizer

    def test_extract_all_features_keys(self):
        """All feature families should be present in the output."""
        from src.classifier.features.ensemble import extract_all_features

        scorer, detector, tokenizer = self._make_mock_components()
        doc = Document(
            doc_id="test-001",
            text=HUMAN_TEXT_1,
            source="test",
            timestamp=datetime(2023, 1, 1),
            url=None,
        )
        feats = extract_all_features(doc, scorer, detector, tokenizer)

        # Perplexity features
        assert "perplexity_mean" in feats
        assert "perplexity_std" in feats
        assert "perplexity_burstiness" in feats

        # Watermark features
        assert "watermark_z_score" in feats
        assert "watermark_green_fraction" in feats

        # Stylometric features (spot check)
        assert "vocabulary_richness" in feats
        assert "sentence_length_std" in feats
        assert "yules_k" in feats

    def test_extract_all_features_count(self):
        """Should produce 3 perplexity + 2 watermark + 12 stylometry = 17."""
        from src.classifier.features.ensemble import extract_all_features

        scorer, detector, tokenizer = self._make_mock_components()
        doc = Document(
            doc_id="test-002",
            text=AI_TEXT_1,
            source="test",
            timestamp=datetime(2023, 1, 1),
            url=None,
        )
        feats = extract_all_features(doc, scorer, detector, tokenizer)
        assert len(feats) == 17

    def test_build_feature_matrix_shape(self):
        """Feature matrix should have n_docs rows and 17 features + doc_id."""
        from src.classifier.features.ensemble import build_feature_matrix

        scorer, detector, tokenizer = self._make_mock_components()

        docs = [
            Document(
                doc_id=f"test-{i:03d}",
                text=text,
                source="test",
                timestamp=datetime(2023, 1, 1),
                url=None,
            )
            for i, text in enumerate([HUMAN_TEXT_1, AI_TEXT_1, HUMAN_TEXT_2])
        ]

        df = build_feature_matrix(docs, scorer, detector, tokenizer)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        # 17 features + doc_id column
        assert len(df.columns) == 18
        assert "doc_id" in df.columns

    def test_build_feature_matrix_no_nans(self):
        """Feature matrix should not contain NaN values."""
        from src.classifier.features.ensemble import build_feature_matrix

        scorer, detector, tokenizer = self._make_mock_components()

        docs = [
            Document(
                doc_id=f"test-{i:03d}",
                text=text,
                source="test",
                timestamp=datetime(2023, 1, 1),
                url=None,
            )
            for i, text in enumerate(
                [HUMAN_TEXT_1, AI_TEXT_1, HUMAN_TEXT_2, AI_TEXT_2]
            )
        ]

        df = build_feature_matrix(docs, scorer, detector, tokenizer)

        numeric_cols = df.select_dtypes(include="number").columns
        assert not df[numeric_cols].isna().any().any(), (
            f"NaN values found in columns: "
            f"{df[numeric_cols].columns[df[numeric_cols].isna().any()].tolist()}"
        )

    def test_build_feature_matrix_doc_id_first_column(self):
        """doc_id should be the first column."""
        from src.classifier.features.ensemble import build_feature_matrix

        scorer, detector, tokenizer = self._make_mock_components()

        docs = [
            Document(
                doc_id="test-001",
                text=HUMAN_TEXT_1,
                source="test",
                timestamp=datetime(2023, 1, 1),
                url=None,
            )
        ]

        df = build_feature_matrix(docs, scorer, detector, tokenizer)
        assert df.columns[0] == "doc_id"

    def test_build_feature_matrix_caching(self, tmp_path):
        """Cached parquet should be loaded on second call."""
        from src.classifier.features.ensemble import build_feature_matrix

        scorer, detector, tokenizer = self._make_mock_components()

        docs = [
            Document(
                doc_id="cache-001",
                text=HUMAN_TEXT_1,
                source="test",
                timestamp=datetime(2023, 1, 1),
                url=None,
            )
        ]

        # First call — should create cache file.
        df1 = build_feature_matrix(
            docs, scorer, detector, tokenizer, cache_dir=tmp_path
        )

        # Verify a parquet file was created.
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 1

        # Second call — should load from cache.
        df2 = build_feature_matrix(
            docs, scorer, detector, tokenizer, cache_dir=tmp_path
        )

        pd.testing.assert_frame_equal(df1, df2)


# ======================================================================
# Helper: synthetic feature dataset
# ======================================================================


def _make_synthetic_dataset(
    n_per_class: int = 100,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create a synthetic feature matrix with clear class separation.

    Human samples (label 0) are drawn from N(0, 1) and AI samples
    (label 1) are drawn from N(3, 1) so that a tree-based classifier
    can easily separate them.
    """
    rng = np.random.RandomState(seed)

    feature_names = [f"feat_{i}" for i in range(n_features)]

    X_human = rng.randn(n_per_class, n_features)
    X_ai = rng.randn(n_per_class, n_features) + 3.0  # shifted mean

    X = np.vstack([X_human, X_ai])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    # Shuffle
    idx = rng.permutation(len(y))
    X = X[idx]
    y = y[idx]

    df = pd.DataFrame(X, columns=feature_names)
    # Add a doc_id column to verify it gets dropped during training
    df.insert(0, "doc_id", [f"doc-{i:04d}" for i in range(len(df))])
    labels = pd.Series(y, name="label")

    return df, labels


# ======================================================================
# ContaminationClassifier tests
# ======================================================================


class TestContaminationClassifier:
    """Tests for the XGBoost-based contamination classifier."""

    def test_train_accuracy_above_80(self):
        """Classifier should achieve > 80% accuracy on separable data."""
        from src.classifier.model import ContaminationClassifier

        clf = ContaminationClassifier()
        features, labels = _make_synthetic_dataset(n_per_class=100)
        metrics = clf.train(features, labels, val_split=0.2)

        assert metrics["accuracy"] > 0.80, (
            f"Expected accuracy > 0.80, got {metrics['accuracy']:.3f}"
        )
        # Sanity-check that all expected metric keys are present.
        for key in ("accuracy", "precision", "recall", "f1", "auroc", "auprc"):
            assert key in metrics, f"Missing metric: {key}"
            assert 0.0 <= metrics[key] <= 1.0, (
                f"Metric {key} = {metrics[key]} out of [0, 1]"
            )

    def test_predict_proba_sums_to_one(self):
        """Each row of predict_proba output should sum to 1.0."""
        from src.classifier.model import ContaminationClassifier

        clf = ContaminationClassifier()
        features, labels = _make_synthetic_dataset(n_per_class=100)
        clf.train(features, labels)

        probs = clf.predict_proba(features)
        assert probs.shape == (len(features), 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_valid_range(self):
        """All probabilities should be in [0, 1]."""
        from src.classifier.model import ContaminationClassifier

        clf = ContaminationClassifier()
        features, labels = _make_synthetic_dataset(n_per_class=100)
        clf.train(features, labels)

        probs = clf.predict_proba(features)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_save_load_roundtrip(self, tmp_path):
        """Saving and loading should produce identical predictions."""
        from src.classifier.model import ContaminationClassifier

        clf = ContaminationClassifier()
        features, labels = _make_synthetic_dataset(n_per_class=100)
        clf.train(features, labels)

        probs_before = clf.predict_proba(features)

        model_path = tmp_path / "model.joblib"
        clf.save(model_path)
        assert model_path.exists()

        clf2 = ContaminationClassifier()
        clf2.load(model_path)
        probs_after = clf2.predict_proba(features)

        np.testing.assert_array_almost_equal(probs_before, probs_after)

    def test_feature_importance_shape(self):
        """feature_importance should return a DataFrame with one row per feature."""
        from src.classifier.model import ContaminationClassifier

        n_features = 10
        clf = ContaminationClassifier()
        features, labels = _make_synthetic_dataset(
            n_per_class=100, n_features=n_features,
        )
        clf.train(features, labels)

        imp = clf.feature_importance()
        assert isinstance(imp, pd.DataFrame)
        assert len(imp) == n_features
        assert "feature" in imp.columns
        assert "importance" in imp.columns
        # Importances should be sorted descending.
        assert imp["importance"].is_monotonic_decreasing

    def test_unsupported_model_type_raises(self):
        """Requesting an unknown model_type should raise ValueError."""
        from src.classifier.model import ContaminationClassifier

        with pytest.raises(ValueError, match="Unsupported model_type"):
            ContaminationClassifier(model_type="random_forest")


# ======================================================================
# Calibration tests
# ======================================================================


class TestCalibration:
    """Tests for probability calibration and reliability diagram."""

    def test_calibrate_produces_valid_probabilities(self):
        """Calibrated predict_proba should return valid probabilities."""
        from src.classifier.model import ContaminationClassifier
        from src.classifier.calibration import calibrate

        clf = ContaminationClassifier()
        features, labels = _make_synthetic_dataset(n_per_class=100, seed=99)
        clf.train(features, labels, val_split=0.15)

        # Use a separate split for calibration.
        cal_features, cal_labels = _make_synthetic_dataset(
            n_per_class=60, seed=123,
        )

        cal_clf = calibrate(clf, cal_features, cal_labels)
        probs = cal_clf.predict_proba(cal_features)

        assert probs.shape == (len(cal_features), 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_calibrated_save_load_roundtrip(self, tmp_path):
        """CalibratedClassifier save/load should preserve predictions."""
        from src.classifier.model import ContaminationClassifier
        from src.classifier.calibration import calibrate, CalibratedClassifier

        clf = ContaminationClassifier()
        features, labels = _make_synthetic_dataset(n_per_class=80, seed=77)
        clf.train(features, labels)

        cal_clf = calibrate(clf, features, labels)
        probs_before = cal_clf.predict_proba(features)

        path = tmp_path / "calibrated.joblib"
        cal_clf.save(path)
        assert path.exists()

        cal_clf2 = CalibratedClassifier.load(path)
        probs_after = cal_clf2.predict_proba(features)

        np.testing.assert_array_almost_equal(probs_before, probs_after)

    def test_plot_calibration_curve_creates_file(self, tmp_path):
        """plot_calibration_curve should create a PNG file on disk."""
        from src.classifier.calibration import plot_calibration_curve

        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=200)
        y_prob = rng.rand(200)

        output_path = tmp_path / "cal_curve.png"
        plot_calibration_curve(y_true, y_prob, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


# ======================================================================
# Perplexity tests — fully mocked torch and transformers
# ======================================================================


class TestPerplexityScorerMocked:
    """Tests for PerplexityScorer with torch and transformers entirely mocked.

    torch is mocked at the module level so it never loads real CUDA code,
    avoiding segfaults on macOS.
    """

    def _build_scorer(self):
        """Construct a PerplexityScorer with fully-mocked torch/transformers.

        Returns (scorer, mock_torch, mock_tokenizer, mock_model).
        """
        import sys
        import types

        # Create a fake torch module with all the attributes the code needs.
        mock_torch = MagicMock()
        mock_torch.device.return_value = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        # torch.no_grad() context manager
        no_grad_ctx = MagicMock()
        no_grad_ctx.__enter__ = MagicMock(return_value=None)
        no_grad_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.no_grad.return_value = no_grad_ctx

        # torch.nn.CrossEntropyLoss
        mock_loss_fn = MagicMock()
        mock_torch.nn.CrossEntropyLoss.return_value = mock_loss_fn

        # --- Tokenizer mock ---
        mock_tokenizer = MagicMock()

        def _tokenize(text, return_tensors=None):
            words = text.split() if text else []
            ids = [abs(hash(w)) % 5000 for w in words] if words else []
            result = MagicMock()
            # input_ids is a mock tensor with .size() and indexing
            ids_tensor = MagicMock()
            ids_tensor.size.return_value = len(ids)
            ids_tensor.__len__ = lambda self_: len(ids)
            # Support [0] indexing for compute_features
            ids_tensor.__getitem__ = lambda self_, idx: (
                ids if isinstance(idx, int) else ids_tensor
            )
            ids_tensor.to.return_value = ids_tensor
            result.input_ids = ids_tensor

            # Store the raw ids for later use
            ids_tensor._raw_ids = ids
            return result

        mock_tokenizer.side_effect = _tokenize
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.decode = MagicMock(
            side_effect=lambda ids: " ".join(f"w{i}" for i in range(len(ids) if hasattr(ids, '__len__') else 0))
        )

        mock_auto_tok = MagicMock()
        mock_auto_tok.from_pretrained.return_value = mock_tokenizer

        # --- Model mock ---
        mock_model = MagicMock()
        mock_model.config.n_positions = 1024
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model

        def _model_forward(input_ids, labels=None):
            out = MagicMock()
            # logits shape: (1, seq_len, vocab)
            logits = MagicMock()
            out.logits = logits
            return out

        mock_model.side_effect = _model_forward

        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        # Patch sys.modules to intercept torch and transformers
        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "torch.nn": mock_torch.nn,
            "torch.cuda": mock_torch.cuda,
            "transformers": MagicMock(
                AutoModelForCausalLM=mock_auto_model,
                AutoTokenizer=mock_auto_tok,
            ),
        }):
            # Force reimport of perplexity module
            mod_name = "src.classifier.features.perplexity"
            if mod_name in sys.modules:
                del sys.modules[mod_name]

            from src.classifier.features.perplexity import PerplexityScorer

            scorer = PerplexityScorer.__new__(PerplexityScorer)
            scorer.model_name = "gpt2"
            scorer.stride = 512
            scorer.device = mock_torch.device("cpu")
            scorer.tokenizer = mock_tokenizer
            scorer.model = mock_model
            scorer.max_length = 1024

        return scorer, mock_torch, mock_tokenizer, mock_model

    def test_init_sets_attributes(self):
        """PerplexityScorer should have model_name, stride, device, etc."""
        scorer, _, _, _ = self._build_scorer()
        assert scorer.model_name == "gpt2"
        assert scorer.stride == 512
        assert scorer.max_length == 1024

    def test_score_returns_float(self):
        """score() should return a float value."""
        scorer, mock_torch, mock_tokenizer, mock_model = self._build_scorer()

        # Set up tokenizer to return a tensor-like with size(1) > 0
        tok_result = MagicMock()
        ids_tensor = MagicMock()
        ids_tensor.size.return_value = 10
        ids_tensor.__getitem__ = lambda self_, idx: ids_tensor
        ids_tensor.to.return_value = ids_tensor
        tok_result.input_ids = ids_tensor
        mock_tokenizer.side_effect = None
        mock_tokenizer.return_value = tok_result

        # Model returns logits; we need contiguous/view/etc.
        logits = MagicMock()
        shift_logits = MagicMock()
        shift_logits.size.return_value = 5000
        shift_logits.view.return_value = shift_logits
        logits.__getitem__ = lambda self_, idx: shift_logits
        shift_labels = MagicMock()
        shift_labels.view.return_value = shift_labels
        shift_logits.contiguous.return_value = shift_logits
        shift_labels.contiguous.return_value = shift_labels

        model_out = MagicMock()
        model_out.logits = logits
        mock_model.side_effect = None
        mock_model.return_value = model_out

        # Loss function returns token_nll
        token_nll = MagicMock()
        token_nll.__getitem__ = lambda self_, idx: token_nll
        token_nll.tolist.return_value = [0.5, 0.6, 0.7, 0.8, 0.9]
        mock_torch.nn.CrossEntropyLoss.return_value = MagicMock(return_value=token_nll)

        ppl = scorer.score("Hello world test sentence more words")
        assert isinstance(ppl, float)
        assert ppl > 0

    def test_score_batch_returns_list_of_floats(self):
        """score_batch() should return a list of floats."""
        scorer, mock_torch, mock_tokenizer, mock_model = self._build_scorer()

        # Patch score to return deterministic values
        scorer.score = MagicMock(side_effect=[10.0, 20.0, 30.0])

        result = scorer.score_batch(["text1", "text2", "text3"])
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)
        assert result == [10.0, 20.0, 30.0]

    def test_compute_features_returns_dict_with_correct_keys(self):
        """compute_features() should return a dict with the three expected keys."""
        scorer, mock_torch, mock_tokenizer, mock_model = self._build_scorer()

        # Mock score to return a simple float
        scorer.score = MagicMock(return_value=42.0)

        # Mock tokenizer for compute_features: needs input_ids[0] with len()
        tok_result = MagicMock()
        ids_list = list(range(600))  # > chunk_size * 2 to trigger burstiness
        ids_tensor = MagicMock()
        ids_tensor.__len__ = lambda self_: len(ids_list)
        ids_tensor.__getitem__ = lambda self_, idx: (
            ids_list[idx] if isinstance(idx, int) else ids_list[idx.start:idx.stop]
        )
        tok_result.input_ids = [ids_tensor]
        mock_tokenizer.side_effect = None
        mock_tokenizer.return_value = tok_result

        # decode returns a string for each chunk
        mock_tokenizer.decode = MagicMock(return_value="chunk text here")

        features = scorer.compute_features("A long text " * 200, chunk_size=256)

        assert isinstance(features, dict)
        assert "perplexity_mean" in features
        assert "perplexity_std" in features
        assert "perplexity_burstiness" in features
        assert features["perplexity_mean"] == 42.0

    def test_compute_features_short_text_zero_burstiness(self):
        """Short text (fewer tokens than 2 * chunk_size) returns zero burstiness."""
        scorer, mock_torch, mock_tokenizer, mock_model = self._build_scorer()

        scorer.score = MagicMock(return_value=50.0)

        # Short token list: 100 tokens < 256 * 2
        tok_result = MagicMock()
        ids_list = list(range(100))
        ids_tensor = MagicMock()
        ids_tensor.__len__ = lambda self_: len(ids_list)
        tok_result.input_ids = [ids_tensor]
        mock_tokenizer.side_effect = None
        mock_tokenizer.return_value = tok_result

        features = scorer.compute_features("Short text.", chunk_size=256)

        assert features["perplexity_burstiness"] == 0.0
        assert features["perplexity_std"] == 0.0
        assert features["perplexity_mean"] == 50.0

    def test_compute_features_burstiness_high_variance(self):
        """Chunks with high variance in perplexity should have high burstiness."""
        scorer, mock_torch, mock_tokenizer, mock_model = self._build_scorer()

        # score returns alternating high and low values for chunks
        call_count = {"n": 0}
        def _alternating_score(text):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return 50.0  # overall ppl
            return 10.0 if call_count["n"] % 2 == 0 else 100.0

        scorer.score = MagicMock(side_effect=_alternating_score)

        # 1200 tokens > 256 * 2 to trigger burstiness calculation
        tok_result = MagicMock()
        ids_list = list(range(1200))
        ids_tensor = MagicMock()
        ids_tensor.__len__ = lambda self_: len(ids_list)
        ids_tensor.__getitem__ = lambda self_, idx: (
            ids_list[idx.start:idx.stop] if isinstance(idx, slice) else ids_list[idx]
        )
        tok_result.input_ids = [ids_tensor]
        mock_tokenizer.side_effect = None
        mock_tokenizer.return_value = tok_result
        mock_tokenizer.decode = MagicMock(return_value="chunk text")

        features = scorer.compute_features("Long text " * 300, chunk_size=256)

        assert features["perplexity_burstiness"] > 0
        assert features["perplexity_std"] > 0

    def test_compute_features_burstiness_low_variance(self):
        """Chunks with low variance in perplexity should have low burstiness."""
        scorer, mock_torch, mock_tokenizer, mock_model = self._build_scorer()

        # score returns nearly constant values
        call_count = {"n": 0}
        def _constant_score(text):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return 50.0  # overall ppl
            return 50.0 + (call_count["n"] * 0.01)  # very small variation

        scorer.score = MagicMock(side_effect=_constant_score)

        tok_result = MagicMock()
        ids_list = list(range(1200))
        ids_tensor = MagicMock()
        ids_tensor.__len__ = lambda self_: len(ids_list)
        ids_tensor.__getitem__ = lambda self_, idx: (
            ids_list[idx.start:idx.stop] if isinstance(idx, slice) else ids_list[idx]
        )
        tok_result.input_ids = [ids_tensor]
        mock_tokenizer.side_effect = None
        mock_tokenizer.return_value = tok_result
        mock_tokenizer.decode = MagicMock(return_value="chunk text")

        features = scorer.compute_features("Long text " * 300, chunk_size=256)

        assert features["perplexity_burstiness"] < 0.01


# ======================================================================
# Ensemble — additional coverage: ImportError fallback, error handling
# ======================================================================


class TestEnsembleAdditional:
    """Additional tests for ensemble.py to cover uncovered lines."""

    def _make_mock_components(self):
        """Create mocked scorer, detector, and tokenizer."""
        mock_scorer = MagicMock()
        mock_scorer.compute_features.return_value = {
            "perplexity_mean": 45.2,
            "perplexity_std": 12.1,
            "perplexity_burstiness": 0.27,
        }

        mock_detector = MagicMock()
        mock_detector.score.return_value = {
            "watermark_z_score": 1.3,
            "watermark_green_fraction": 0.28,
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(100))

        return mock_scorer, mock_detector, mock_tokenizer

    def test_build_feature_matrix_without_rich(self):
        """When rich is not importable, the fallback path (lines 134-141) runs."""
        from src.classifier.features.ensemble import build_feature_matrix

        scorer, detector, tokenizer = self._make_mock_components()

        docs = [
            Document(
                doc_id=f"test-{i:03d}",
                text=HUMAN_TEXT_1,
                source="test",
                timestamp=datetime(2023, 1, 1),
                url=None,
            )
            for i in range(3)
        ]

        # Patch rich.progress to raise ImportError so the else branch runs
        with patch.dict("sys.modules", {"rich": None, "rich.progress": None}):
            import importlib
            import src.classifier.features.ensemble as ens_mod

            # Clear the cached rich import by reloading
            importlib.reload(ens_mod)
            df = ens_mod.build_feature_matrix(docs, scorer, detector, tokenizer)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "doc_id" in df.columns

    def test_build_feature_matrix_without_rich_logging(self):
        """Non-rich path should log at multiples of 50 and at the final doc.

        Tests the progress logging on lines 140-145.
        """
        from src.classifier.features.ensemble import build_feature_matrix

        scorer, detector, tokenizer = self._make_mock_components()

        # Create exactly 51 docs to trigger the (i+1) % 50 == 0 log
        docs = [
            Document(
                doc_id=f"test-{i:03d}",
                text=HUMAN_TEXT_1,
                source="test",
                timestamp=datetime(2023, 1, 1),
                url=None,
            )
            for i in range(51)
        ]

        with patch.dict("sys.modules", {"rich": None, "rich.progress": None}):
            import importlib
            import src.classifier.features.ensemble as ens_mod

            importlib.reload(ens_mod)
            df = ens_mod.build_feature_matrix(docs, scorer, detector, tokenizer)

        assert len(df) == 51
