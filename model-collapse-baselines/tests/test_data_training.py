"""Comprehensive tests for data and training modules.

Covers:
- src/data/tokenization.py
- src/data/real_data.py
- src/data/synthetic.py
- src/training/trainer.py
- src/training/lineage.py
- src/training/checkpointing.py (improving from 73%)

All tests use unittest.mock -- no real model downloads or GPU usage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ===================================================================
# Tokenization tests
# ===================================================================


class TestGetTokenizer:
    """Tests for src.data.tokenization.get_tokenizer."""

    def _call_get_tokenizer(self, model_name, mock_tok):
        """Helper that patches AutoTokenizer.from_pretrained inside the function."""
        with patch("transformers.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_tok
            from src.data.tokenization import get_tokenizer

            result = get_tokenizer(model_name)
            mock_cls.from_pretrained.assert_called_once_with(
                model_name, trust_remote_code=True
            )
        return result

    def test_returns_tokenizer(self):
        """get_tokenizer should return a configured tokenizer."""
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tok.eos_token_id = 2

        result = self._call_get_tokenizer("mock-model", mock_tok)
        assert result is mock_tok

    def test_sets_pad_token_when_none(self):
        """When pad_token is None, it should be set to eos_token."""
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tok.eos_token_id = 2

        result = self._call_get_tokenizer("mock-model", mock_tok)
        assert result.pad_token == "<eos>"
        assert result.pad_token_id == 2

    def test_preserves_existing_pad_token(self):
        """When pad_token already exists, it should not be overwritten."""
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 0
        mock_tok.eos_token = "<eos>"
        mock_tok.eos_token_id = 2

        result = self._call_get_tokenizer("mock-model", mock_tok)
        assert result.pad_token == "<pad>"

    def test_sets_left_padding(self):
        """padding_side should be set to 'left'."""
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"

        result = self._call_get_tokenizer("mock-model", mock_tok)
        assert result.padding_side == "left"


# ===================================================================
# Real data tests
# ===================================================================


class TestCorpusConfig:
    """Tests for src.data.real_data.CorpusConfig."""

    def test_default_values(self):
        from src.data.real_data import CorpusConfig

        cfg = CorpusConfig()
        assert cfg.dataset == "openwebtext"
        assert cfg.split == "train"
        assert cfg.max_documents == 100_000
        assert cfg.max_length == 512
        assert cfg.seed == 42

    def test_custom_values(self):
        from src.data.real_data import CorpusConfig

        cfg = CorpusConfig(
            dataset="c4",
            split="validation",
            max_documents=500,
            max_length=128,
            seed=99,
        )
        assert cfg.dataset == "c4"
        assert cfg.split == "validation"
        assert cfg.max_documents == 500
        assert cfg.max_length == 128
        assert cfg.seed == 99


def _make_real_loader(num_docs=20, vocab_size=100):
    """Helper to create a RealDataLoader with a mocked tokenizer.

    Returns (loader, mock_tokenizer, vocab_size).
    """
    from src.data.real_data import CorpusConfig, RealDataLoader

    config = CorpusConfig(
        dataset="mock_dataset",
        split="train",
        max_documents=num_docs,
        max_length=64,
        seed=42,
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.__len__ = MagicMock(return_value=vocab_size)

    def fake_tokenize(text_list, **kwargs):
        result = {"input_ids": []}
        for t in text_list:
            words = t.split()
            ids = [hash(w) % vocab_size for w in words]
            max_len = kwargs.get("max_length", 64)
            ids = ids[:max_len]
            result["input_ids"].append(ids)
        return result

    mock_tokenizer.side_effect = fake_tokenize

    loader = RealDataLoader(config, mock_tokenizer)
    return loader, mock_tokenizer, vocab_size


class TestRealDataLoader:
    """Tests for src.data.real_data.RealDataLoader."""

    def test_init_does_not_load(self):
        """Initialization should not trigger dataset loading."""
        loader, _, _ = _make_real_loader()
        assert loader._corpus is None

    def test_get_corpus_loads_on_first_call(self):
        """get_corpus should load and tokenize on first call."""
        from datasets import Dataset

        texts = [f"doc {i}" for i in range(10)]
        mock_ds = Dataset.from_dict({"text": texts})

        loader, _, _ = _make_real_loader(num_docs=10)

        with patch("datasets.load_dataset", return_value=mock_ds):
            corpus = loader.get_corpus()

        assert len(corpus) == 10
        assert "text" in corpus.column_names
        assert "input_ids" in corpus.column_names

    def test_get_corpus_caches(self):
        """get_corpus should return the same object on second call."""
        from datasets import Dataset

        texts = [f"doc {i}" for i in range(5)]
        mock_ds = Dataset.from_dict({"text": texts})

        loader, _, _ = _make_real_loader(num_docs=5)

        with patch("datasets.load_dataset", return_value=mock_ds) as mock_ld:
            c1 = loader.get_corpus()
            c2 = loader.get_corpus()

        assert c1 is c2
        assert mock_ld.call_count == 1

    def test_sample_returns_correct_count(self):
        """sample(n) should return exactly n documents."""
        from datasets import Dataset

        texts = [f"doc {i}" for i in range(20)]
        mock_ds = Dataset.from_dict({"text": texts})

        loader, _, _ = _make_real_loader(num_docs=20)

        with patch("datasets.load_dataset", return_value=mock_ds):
            sample = loader.sample(5)

        assert len(sample) == 5

    def test_sample_clamps_to_corpus_size(self):
        """sample(n) should not return more than the corpus size."""
        from datasets import Dataset

        texts = [f"doc {i}" for i in range(10)]
        mock_ds = Dataset.from_dict({"text": texts})

        loader, _, _ = _make_real_loader(num_docs=10)

        with patch("datasets.load_dataset", return_value=mock_ds):
            sample = loader.sample(100)

        assert len(sample) == 10

    def test_sample_deterministic_with_seed(self):
        """Same seed should produce same sample order."""
        from datasets import Dataset

        texts = [f"doc {i}" for i in range(20)]
        mock_ds = Dataset.from_dict({"text": texts})

        loader, _, _ = _make_real_loader(num_docs=20)

        with patch("datasets.load_dataset", return_value=mock_ds):
            s1 = loader.sample(10, seed=123)
            s2 = loader.sample(10, seed=123)

        assert s1["text"] == s2["text"]

    def test_sample_uses_config_seed_as_default(self):
        """sample() without explicit seed should use config seed."""
        from datasets import Dataset

        texts = [f"doc {i}" for i in range(20)]
        mock_ds = Dataset.from_dict({"text": texts})

        loader, _, _ = _make_real_loader(num_docs=20)

        with patch("datasets.load_dataset", return_value=mock_ds):
            s1 = loader.sample(10)
            s2 = loader.sample(10, seed=42)

        assert s1["text"] == s2["text"]

    def test_token_distribution_sums_to_one(self):
        """Token distribution should sum to approximately 1.0."""
        from datasets import Dataset

        texts = [f"word_{i % 5} word_{(i + 1) % 5}" for i in range(20)]
        mock_ds = Dataset.from_dict({"text": texts})

        loader, _, vocab_size = _make_real_loader(num_docs=20)

        with patch("datasets.load_dataset", return_value=mock_ds):
            dist = loader.get_token_distribution()

        assert dist.shape == (vocab_size,)
        assert abs(dist.sum() - 1.0) < 1e-10

    def test_token_distribution_laplace_smoothing(self):
        """Every token should have nonzero probability (Laplace smoothing)."""
        from datasets import Dataset

        texts = ["hello world"] * 5
        mock_ds = Dataset.from_dict({"text": texts})

        loader, _, vocab_size = _make_real_loader(num_docs=5)

        with patch("datasets.load_dataset", return_value=mock_ds):
            dist = loader.get_token_distribution()

        assert np.all(dist > 0)
        assert dist.min() > 0

    def test_token_distribution_is_cached(self):
        """Calling get_token_distribution twice should return the same array."""
        from datasets import Dataset

        texts = ["hello world"] * 5
        mock_ds = Dataset.from_dict({"text": texts})

        loader, _, _ = _make_real_loader(num_docs=5)

        with patch("datasets.load_dataset", return_value=mock_ds):
            d1 = loader.get_token_distribution()
            d2 = loader.get_token_distribution()

        assert d1 is d2

    def test_load_and_tokenize_renames_content_column(self):
        """If dataset has 'content' instead of 'text', it should be renamed."""
        from datasets import Dataset

        mock_ds = Dataset.from_dict(
            {"content": [f"doc {i}" for i in range(5)]}
        )

        loader, _, _ = _make_real_loader(num_docs=5)

        with patch("datasets.load_dataset", return_value=mock_ds):
            corpus = loader.get_corpus()

        assert "text" in corpus.column_names

    def test_load_and_tokenize_removes_extra_columns(self):
        """Extra columns besides 'text' should be removed before tokenizing."""
        from datasets import Dataset

        mock_ds = Dataset.from_dict({
            "text": [f"doc {i}" for i in range(5)],
            "metadata": ["meta"] * 5,
        })

        loader, _, _ = _make_real_loader(num_docs=5)

        with patch("datasets.load_dataset", return_value=mock_ds):
            corpus = loader.get_corpus()

        assert "metadata" not in corpus.column_names

    def test_load_and_tokenize_falls_back_to_first_column(self):
        """If no recognized text column, fall back to first column."""
        from datasets import Dataset

        mock_ds = Dataset.from_dict({
            "body": [f"doc {i}" for i in range(5)],
        })

        loader, _, _ = _make_real_loader(num_docs=5)

        with patch("datasets.load_dataset", return_value=mock_ds):
            corpus = loader.get_corpus()

        assert "text" in corpus.column_names


# ===================================================================
# Synthetic generation tests
# ===================================================================


class TestGenerationConfig:
    """Tests for src.data.synthetic.GenerationConfig."""

    def test_default_values(self):
        from src.data.synthetic import GenerationConfig

        cfg = GenerationConfig()
        assert cfg.num_samples == 50_000
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == 1.0
        assert cfg.top_p == 0.95
        assert cfg.batch_size == 32

    def test_custom_values(self):
        from src.data.synthetic import GenerationConfig

        cfg = GenerationConfig(
            num_samples=100,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            batch_size=8,
        )
        assert cfg.num_samples == 100
        assert cfg.max_new_tokens == 64
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9
        assert cfg.batch_size == 8


def _make_mock_model_and_tokenizer(vocab_size=100):
    """Create mock model + tokenizer for synthetic generation tests."""
    import torch

    mock_model = MagicMock()
    mock_param = torch.zeros(1)
    mock_model.parameters.side_effect = lambda: iter([mock_param])

    def fake_generate(**kwargs):
        input_ids = kwargs["input_ids"]
        bsz = input_ids.shape[0]
        new_tokens = torch.randint(0, vocab_size, (bsz, 8))
        return torch.cat([input_ids, new_tokens], dim=1)

    mock_model.generate = MagicMock(side_effect=fake_generate)

    mock_tokenizer = MagicMock()
    mock_tokenizer.bos_token = "<bos>"
    mock_tokenizer.pad_token_id = 0

    def fake_tokenize(texts, **kwargs):
        import torch as _torch

        bsz = len(texts)
        input_ids = _torch.ones(bsz, 2, dtype=_torch.long)
        attn_mask = _torch.ones(bsz, 2, dtype=_torch.long)
        result = MagicMock()
        result.__getitem__ = lambda self_, k: {
            "input_ids": input_ids, "attention_mask": attn_mask
        }[k]
        result.keys = lambda: ["input_ids", "attention_mask"]

        def to_device(device):
            return result

        result.to = to_device

        # Support ** unpacking for model.generate(**inputs)
        def _iter():
            return iter(["input_ids", "attention_mask"])

        result.__iter__ = _iter
        result.__contains__ = lambda self_, k: k in ("input_ids", "attention_mask")
        return result

    mock_tokenizer.side_effect = fake_tokenize

    def fake_batch_decode(token_ids, **kwargs):
        return [f"generated text {i}" for i in range(token_ids.shape[0])]

    mock_tokenizer.batch_decode = MagicMock(side_effect=fake_batch_decode)

    return mock_model, mock_tokenizer


class TestSyntheticGenerator:
    """Tests for src.data.synthetic.SyntheticGenerator."""

    def test_generate_batch_returns_list_of_strings(self):
        """generate_batch should return a list of strings."""
        from src.data.synthetic import GenerationConfig, SyntheticGenerator

        cfg = GenerationConfig(batch_size=4, max_new_tokens=16)
        model, tokenizer = _make_mock_model_and_tokenizer()
        gen = SyntheticGenerator(model, tokenizer, cfg)

        texts = gen.generate_batch(["hello", "world", "test", "prompt"])

        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts)
        assert len(texts) == 4

    def test_generate_batch_unconditional(self):
        """Empty prompts should be replaced with bos_token."""
        from src.data.synthetic import GenerationConfig, SyntheticGenerator

        cfg = GenerationConfig(batch_size=2, max_new_tokens=16)
        model, tokenizer = _make_mock_model_and_tokenizer()
        gen = SyntheticGenerator(model, tokenizer, cfg)

        texts = gen.generate_batch(["", ""])

        assert len(texts) == 2
        for t in texts:
            assert len(t) > 0

    def test_generate_batch_nonempty_output(self):
        """All generated texts should be non-empty."""
        from src.data.synthetic import GenerationConfig, SyntheticGenerator

        cfg = GenerationConfig(batch_size=3, max_new_tokens=16)
        model, tokenizer = _make_mock_model_and_tokenizer()
        gen = SyntheticGenerator(model, tokenizer, cfg)

        texts = gen.generate_batch(["prompt1", "prompt2", "prompt3"])

        for t in texts:
            assert len(t) > 0

    def test_generate_corpus_returns_dataset(self):
        """generate_corpus should return a Dataset with 'text' column."""
        from src.data.synthetic import GenerationConfig, SyntheticGenerator

        cfg = GenerationConfig(batch_size=4, max_new_tokens=16, num_samples=8)
        model, tokenizer = _make_mock_model_and_tokenizer()
        gen = SyntheticGenerator(model, tokenizer, cfg)

        corpus = gen.generate_corpus(n=8, seed=0)

        assert "text" in corpus.column_names
        assert len(corpus) == 8

    def test_generate_corpus_unconditional(self):
        """Unconditional generation (no prompts) should work."""
        from src.data.synthetic import GenerationConfig, SyntheticGenerator

        cfg = GenerationConfig(batch_size=4, max_new_tokens=16, num_samples=6)
        model, tokenizer = _make_mock_model_and_tokenizer()
        gen = SyntheticGenerator(model, tokenizer, cfg)

        corpus = gen.generate_corpus(n=6, prompts=None, seed=0)

        assert len(corpus) == 6
        for text in corpus["text"]:
            assert len(text) > 0

    def test_generate_corpus_with_prompts(self):
        """Prompt-conditioned generation should cycle prompts."""
        from src.data.synthetic import GenerationConfig, SyntheticGenerator

        cfg = GenerationConfig(batch_size=4, max_new_tokens=16, num_samples=8)
        model, tokenizer = _make_mock_model_and_tokenizer()
        gen = SyntheticGenerator(model, tokenizer, cfg)

        prompts = ["Write about cats.", "Write about dogs."]
        corpus = gen.generate_corpus(n=8, prompts=prompts, seed=0)

        assert len(corpus) == 8

    def test_generate_corpus_default_n(self):
        """When n is None, should use config.num_samples."""
        from src.data.synthetic import GenerationConfig, SyntheticGenerator

        cfg = GenerationConfig(batch_size=4, max_new_tokens=16, num_samples=8)
        model, tokenizer = _make_mock_model_and_tokenizer()
        gen = SyntheticGenerator(model, tokenizer, cfg)

        corpus = gen.generate_corpus(seed=0)

        assert len(corpus) == 8

    def test_generate_corpus_with_empty_prompts_list(self):
        """Empty prompts list should fall back to unconditional."""
        from src.data.synthetic import GenerationConfig, SyntheticGenerator

        cfg = GenerationConfig(batch_size=4, max_new_tokens=16, num_samples=4)
        model, tokenizer = _make_mock_model_and_tokenizer()
        gen = SyntheticGenerator(model, tokenizer, cfg)

        corpus = gen.generate_corpus(n=4, prompts=[], seed=0)

        assert len(corpus) == 4


# ===================================================================
# Trainer tests
# ===================================================================


class TestTrainingConfig:
    """Tests for src.training.trainer.TrainingConfig."""

    def test_default_values(self):
        from src.training.trainer import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.epochs == 1
        assert cfg.batch_size == 8
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.learning_rate == 2e-5
        assert cfg.warmup_ratio == 0.05
        assert cfg.weight_decay == 0.01
        assert cfg.max_steps == -1
        assert cfg.use_lora is False
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == 32
        assert cfg.from_pretrained_each_generation is True

    def test_custom_values(self):
        from src.training.trainer import TrainingConfig

        cfg = TrainingConfig(epochs=3, batch_size=16, use_lora=True, lora_rank=8)
        assert cfg.epochs == 3
        assert cfg.batch_size == 16
        assert cfg.use_lora is True
        assert cfg.lora_rank == 8

    def test_lora_target_modules_default(self):
        from src.training.trainer import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.lora_target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]


class TestGenerationTrainer:
    """Tests for src.training.trainer.GenerationTrainer."""

    def test_init(self):
        """GenerationTrainer should store its parameters."""
        from src.training.trainer import GenerationTrainer, TrainingConfig

        mock_tokenizer = MagicMock()
        cfg = TrainingConfig(epochs=2)
        trainer = GenerationTrainer("mock-model", mock_tokenizer, cfg)

        assert trainer._model_name_or_path == "mock-model"
        assert trainer._tokenizer is mock_tokenizer
        assert trainer._config is cfg
        assert trainer._model is None

    def test_get_model_loads_once(self):
        """get_model should load the model on first call, then cache."""
        from src.training.trainer import GenerationTrainer, TrainingConfig

        mock_model = MagicMock()

        cfg = TrainingConfig(use_lora=False)
        trainer = GenerationTrainer("mock-model", MagicMock(), cfg)

        with patch("transformers.AutoModelForCausalLM") as mock_auto_model:
            mock_auto_model.from_pretrained.return_value = mock_model

            m1 = trainer.get_model()
            m2 = trainer.get_model()

        assert m1 is mock_model
        assert m2 is mock_model
        # from_pretrained should only be called once
        assert mock_auto_model.from_pretrained.call_count == 1

    def test_load_model_basic(self):
        """_load_model should call AutoModelForCausalLM.from_pretrained."""
        from src.training.trainer import GenerationTrainer, TrainingConfig

        mock_model = MagicMock()

        cfg = TrainingConfig(use_lora=False)
        trainer = GenerationTrainer("my-model", MagicMock(), cfg)

        with patch("transformers.AutoModelForCausalLM") as mock_auto_model:
            mock_auto_model.from_pretrained.return_value = mock_model
            result = trainer._load_model()

        mock_auto_model.from_pretrained.assert_called_once_with(
            "my-model", trust_remote_code=True
        )
        assert result is mock_model

    def test_load_model_with_lora(self):
        """When use_lora=True, _apply_lora should wrap the model with LoRA."""
        from src.training.trainer import GenerationTrainer, TrainingConfig

        mock_base_model = MagicMock()
        mock_peft_model = MagicMock()

        param1 = MagicMock()
        param1.requires_grad = True
        param1.numel.return_value = 100
        param2 = MagicMock()
        param2.requires_grad = False
        param2.numel.return_value = 1000
        mock_peft_model.parameters.return_value = [param1, param2]

        cfg = TrainingConfig(use_lora=True, lora_rank=8, lora_alpha=16)
        trainer = GenerationTrainer("my-model", MagicMock(), cfg)

        # Test _apply_lora directly by patching peft imports
        import peft
        with patch.object(peft, "LoraConfig") as mock_lora_config, \
             patch.object(peft, "TaskType") as mock_task_type, \
             patch.object(peft, "get_peft_model", return_value=mock_peft_model):
            result = trainer._apply_lora(mock_base_model)

        assert result is mock_peft_model
        mock_lora_config.assert_called_once()
        # Verify LoRA config params
        call_kwargs = mock_lora_config.call_args[1]
        assert call_kwargs["r"] == 8
        assert call_kwargs["lora_alpha"] == 16

    def test_train_full(self, tmp_path):
        """train() should invoke HuggingFace Trainer and save model."""
        from datasets import Dataset

        from src.training.trainer import GenerationTrainer, TrainingConfig

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        cfg = TrainingConfig(use_lora=False, epochs=1, max_steps=10)
        trainer = GenerationTrainer("mock-model", mock_tokenizer, cfg)

        ds = Dataset.from_dict({
            "text": ["hello world"] * 10,
            "input_ids": [[1, 2, 3]] * 10,
        })

        output_dir = tmp_path / "output"

        with patch("transformers.AutoModelForCausalLM") as mock_auto_model, \
             patch("transformers.Trainer") as mock_trainer_cls, \
             patch("transformers.TrainingArguments"), \
             patch("transformers.DataCollatorForLanguageModeling"):
            mock_auto_model.from_pretrained.return_value = mock_model

            trainer.train(ds, output_dir)

            mock_trainer_cls.return_value.train.assert_called_once()
            mock_model.save_pretrained.assert_called_once()
            mock_tokenizer.save_pretrained.assert_called_once()

    def test_train_tokenizes_if_needed(self, tmp_path):
        """If input_ids are missing, train() should tokenize the dataset."""
        from datasets import Dataset

        from src.training.trainer import GenerationTrainer, TrainingConfig

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        def fake_tokenize(texts, **kwargs):
            return {"input_ids": [[1, 2, 3] for _ in texts]}

        mock_tokenizer.side_effect = fake_tokenize

        cfg = TrainingConfig(use_lora=False, epochs=1, max_steps=10)
        trainer = GenerationTrainer("mock-model", mock_tokenizer, cfg)

        ds = Dataset.from_dict({"text": ["hello world"] * 5})

        output_dir = tmp_path / "output"

        with patch("transformers.AutoModelForCausalLM") as mock_auto_model, \
             patch("transformers.Trainer") as mock_trainer_cls, \
             patch("transformers.TrainingArguments"), \
             patch("transformers.DataCollatorForLanguageModeling"):
            mock_auto_model.from_pretrained.return_value = mock_model
            trainer.train(ds, output_dir)
            mock_trainer_cls.return_value.train.assert_called_once()

    def test_bf16_available_no_cuda(self):
        """_bf16_available should return False when CUDA is not available."""
        from src.training.trainer import GenerationTrainer

        with patch("torch.cuda.is_available", return_value=False):
            assert GenerationTrainer._bf16_available() is False

    def test_bf16_available_with_cuda_and_bf16(self):
        """_bf16_available should return True when CUDA + bf16 are supported."""
        from src.training.trainer import GenerationTrainer

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.is_bf16_supported", return_value=True):
            assert GenerationTrainer._bf16_available() is True

    def test_bf16_available_exception(self):
        """_bf16_available should return False on any exception."""
        from src.training.trainer import GenerationTrainer

        with patch("torch.cuda.is_available", side_effect=RuntimeError("no cuda")):
            assert GenerationTrainer._bf16_available() is False

    def test_train_with_lora_saves_adapter(self, tmp_path):
        """With use_lora=True, train() should save adapter weights."""
        from datasets import Dataset

        from src.training.trainer import GenerationTrainer, TrainingConfig

        mock_model = MagicMock()
        mock_peft_model = MagicMock()
        param = MagicMock()
        param.requires_grad = True
        param.numel.return_value = 100
        mock_peft_model.parameters.return_value = [param]

        cfg = TrainingConfig(use_lora=True, epochs=1, max_steps=5)
        mock_tokenizer = MagicMock()
        trainer = GenerationTrainer("mock-model", mock_tokenizer, cfg)

        ds = Dataset.from_dict({
            "text": ["hello"] * 5,
            "input_ids": [[1, 2, 3]] * 5,
        })

        output_dir = tmp_path / "lora_output"

        with patch("transformers.AutoModelForCausalLM") as mock_auto_model, \
             patch("transformers.Trainer") as mock_trainer_cls, \
             patch("transformers.TrainingArguments"), \
             patch("transformers.DataCollatorForLanguageModeling"), \
             patch("peft.get_peft_model", return_value=mock_peft_model), \
             patch("peft.LoraConfig"), \
             patch("peft.TaskType"):
            mock_auto_model.from_pretrained.return_value = mock_model
            trainer.train(ds, output_dir)
            mock_peft_model.save_pretrained.assert_called_once()

    def test_prepare_dataset_removes_extra_columns(self):
        """_prepare_dataset should remove non-tensor columns."""
        from datasets import Dataset

        from src.training.trainer import GenerationTrainer, TrainingConfig

        cfg = TrainingConfig()
        trainer = GenerationTrainer("mock", MagicMock(), cfg)

        ds = Dataset.from_dict({
            "text": ["hello"] * 5,
            "input_ids": [[1, 2, 3]] * 5,
            "source": ["real"] * 5,
        })

        result = trainer._prepare_dataset(ds)

        assert "input_ids" in result.column_names
        assert "text" not in result.column_names
        assert "source" not in result.column_names


# ===================================================================
# Lineage orchestrator tests
# ===================================================================


def _lineage_config(tmp_path, num_generations=3):
    """Helper to create a minimal config for the orchestrator."""
    return {
        "experiment": {
            "name": "test",
            "seed": 42,
            "num_generations": num_generations,
            "output_dir": str(tmp_path),
        },
        "base_model": "mock-model",
        "real_data": {
            "dataset": "mock",
            "max_documents": 50,
            "max_length": 32,
        },
        "synthetic_generation": {
            "num_samples": 20,
            "max_new_tokens": 16,
            "batch_size": 4,
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-5,
            "max_steps": 5,
            "use_lora": False,
            "from_pretrained_each_generation": True,
        },
        "schedule": {"type": "zero"},
    }


class TestLineageOrchestrator:
    """Tests for src.training.lineage.LineageOrchestrator."""

    def test_init(self, tmp_path):
        """LineageOrchestrator should parse config correctly."""
        from src.training.lineage import LineageOrchestrator

        config = _lineage_config(tmp_path, num_generations=5)
        orch = LineageOrchestrator(config)

        assert orch._base_model_name == "mock-model"
        assert orch._num_generations == 5
        assert orch._seed == 42
        assert orch._fresh_finetune is True

    def test_init_continual_finetune(self, tmp_path):
        """from_pretrained_each_generation=False should set _fresh_finetune=False."""
        from src.training.lineage import LineageOrchestrator

        config = _lineage_config(tmp_path)
        config["training"]["from_pretrained_each_generation"] = False
        orch = LineageOrchestrator(config)

        assert orch._fresh_finetune is False

    def test_init_default_output_dir(self, tmp_path):
        """Default output_dir should be 'data' if not specified."""
        from src.training.lineage import LineageOrchestrator

        config = _lineage_config(tmp_path)
        del config["experiment"]["output_dir"]
        orch = LineageOrchestrator(config)

        assert orch._output_dir == Path("data")

    def test_init_default_seed(self, tmp_path):
        """Default seed should be 42 if not specified."""
        from src.training.lineage import LineageOrchestrator

        config = _lineage_config(tmp_path)
        del config["experiment"]["seed"]
        orch = LineageOrchestrator(config)

        assert orch._seed == 42

    def _setup_run_mocks(self):
        """Set up common mocks for run/run_generation tests."""
        from datasets import Dataset

        mock_tokenizer = MagicMock()
        mock_corpus = Dataset.from_dict({
            "text": ["doc"] * 10, "input_ids": [[1, 2]] * 10
        })
        mock_synth_corpus = Dataset.from_dict({"text": ["synth"] * 10})
        mock_mixed = Dataset.from_dict({
            "text": ["mixed"] * 10, "input_ids": [[1, 2]] * 10
        })
        mock_model = MagicMock()

        return mock_tokenizer, mock_corpus, mock_synth_corpus, mock_mixed, mock_model

    @patch("src.data.mixing.mix_data")
    @patch("src.data.synthetic.SyntheticGenerator")
    @patch("src.data.real_data.RealDataLoader")
    @patch("src.data.tokenization.get_tokenizer")
    @patch("src.training.schedules.schedule_from_config")
    @patch("transformers.AutoModelForCausalLM")
    @patch("src.training.trainer.GenerationTrainer")
    def test_run_executes_all_generations(
        self,
        mock_trainer_cls,
        mock_auto_model,
        mock_schedule_from_config,
        mock_get_tokenizer,
        mock_real_loader_cls,
        mock_synth_gen_cls,
        mock_mix_data,
        tmp_path,
    ):
        """run() should execute all generations."""
        from src.training.lineage import LineageOrchestrator

        config = _lineage_config(tmp_path, num_generations=2)

        mock_tokenizer, mock_corpus, mock_synth_corpus, mock_mixed, mock_model = \
            self._setup_run_mocks()

        mock_schedule = MagicMock(return_value=0.0)
        mock_schedule_from_config.return_value = mock_schedule
        mock_get_tokenizer.return_value = mock_tokenizer

        mock_loader = MagicMock()
        mock_loader.get_corpus.return_value = mock_corpus
        mock_real_loader_cls.return_value = mock_loader

        mock_auto_model.from_pretrained.return_value = mock_model
        mock_synth_gen_cls.return_value.generate_corpus.return_value = mock_synth_corpus
        mock_mix_data.return_value = mock_mixed

        mock_trainer = MagicMock()
        mock_trainer.get_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        orch = LineageOrchestrator(config)
        orch.run()

        assert mock_trainer.train.call_count == 2

    @patch("src.data.mixing.mix_data")
    @patch("src.data.synthetic.SyntheticGenerator")
    @patch("src.data.real_data.RealDataLoader")
    @patch("src.data.tokenization.get_tokenizer")
    @patch("src.training.schedules.schedule_from_config")
    @patch("transformers.AutoModelForCausalLM")
    @patch("src.training.trainer.GenerationTrainer")
    def test_run_generation_single(
        self,
        mock_trainer_cls,
        mock_auto_model,
        mock_schedule_from_config,
        mock_get_tokenizer,
        mock_real_loader_cls,
        mock_synth_gen_cls,
        mock_mix_data,
        tmp_path,
    ):
        """run_generation() should run a single generation step."""
        from src.training.lineage import LineageOrchestrator

        config = _lineage_config(tmp_path, num_generations=3)

        mock_tokenizer, mock_corpus, mock_synth_corpus, mock_mixed, mock_model = \
            self._setup_run_mocks()

        mock_schedule = MagicMock(return_value=0.0)
        mock_schedule_from_config.return_value = mock_schedule
        mock_get_tokenizer.return_value = mock_tokenizer

        mock_loader = MagicMock()
        mock_loader.get_corpus.return_value = mock_corpus
        mock_real_loader_cls.return_value = mock_loader

        mock_auto_model.from_pretrained.return_value = mock_model
        mock_synth_gen_cls.return_value.generate_corpus.return_value = mock_synth_corpus
        mock_mix_data.return_value = mock_mixed

        mock_trainer = MagicMock()
        mock_trainer.get_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        orch = LineageOrchestrator(config)
        orch.run_generation(0)

        mock_trainer.train.assert_called_once()

    @patch("src.data.mixing.mix_data")
    @patch("src.data.synthetic.SyntheticGenerator")
    @patch("src.data.real_data.RealDataLoader")
    @patch("src.data.tokenization.get_tokenizer")
    @patch("src.training.schedules.schedule_from_config")
    @patch("transformers.AutoModelForCausalLM")
    @patch("src.training.trainer.GenerationTrainer")
    def test_run_generation_uses_previous_checkpoint(
        self,
        mock_trainer_cls,
        mock_auto_model,
        mock_schedule_from_config,
        mock_get_tokenizer,
        mock_real_loader_cls,
        mock_synth_gen_cls,
        mock_mix_data,
        tmp_path,
    ):
        """run_generation for gen>0 should use the previous checkpoint if it exists."""
        from src.training.lineage import LineageOrchestrator

        config = _lineage_config(tmp_path, num_generations=3)

        mock_tokenizer, mock_corpus, mock_synth_corpus, mock_mixed, mock_model = \
            self._setup_run_mocks()

        mock_schedule = MagicMock(return_value=0.0)
        mock_schedule_from_config.return_value = mock_schedule
        mock_get_tokenizer.return_value = mock_tokenizer

        mock_loader = MagicMock()
        mock_loader.get_corpus.return_value = mock_corpus
        mock_real_loader_cls.return_value = mock_loader

        mock_auto_model.from_pretrained.return_value = mock_model
        mock_synth_gen_cls.return_value.generate_corpus.return_value = mock_synth_corpus
        mock_mix_data.return_value = mock_mixed

        mock_trainer = MagicMock()
        mock_trainer.get_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        # Create a fake checkpoint for generation 0
        ckpt_dir = tmp_path / "checkpoints" / "generation_00" / "model"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "config.json").write_text('{"model_type": "gpt2"}')

        orch = LineageOrchestrator(config)
        orch.run_generation(1)

        # The model for generation should have been loaded from checkpoint
        model_path_arg = mock_auto_model.from_pretrained.call_args[0][0]
        assert "generation_00" in str(model_path_arg)

    def test_resume_all_complete(self, tmp_path):
        """resume() should do nothing when all generations are complete."""
        from src.training.lineage import LineageOrchestrator

        config = _lineage_config(tmp_path, num_generations=2)
        orch = LineageOrchestrator(config)

        for gen in range(2):
            ckpt_dir = tmp_path / "checkpoints" / f"generation_{gen:02d}" / "model"
            ckpt_dir.mkdir(parents=True)
            (ckpt_dir / "config.json").write_text('{}')

        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir(parents=True)
        records = [
            {"generation": 0, "alpha": 0.0},
            {"generation": 1, "alpha": 0.0},
        ]
        (metrics_dir / "metrics.json").write_text(json.dumps(records))

        with patch.object(orch, "run_generation") as mock_run_gen:
            orch.resume()
            mock_run_gen.assert_not_called()

    def test_resume_partial(self, tmp_path):
        """resume() should continue from the last checkpoint."""
        from src.training.lineage import LineageOrchestrator

        config = _lineage_config(tmp_path, num_generations=3)
        orch = LineageOrchestrator(config)

        ckpt_dir = tmp_path / "checkpoints" / "generation_00" / "model"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "config.json").write_text('{}')

        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir(parents=True)
        records = [{"generation": 0, "alpha": 0.0}]
        (metrics_dir / "metrics.json").write_text(json.dumps(records))

        with patch.object(orch, "run_generation") as mock_run_gen:
            orch.resume()
            assert mock_run_gen.call_count == 2
            mock_run_gen.assert_any_call(1)
            mock_run_gen.assert_any_call(2)

    def test_resume_from_scratch(self, tmp_path):
        """resume() with no checkpoints should run all generations."""
        from src.training.lineage import LineageOrchestrator

        config = _lineage_config(tmp_path, num_generations=3)
        orch = LineageOrchestrator(config)

        (tmp_path / "checkpoints").mkdir(parents=True, exist_ok=True)
        (tmp_path / "metrics").mkdir(parents=True, exist_ok=True)

        with patch.object(orch, "run_generation") as mock_run_gen:
            orch.resume()
            assert mock_run_gen.call_count == 3

    @patch("src.data.mixing.mix_data")
    @patch("src.data.synthetic.SyntheticGenerator")
    @patch("src.data.real_data.RealDataLoader")
    @patch("src.data.tokenization.get_tokenizer")
    @patch("src.training.schedules.schedule_from_config")
    @patch("transformers.AutoModelForCausalLM")
    @patch("src.training.trainer.GenerationTrainer")
    def test_run_continual_finetune(
        self,
        mock_trainer_cls,
        mock_auto_model,
        mock_schedule_from_config,
        mock_get_tokenizer,
        mock_real_loader_cls,
        mock_synth_gen_cls,
        mock_mix_data,
        tmp_path,
    ):
        """With from_pretrained_each_generation=False, gen>0 should use previous model."""
        from datasets import Dataset

        from src.training.lineage import LineageOrchestrator

        config = _lineage_config(tmp_path, num_generations=2)
        config["training"]["from_pretrained_each_generation"] = False

        mock_tokenizer = MagicMock()
        mock_corpus = Dataset.from_dict({
            "text": ["doc"] * 10, "input_ids": [[1, 2]] * 10
        })
        mock_synth_corpus = Dataset.from_dict({"text": ["synth"] * 10})
        mock_mixed = Dataset.from_dict({
            "text": ["mixed"] * 10, "input_ids": [[1, 2]] * 10
        })
        mock_model = MagicMock()

        mock_schedule = MagicMock(return_value=0.0)
        mock_schedule_from_config.return_value = mock_schedule
        mock_get_tokenizer.return_value = mock_tokenizer

        mock_loader = MagicMock()
        mock_loader.get_corpus.return_value = mock_corpus
        mock_real_loader_cls.return_value = mock_loader

        mock_auto_model.from_pretrained.return_value = mock_model
        mock_synth_gen_cls.return_value.generate_corpus.return_value = mock_synth_corpus
        mock_mix_data.return_value = mock_mixed

        mock_trainer = MagicMock()
        mock_trainer.get_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        orch = LineageOrchestrator(config)
        orch.run()

        calls = mock_trainer_cls.call_args_list
        assert len(calls) == 2
        # Gen 0: always starts from base
        assert calls[0][1]["model_name_or_path"] == "mock-model"
        # Gen 1: with continual finetune, should use checkpoint path
        assert "generation_00" in calls[1][1]["model_name_or_path"]

    def test_cleanup_gpu_no_error(self):
        """_cleanup_gpu should not raise even without CUDA."""
        from src.training.lineage import LineageOrchestrator

        LineageOrchestrator._cleanup_gpu(None)

    def test_generate_synthetic_static(self):
        """_generate_synthetic should load a model and generate a corpus."""
        import torch

        from src.data.synthetic import GenerationConfig
        from src.training.lineage import LineageOrchestrator

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        gen_config = GenerationConfig(num_samples=4, batch_size=4, max_new_tokens=8)

        with patch("transformers.AutoModelForCausalLM") as mock_auto_model:
            mock_auto_model.from_pretrained.return_value = mock_model

            with patch("src.data.synthetic.SyntheticGenerator") as mock_gen_cls:
                from datasets import Dataset

                mock_corpus = Dataset.from_dict({"text": ["generated"] * 4})
                mock_gen_cls.return_value.generate_corpus.return_value = mock_corpus

                result = LineageOrchestrator._generate_synthetic(
                    model_path="mock-path",
                    tokenizer=mock_tokenizer,
                    gen_config=gen_config,
                    seed=42,
                )

        assert len(result) == 4
        mock_auto_model.from_pretrained.assert_called_once()

    @patch("src.data.mixing.mix_data")
    @patch("src.data.synthetic.SyntheticGenerator")
    @patch("src.data.real_data.RealDataLoader")
    @patch("src.data.tokenization.get_tokenizer")
    @patch("src.training.schedules.schedule_from_config")
    @patch("transformers.AutoModelForCausalLM")
    @patch("src.training.trainer.GenerationTrainer")
    def test_run_generation_no_previous_checkpoint_falls_back(
        self,
        mock_trainer_cls,
        mock_auto_model,
        mock_schedule_from_config,
        mock_get_tokenizer,
        mock_real_loader_cls,
        mock_synth_gen_cls,
        mock_mix_data,
        tmp_path,
    ):
        """run_generation for gen>0 without checkpoint should use base model."""
        from src.training.lineage import LineageOrchestrator

        config = _lineage_config(tmp_path, num_generations=3)

        mock_tokenizer, mock_corpus, mock_synth_corpus, mock_mixed, mock_model = \
            self._setup_run_mocks()

        mock_schedule = MagicMock(return_value=0.0)
        mock_schedule_from_config.return_value = mock_schedule
        mock_get_tokenizer.return_value = mock_tokenizer

        mock_loader = MagicMock()
        mock_loader.get_corpus.return_value = mock_corpus
        mock_real_loader_cls.return_value = mock_loader

        mock_auto_model.from_pretrained.return_value = mock_model
        mock_synth_gen_cls.return_value.generate_corpus.return_value = mock_synth_corpus
        mock_mix_data.return_value = mock_mixed

        mock_trainer = MagicMock()
        mock_trainer.get_model.return_value = mock_model
        mock_trainer_cls.return_value = mock_trainer

        # No checkpoint exists for gen 0
        (tmp_path / "checkpoints").mkdir(parents=True, exist_ok=True)

        orch = LineageOrchestrator(config)
        orch.run_generation(1)

        # Should fall back to base model
        model_path_arg = mock_auto_model.from_pretrained.call_args[0][0]
        assert model_path_arg == "mock-model"


# ===================================================================
# Checkpointing tests (improving coverage from 73%)
# ===================================================================


class TestMetricsRecorder:
    """Tests for src.training.checkpointing.MetricsRecorder."""

    def test_init_creates_directory(self, tmp_path):
        """MetricsRecorder should create the output directory."""
        from src.training.checkpointing import MetricsRecorder

        metrics_dir = tmp_path / "new_metrics"
        MetricsRecorder(metrics_dir)
        assert metrics_dir.exists()

    def test_record_writes_json(self, tmp_path):
        """record() should write metrics to a JSON file."""
        from src.training.checkpointing import GenerationMetrics, MetricsRecorder

        recorder = MetricsRecorder(tmp_path / "metrics")
        metrics = GenerationMetrics(generation=0, alpha=0.5, train_loss=2.5)
        recorder.record(metrics)

        json_path = tmp_path / "metrics" / "metrics.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["generation"] == 0
        assert data[0]["alpha"] == 0.5
        assert data[0]["train_loss"] == 2.5

    def test_record_sets_timestamp(self, tmp_path):
        """record() should auto-set timestamp if not provided."""
        from src.training.checkpointing import GenerationMetrics, MetricsRecorder

        recorder = MetricsRecorder(tmp_path / "metrics")
        metrics = GenerationMetrics(generation=0, alpha=0.0)
        assert metrics.timestamp == ""
        recorder.record(metrics)
        assert metrics.timestamp != ""

    def test_record_preserves_existing_timestamp(self, tmp_path):
        """record() should not overwrite an existing timestamp."""
        from src.training.checkpointing import GenerationMetrics, MetricsRecorder

        recorder = MetricsRecorder(tmp_path / "metrics")
        metrics = GenerationMetrics(
            generation=0, alpha=0.0, timestamp="2025-01-01T00:00:00"
        )
        recorder.record(metrics)
        assert metrics.timestamp == "2025-01-01T00:00:00"

    def test_record_appends(self, tmp_path):
        """Multiple record() calls should append to the list."""
        from src.training.checkpointing import GenerationMetrics, MetricsRecorder

        recorder = MetricsRecorder(tmp_path / "metrics")
        for i in range(3):
            recorder.record(GenerationMetrics(generation=i, alpha=0.0))
        assert len(recorder.get_all()) == 3

    def test_get_all_returns_copy(self, tmp_path):
        """get_all() should return a copy of the records list."""
        from src.training.checkpointing import GenerationMetrics, MetricsRecorder

        recorder = MetricsRecorder(tmp_path / "metrics")
        recorder.record(GenerationMetrics(generation=0, alpha=0.0))

        all_records = recorder.get_all()
        assert len(all_records) == 1
        all_records.append({"generation": 99})
        assert len(recorder.get_all()) == 1

    def test_latest_generation_empty(self, tmp_path):
        """latest_generation() should return -1 when no records exist."""
        from src.training.checkpointing import MetricsRecorder

        recorder = MetricsRecorder(tmp_path / "metrics")
        assert recorder.latest_generation() == -1

    def test_latest_generation_with_records(self, tmp_path):
        """latest_generation() should return the highest generation number."""
        from src.training.checkpointing import GenerationMetrics, MetricsRecorder

        recorder = MetricsRecorder(tmp_path / "metrics")
        recorder.record(GenerationMetrics(generation=0, alpha=0.0))
        recorder.record(GenerationMetrics(generation=2, alpha=0.0))
        recorder.record(GenerationMetrics(generation=1, alpha=0.0))
        assert recorder.latest_generation() == 2

    def test_resume_from_existing_json(self, tmp_path):
        """MetricsRecorder should resume from existing metrics.json."""
        from src.training.checkpointing import GenerationMetrics, MetricsRecorder

        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()

        existing = [
            {"generation": 0, "alpha": 0.0, "timestamp": "t0"},
            {"generation": 1, "alpha": 0.0, "timestamp": "t1"},
        ]
        (metrics_dir / "metrics.json").write_text(json.dumps(existing))

        recorder = MetricsRecorder(metrics_dir)
        assert len(recorder.get_all()) == 2
        assert recorder.latest_generation() == 1

        recorder.record(GenerationMetrics(generation=2, alpha=0.0))
        assert len(recorder.get_all()) == 3

    def test_write_parquet_success(self, tmp_path):
        """_write_parquet should create a parquet file when extra has content."""
        from src.training.checkpointing import GenerationMetrics, MetricsRecorder

        recorder = MetricsRecorder(tmp_path / "metrics")
        # Use non-empty extra to avoid pyarrow empty-struct limitation
        recorder.record(GenerationMetrics(
            generation=0, alpha=0.5, train_loss=2.0,
            extra={"note": "test"},
        ))

        parquet_path = tmp_path / "metrics" / "metrics.parquet"
        assert parquet_path.exists()

        import pandas as pd

        df = pd.read_parquet(parquet_path)
        assert len(df) == 1
        assert df["generation"].iloc[0] == 0
        assert df["alpha"].iloc[0] == 0.5

    def test_write_parquet_graceful_failure_on_empty_extra(self, tmp_path):
        """_write_parquet should silently fail when pyarrow can't handle empty struct."""
        from src.training.checkpointing import GenerationMetrics, MetricsRecorder

        recorder = MetricsRecorder(tmp_path / "metrics")
        # Empty extra dict causes pyarrow to fail -- the method should handle it
        recorder.record(GenerationMetrics(generation=0, alpha=0.0))

        # The JSON file should still be written
        json_path = tmp_path / "metrics" / "metrics.json"
        assert json_path.exists()
        # Parquet might or might not exist, but no exception was raised

    def test_write_parquet_graceful_failure_on_import(self, tmp_path):
        """_write_parquet should not raise when pandas import fails."""
        from src.training.checkpointing import MetricsRecorder

        recorder = MetricsRecorder(tmp_path / "metrics")
        recorder._records = [{"generation": 0}]
        # Simulate pandas raising inside _write_parquet by making to_parquet fail
        with patch("pandas.DataFrame.to_parquet", side_effect=Exception("no parquet")):
            recorder._write_parquet()
        # Should not raise

    def test_parquet_multiple_records(self, tmp_path):
        """Parquet should contain all recorded metrics."""
        from src.training.checkpointing import GenerationMetrics, MetricsRecorder

        recorder = MetricsRecorder(tmp_path / "metrics")
        for i in range(5):
            recorder.record(
                GenerationMetrics(
                    generation=i, alpha=i * 0.2, train_loss=3.0 - i * 0.1,
                    extra={"step": i},  # non-empty extra for pyarrow compat
                )
            )

        import pandas as pd

        parquet_path = tmp_path / "metrics" / "metrics.parquet"
        df = pd.read_parquet(parquet_path)
        assert len(df) == 5
        assert list(df["generation"]) == [0, 1, 2, 3, 4]

    def test_extra_field_persisted(self, tmp_path):
        """Extra dict should be persisted in JSON."""
        from src.training.checkpointing import GenerationMetrics, MetricsRecorder

        recorder = MetricsRecorder(tmp_path / "metrics")
        metrics = GenerationMetrics(
            generation=0,
            alpha=0.0,
            extra={"custom_metric": 42.0, "notes": "test"},
        )
        recorder.record(metrics)

        with open(tmp_path / "metrics" / "metrics.json") as f:
            data = json.load(f)

        assert data[0]["extra"]["custom_metric"] == 42.0
        assert data[0]["extra"]["notes"] == "test"


class TestCheckpointManager:
    """Tests for src.training.checkpointing.CheckpointManager."""

    def test_init_creates_root_dir(self, tmp_path):
        """CheckpointManager should create the root directory."""
        from src.training.checkpointing import CheckpointManager

        root = tmp_path / "checkpoints"
        CheckpointManager(root)
        assert root.exists()

    def test_save_creates_directory_structure(self, tmp_path):
        """save() should create generation_XX/model/ and metadata.json."""
        from src.training.checkpointing import CheckpointManager

        mgr = CheckpointManager(tmp_path / "ckpt")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        gen_dir = mgr.save(
            generation=0,
            model=mock_model,
            tokenizer=mock_tokenizer,
            metadata={"alpha": 0.5},
        )

        assert gen_dir.exists()
        assert (gen_dir / "model").exists()
        assert (gen_dir / "metadata.json").exists()

        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

    def test_save_writes_metadata(self, tmp_path):
        """save() should write correct metadata to JSON."""
        from src.training.checkpointing import CheckpointManager

        mgr = CheckpointManager(tmp_path / "ckpt")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mgr.save(
            generation=3,
            model=mock_model,
            tokenizer=mock_tokenizer,
            metadata={"alpha": 0.75, "seed": 99},
        )

        meta_path = tmp_path / "ckpt" / "generation_03" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["generation"] == 3
        assert meta["alpha"] == 0.75
        assert meta["seed"] == 99
        assert "timestamp" in meta

    def test_save_no_metadata(self, tmp_path):
        """save() should work without metadata."""
        from src.training.checkpointing import CheckpointManager

        mgr = CheckpointManager(tmp_path / "ckpt")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        gen_dir = mgr.save(generation=0, model=mock_model, tokenizer=mock_tokenizer)

        with open(gen_dir / "metadata.json") as f:
            meta = json.load(f)

        assert meta["generation"] == 0
        assert "timestamp" in meta

    def test_load_standard_model(self, tmp_path):
        """load() should load a standard model when no adapter_config.json exists."""
        from src.training.checkpointing import CheckpointManager

        mgr = CheckpointManager(tmp_path / "ckpt")

        model_dir = tmp_path / "ckpt" / "generation_00" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text('{"model_type": "gpt2"}')

        mock_model = MagicMock()
        mock_tok = MagicMock()

        with patch("transformers.AutoModelForCausalLM") as mock_auto_model, \
             patch("transformers.AutoTokenizer") as mock_auto_tok:
            mock_auto_model.from_pretrained.return_value = mock_model
            mock_auto_tok.from_pretrained.return_value = mock_tok

            model, tokenizer = mgr.load(0)

        assert model is mock_model
        assert tokenizer is mock_tok
        mock_auto_model.from_pretrained.assert_called_once_with(
            str(model_dir), trust_remote_code=True
        )

    def test_load_peft_model(self, tmp_path):
        """load() should use AutoPeftModelForCausalLM when adapter_config.json exists."""
        from src.training.checkpointing import CheckpointManager

        mgr = CheckpointManager(tmp_path / "ckpt")

        model_dir = tmp_path / "ckpt" / "generation_00" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "adapter_config.json").write_text('{}')

        mock_model = MagicMock()
        mock_tok = MagicMock()

        with patch("peft.AutoPeftModelForCausalLM") as mock_auto_peft, \
             patch("transformers.AutoTokenizer") as mock_auto_tok:
            mock_auto_peft.from_pretrained.return_value = mock_model
            mock_auto_tok.from_pretrained.return_value = mock_tok

            model, tokenizer = mgr.load(0)

        assert model is mock_model
        mock_auto_peft.from_pretrained.assert_called_once()

    def test_load_nonexistent_raises(self, tmp_path):
        """load() should raise FileNotFoundError for missing generation."""
        from src.training.checkpointing import CheckpointManager

        mgr = CheckpointManager(tmp_path / "ckpt")

        with pytest.raises(FileNotFoundError, match="No checkpoint for generation 5"):
            mgr.load(5)

    def test_latest_generation_empty(self, tmp_path):
        """latest_generation() should return -1 when no checkpoints exist."""
        from src.training.checkpointing import CheckpointManager

        mgr = CheckpointManager(tmp_path / "ckpt")
        assert mgr.latest_generation() == -1

    def test_latest_generation_with_checkpoints(self, tmp_path):
        """latest_generation() should return the highest generation index."""
        from src.training.checkpointing import CheckpointManager

        root = tmp_path / "ckpt"
        mgr = CheckpointManager(root)

        for gen in [0, 2, 5]:
            (root / f"generation_{gen:02d}" / "model").mkdir(parents=True)

        assert mgr.latest_generation() == 5

    def test_latest_generation_ignores_non_generation_dirs(self, tmp_path):
        """latest_generation() should ignore directories that don't match the pattern."""
        from src.training.checkpointing import CheckpointManager

        root = tmp_path / "ckpt"
        mgr = CheckpointManager(root)

        (root / "generation_00").mkdir()
        (root / "random_dir").mkdir()
        (root / "generation_bad").mkdir()

        assert mgr.latest_generation() == 0

    def test_latest_generation_handles_malformed_names(self, tmp_path):
        """latest_generation() should skip directories with unparseable generation numbers."""
        from src.training.checkpointing import CheckpointManager

        root = tmp_path / "ckpt"
        mgr = CheckpointManager(root)

        (root / "generation_abc").mkdir()
        (root / "generation_01").mkdir()

        assert mgr.latest_generation() == 1

    def test_generation_exists(self, tmp_path):
        """generation_exists() should check for the model subdirectory."""
        from src.training.checkpointing import CheckpointManager

        root = tmp_path / "ckpt"
        mgr = CheckpointManager(root)

        assert not mgr.generation_exists(0)

        (root / "generation_00").mkdir(parents=True)
        assert not mgr.generation_exists(0)

        (root / "generation_00" / "model").mkdir()
        assert mgr.generation_exists(0)

    def test_generation_dir_formatting(self, tmp_path):
        """_generation_dir should zero-pad the generation number."""
        from src.training.checkpointing import CheckpointManager

        root = tmp_path / "ckpt"
        mgr = CheckpointManager(root)

        assert mgr._generation_dir(0).name == "generation_00"
        assert mgr._generation_dir(5).name == "generation_05"
        assert mgr._generation_dir(15).name == "generation_15"

    def test_save_returns_gen_dir_path(self, tmp_path):
        """save() should return the path to the generation directory."""
        from src.training.checkpointing import CheckpointManager

        mgr = CheckpointManager(tmp_path / "ckpt")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        result = mgr.save(generation=2, model=mock_model, tokenizer=mock_tokenizer)

        expected = tmp_path / "ckpt" / "generation_02"
        assert result == expected

    def test_multiple_saves(self, tmp_path):
        """Multiple save() calls should create separate directories."""
        from src.training.checkpointing import CheckpointManager

        mgr = CheckpointManager(tmp_path / "ckpt")

        for gen in range(3):
            mgr.save(
                generation=gen,
                model=MagicMock(),
                tokenizer=MagicMock(),
                metadata={"generation": gen},
            )

        assert mgr.latest_generation() == 2
        for gen in range(3):
            assert mgr.generation_exists(gen)


class TestGenerationMetrics:
    """Tests for src.training.checkpointing.GenerationMetrics."""

    def test_default_values(self):
        """GenerationMetrics should have sensible defaults."""
        from src.training.checkpointing import GenerationMetrics

        m = GenerationMetrics()
        assert m.generation == 0
        assert m.alpha == 0.0
        assert m.train_loss is None
        assert m.perplexity is None
        assert m.kl_divergence is None
        assert m.embedding_variance is None
        assert m.vocab_coverage is None
        assert m.timestamp == ""
        assert m.extra == {}

    def test_custom_values(self):
        """GenerationMetrics should store provided values."""
        from src.training.checkpointing import GenerationMetrics

        m = GenerationMetrics(
            generation=5,
            alpha=0.8,
            train_loss=1.5,
            perplexity=20.0,
            kl_divergence=0.3,
            extra={"custom": True},
        )
        assert m.generation == 5
        assert m.alpha == 0.8
        assert m.train_loss == 1.5
        assert m.perplexity == 20.0
        assert m.kl_divergence == 0.3
        assert m.extra == {"custom": True}

    def test_asdict(self):
        """GenerationMetrics should be convertible to a dict via asdict."""
        from dataclasses import asdict

        from src.training.checkpointing import GenerationMetrics

        m = GenerationMetrics(generation=1, alpha=0.5)
        d = asdict(m)
        assert d["generation"] == 1
        assert d["alpha"] == 0.5
        assert "extra" in d
