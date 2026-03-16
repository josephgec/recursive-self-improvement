"""Tests for fine-tuning trainer, backends, evaluation, and registry."""

import pytest

from src.finetuning.data_loader import DataLoader
from src.finetuning.trainer import Trainer
from src.finetuning.openai_ft import OpenAIFineTuner
from src.finetuning.local_ft import LocalFineTuner
from src.finetuning.evaluation import Evaluator
from src.finetuning.model_registry import ModelRegistry
from src.synthesis.synthesizer import TrainingPair


class TestDataLoader:
    def test_load_pairs(self, sample_training_pairs):
        loader = DataLoader()
        loader.load_pairs(sample_training_pairs)
        assert loader.count == len(sample_training_pairs)

    def test_split(self, sample_training_pairs):
        loader = DataLoader()
        loader.load_pairs(sample_training_pairs)
        train, val, test = loader.split(0.5, 0.25, 0.25)
        assert len(train) + len(val) + len(test) == len(sample_training_pairs)

    def test_get_batches(self, sample_training_pairs):
        loader = DataLoader()
        loader.load_pairs(sample_training_pairs)
        batches = loader.get_batches(batch_size=2)
        assert len(batches) == 2
        assert len(batches[0]) == 2

    def test_filter_by_strategy(self, sample_training_pairs):
        loader = DataLoader()
        loader.load_pairs(sample_training_pairs)
        direct = loader.filter_by_strategy("direct_solution")
        assert len(direct) == 2

    def test_filter_by_quality(self, sample_training_pairs):
        loader = DataLoader()
        loader.load_pairs(sample_training_pairs)
        high = loader.filter_by_quality(min_score=0.85)
        assert all(p.quality_score >= 0.85 for p in high)

    def test_stats(self, sample_training_pairs):
        loader = DataLoader()
        loader.load_pairs(sample_training_pairs)
        s = loader.stats()
        assert s["total"] == 4
        assert "by_strategy" in s
        assert "avg_quality" in s

    def test_stats_empty(self):
        loader = DataLoader()
        s = loader.stats()
        assert s["total"] == 0

    def test_load_from_dicts(self):
        loader = DataLoader()
        data = [
            {"prompt": "test", "completion": "code", "quality_score": 0.5},
            {"prompt": "test2", "completion": "code2", "quality_score": 0.8},
        ]
        pairs = loader.load_from_dicts(data)
        assert len(pairs) == 2
        assert loader.count == 2

    def test_pairs_property(self, sample_training_pairs):
        loader = DataLoader()
        loader.load_pairs(sample_training_pairs)
        pairs = loader.pairs
        assert len(pairs) == len(sample_training_pairs)
        # Should be a copy
        pairs.clear()
        assert loader.count == len(sample_training_pairs)


class TestOpenAIFineTuner:
    def test_fine_tune(self, sample_training_pairs):
        tuner = OpenAIFineTuner(n_epochs=2)
        result = tuner.fine_tune(sample_training_pairs)
        assert result["status"] == "succeeded"
        assert result["backend"] == "openai"
        assert "fine_tuned_model" in result
        assert "metrics" in result

    def test_metrics(self, sample_training_pairs):
        tuner = OpenAIFineTuner(n_epochs=2)
        result = tuner.fine_tune(sample_training_pairs)
        metrics = result["metrics"]
        assert "train_loss_final" in metrics
        assert "val_loss_final" in metrics
        assert "train_loss_history" in metrics
        assert metrics["n_train"] == len(sample_training_pairs)

    def test_upload_file(self, sample_training_pairs):
        tuner = OpenAIFineTuner()
        file_id = tuner.upload_file(sample_training_pairs)
        assert file_id.startswith("file-")

    def test_create_job(self):
        tuner = OpenAIFineTuner()
        job_id = tuner.create_job()
        assert job_id.startswith("ftjob-")

    def test_job_status(self, sample_training_pairs):
        tuner = OpenAIFineTuner()
        tuner.upload_file(sample_training_pairs)
        tuner.create_job()
        status = tuner.get_job_status()
        assert status["status"] == "succeeded"
        assert "fine_tuned_model" in status

    def test_with_validation(self, sample_training_pairs):
        tuner = OpenAIFineTuner()
        train = sample_training_pairs[:3]
        val = sample_training_pairs[3:]
        result = tuner.fine_tune(train, val)
        assert result["metrics"]["n_val"] == len(val)


class TestLocalFineTuner:
    def test_fine_tune(self, sample_training_pairs):
        tuner = LocalFineTuner(epochs=2)
        result = tuner.fine_tune(sample_training_pairs)
        assert result["status"] == "completed"
        assert result["backend"] == "local"
        assert "model_path" in result
        assert "metrics" in result

    def test_model_path(self, sample_training_pairs):
        tuner = LocalFineTuner()
        result = tuner.fine_tune(sample_training_pairs)
        assert tuner.model_path is not None
        assert tuner.model_path.startswith("data/models/")

    def test_config(self, sample_training_pairs):
        tuner = LocalFineTuner(epochs=3, batch_size=4, learning_rate=1e-4)
        result = tuner.fine_tune(sample_training_pairs)
        assert result["config"]["epochs"] == 3
        assert result["config"]["batch_size"] == 4

    def test_with_validation(self, sample_training_pairs):
        tuner = LocalFineTuner()
        result = tuner.fine_tune(sample_training_pairs[:3], sample_training_pairs[3:])
        assert result["metrics"]["n_val"] == 1


class TestTrainer:
    def test_train_openai(self, sample_training_pairs):
        trainer = Trainer(backend="openai")
        result = trainer.train(sample_training_pairs, model_name="test-model")
        assert result["model_name"] == "test-model"
        assert "metrics" in result

    def test_train_local(self, sample_training_pairs):
        trainer = Trainer(backend="local")
        result = trainer.train(sample_training_pairs, model_name="test-local")
        assert result["model_name"] == "test-local"

    def test_unknown_backend(self, sample_training_pairs):
        trainer = Trainer(backend="unknown")
        with pytest.raises(ValueError, match="Unknown backend"):
            trainer.train(sample_training_pairs)

    def test_model_registered(self, sample_training_pairs):
        registry = ModelRegistry()
        trainer = Trainer(backend="openai", registry=registry)
        trainer.train(sample_training_pairs, model_name="registered-model")
        assert "registered-model" in registry.list_models()

    def test_result_property(self, sample_training_pairs):
        trainer = Trainer(backend="openai")
        assert trainer.result is None
        trainer.train(sample_training_pairs)
        assert trainer.result is not None

    def test_get_best_model(self, sample_training_pairs):
        trainer = Trainer(backend="openai")
        trainer.train(sample_training_pairs, model_name="m1")
        best = trainer.get_best_model()
        assert best is not None

    def test_auto_model_name(self, sample_training_pairs):
        trainer = Trainer(backend="openai")
        result = trainer.train(sample_training_pairs)
        assert "model_name" in result
        assert result["model_name"].startswith("soar-ft-")


class TestEvaluator:
    def test_evaluate_base(self):
        evaluator = Evaluator()
        metrics = evaluator.evaluate("base", is_base=True)
        assert "zero_shot_solve_rate" in metrics
        assert "initial_quality" in metrics
        assert "mutation_quality" in metrics

    def test_evaluate_finetuned(self):
        evaluator = Evaluator()
        metrics = evaluator.evaluate("ft-model", is_base=False)
        assert metrics["zero_shot_solve_rate"] > 0

    def test_compare(self):
        evaluator = Evaluator()
        evaluator.evaluate("base", is_base=True)
        evaluator.evaluate("ft-model", is_base=False)
        comparison = evaluator.compare("base", "ft-model")
        assert "zero_shot_solve_rate" in comparison
        assert "overall_improved" in comparison
        for metric in ["zero_shot_solve_rate", "initial_quality", "mutation_quality"]:
            assert "base" in comparison[metric]
            assert "finetuned" in comparison[metric]
            assert "improvement" in comparison[metric]

    def test_compare_missing(self):
        evaluator = Evaluator()
        result = evaluator.compare("missing1", "missing2")
        assert "error" in result

    def test_results_property(self):
        evaluator = Evaluator()
        evaluator.evaluate("model-a", is_base=True)
        assert "model-a" in evaluator.results

    def test_clear(self):
        evaluator = Evaluator()
        evaluator.evaluate("model-a", is_base=True)
        evaluator.clear()
        assert len(evaluator.results) == 0

    def test_n_tasks(self):
        evaluator = Evaluator()
        metrics = evaluator.evaluate("model", n_tasks=100)
        assert metrics["n_tasks"] == 100


class TestModelRegistry:
    def test_register_and_get(self):
        registry = ModelRegistry()
        vid = registry.register("model-1", "openai", {"loss": 0.5})
        assert len(vid) > 0
        model = registry.get("model-1")
        assert model is not None
        assert model["name"] == "model-1"
        assert model["metrics"]["loss"] == 0.5

    def test_list_models(self):
        registry = ModelRegistry()
        registry.register("m1", "openai")
        registry.register("m2", "local")
        models = registry.list_models()
        assert "m1" in models
        assert "m2" in models

    def test_get_best(self):
        registry = ModelRegistry()
        registry.register("m1", "openai", {"val_loss_final": 0.5})
        registry.register("m2", "openai", {"val_loss_final": 0.3})
        best = registry.get_best()
        assert best["name"] == "m2"

    def test_get_best_empty(self):
        registry = ModelRegistry()
        assert registry.get_best() is None

    def test_delete(self):
        registry = ModelRegistry()
        registry.register("m1", "openai")
        assert registry.delete("m1") is True
        assert registry.get("m1") is None
        assert registry.delete("nonexistent") is False

    def test_count(self):
        registry = ModelRegistry()
        assert registry.count() == 0
        registry.register("m1", "openai")
        assert registry.count() == 1

    def test_summary(self):
        registry = ModelRegistry()
        registry.register("m1", "openai", training_pairs_count=100)
        s = registry.summary()
        assert s["total_models"] == 1
        assert s["models"][0]["training_pairs"] == 100

    def test_get_nonexistent(self):
        registry = ModelRegistry()
        assert registry.get("nonexistent") is None
