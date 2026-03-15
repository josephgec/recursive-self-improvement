"""Lineage orchestrator -- runs the recursive model-collapse loop."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LineageOrchestrator:
    """Orchestrate the multi-generation model-collapse experiment.

    Core loop for each generation *t*:

    1. Generate synthetic data from ``M_{t-1}``
    2. Mix real + synthetic at ratio ``alpha_t``
    3. Fine-tune ``M_t`` on the mixed dataset
    4. Measure quality metrics
    5. Checkpoint ``M_t``

    Supports both *fresh fine-tune* (always start from the pre-trained base)
    and *continual fine-tune* (start from ``M_{t-1}``), controlled by
    ``config["training"]["from_pretrained_each_generation"]``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._base_model_name: str = config["base_model"]
        self._num_generations: int = config["experiment"]["num_generations"]
        self._output_dir = Path(
            config["experiment"].get("output_dir", "data")
        )
        self._seed: int = config["experiment"].get("seed", 42)
        self._fresh_finetune: bool = config["training"].get(
            "from_pretrained_each_generation", True
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full lineage from generation 0 to num_generations-1."""
        from src.data.mixing import mix_data
        from src.data.real_data import CorpusConfig, RealDataLoader
        from src.data.synthetic import GenerationConfig, SyntheticGenerator
        from src.data.tokenization import get_tokenizer
        from src.training.checkpointing import (
            CheckpointManager,
            GenerationMetrics,
            MetricsRecorder,
        )
        from src.training.schedules import AlphaSchedule, schedule_from_config
        from src.training.trainer import GenerationTrainer, TrainingConfig

        # --- Setup -------------------------------------------------------
        tokenizer = get_tokenizer(self._base_model_name)

        real_cfg = CorpusConfig(**self._config.get("real_data", {}))
        real_loader = RealDataLoader(real_cfg, tokenizer)
        real_corpus = real_loader.get_corpus()

        gen_cfg = GenerationConfig(**self._config.get("synthetic_generation", {}))
        train_cfg = TrainingConfig(**self._config.get("training", {}))

        schedule_dict = self._config.get("schedule", {"type": "zero"})
        schedule: AlphaSchedule = schedule_from_config(schedule_dict)

        checkpoint_mgr = CheckpointManager(self._output_dir / "checkpoints")
        metrics_recorder = MetricsRecorder(self._output_dir / "metrics")

        # --- Generation 0: baseline model --------------------------------
        current_model_path = self._base_model_name

        for gen in range(self._num_generations):
            logger.info(
                "===== Generation %d / %d =====", gen, self._num_generations
            )

            alpha_t = schedule(gen, self._num_generations)
            logger.info("Alpha schedule: alpha_%d = %.4f", gen, alpha_t)

            # 1. Generate synthetic data from current model.
            synthetic_corpus = self._generate_synthetic(
                model_path=current_model_path,
                tokenizer=tokenizer,
                gen_config=gen_cfg,
                seed=self._seed + gen,
            )

            # 2. Mix real + synthetic.
            total_size = gen_cfg.num_samples
            mixed_dataset = mix_data(
                real_data=real_corpus,
                synthetic_data=synthetic_corpus,
                alpha=alpha_t,
                total_size=total_size,
                seed=self._seed + gen,
            )
            logger.info(
                "Mixed dataset: %d rows (alpha=%.2f)", len(mixed_dataset), alpha_t
            )

            # 3. Determine starting point for fine-tuning.
            if self._fresh_finetune or gen == 0:
                finetune_base = self._base_model_name
            else:
                finetune_base = str(
                    checkpoint_mgr._generation_dir(gen - 1) / "model"
                )

            # 4. Fine-tune.
            trainer = GenerationTrainer(
                model_name_or_path=finetune_base,
                tokenizer=tokenizer,
                config=train_cfg,
            )
            gen_output_dir = self._output_dir / "training" / f"generation_{gen:02d}"
            trainer.train(mixed_dataset, gen_output_dir)
            model = trainer.get_model()

            # 5. Checkpoint.
            checkpoint_mgr.save(
                generation=gen,
                model=model,
                tokenizer=tokenizer,
                metadata={"alpha": alpha_t, "seed": self._seed + gen},
            )

            # 6. Record metrics.
            metrics = GenerationMetrics(
                generation=gen,
                alpha=alpha_t,
            )
            metrics_recorder.record(metrics)

            # Update path for next generation.
            current_model_path = str(
                checkpoint_mgr._generation_dir(gen) / "model"
            )

            # Clean up GPU memory.
            self._cleanup_gpu(model)

        logger.info("Lineage complete: %d generations", self._num_generations)

    def run_generation(self, generation: int) -> None:
        """Run a single generation step (for testing or manual control).

        Args:
            generation: The generation index to run.
        """
        from src.data.mixing import mix_data
        from src.data.real_data import CorpusConfig, RealDataLoader
        from src.data.synthetic import GenerationConfig, SyntheticGenerator
        from src.data.tokenization import get_tokenizer
        from src.training.checkpointing import (
            CheckpointManager,
            GenerationMetrics,
            MetricsRecorder,
        )
        from src.training.schedules import schedule_from_config
        from src.training.trainer import GenerationTrainer, TrainingConfig

        tokenizer = get_tokenizer(self._base_model_name)

        real_cfg = CorpusConfig(**self._config.get("real_data", {}))
        real_loader = RealDataLoader(real_cfg, tokenizer)
        real_corpus = real_loader.get_corpus()

        gen_cfg = GenerationConfig(**self._config.get("synthetic_generation", {}))
        train_cfg = TrainingConfig(**self._config.get("training", {}))

        schedule_dict = self._config.get("schedule", {"type": "zero"})
        schedule = schedule_from_config(schedule_dict)
        alpha_t = schedule(generation, self._num_generations)

        checkpoint_mgr = CheckpointManager(self._output_dir / "checkpoints")
        metrics_recorder = MetricsRecorder(self._output_dir / "metrics")

        # Determine model to generate from.
        if generation == 0:
            current_model_path = self._base_model_name
        elif checkpoint_mgr.generation_exists(generation - 1):
            current_model_path = str(
                checkpoint_mgr._generation_dir(generation - 1) / "model"
            )
        else:
            current_model_path = self._base_model_name

        # Generate synthetic.
        synthetic_corpus = self._generate_synthetic(
            model_path=current_model_path,
            tokenizer=tokenizer,
            gen_config=gen_cfg,
            seed=self._seed + generation,
        )

        # Mix.
        mixed_dataset = mix_data(
            real_data=real_corpus,
            synthetic_data=synthetic_corpus,
            alpha=alpha_t,
            total_size=gen_cfg.num_samples,
            seed=self._seed + generation,
        )

        # Fine-tune.
        if self._fresh_finetune or generation == 0:
            finetune_base = self._base_model_name
        else:
            finetune_base = current_model_path

        trainer = GenerationTrainer(
            model_name_or_path=finetune_base,
            tokenizer=tokenizer,
            config=train_cfg,
        )
        gen_output_dir = (
            self._output_dir / "training" / f"generation_{generation:02d}"
        )
        trainer.train(mixed_dataset, gen_output_dir)
        model = trainer.get_model()

        # Checkpoint & metrics.
        checkpoint_mgr.save(
            generation=generation,
            model=model,
            tokenizer=tokenizer,
            metadata={"alpha": alpha_t},
        )
        metrics_recorder.record(
            GenerationMetrics(generation=generation, alpha=alpha_t)
        )
        self._cleanup_gpu(model)

    def resume(self) -> None:
        """Resume a previously interrupted lineage from the last checkpoint.

        Finds the latest completed generation and continues from there.
        """
        from src.training.checkpointing import CheckpointManager, MetricsRecorder

        checkpoint_mgr = CheckpointManager(self._output_dir / "checkpoints")
        metrics_recorder = MetricsRecorder(self._output_dir / "metrics")

        latest = max(
            checkpoint_mgr.latest_generation(),
            metrics_recorder.latest_generation(),
        )
        start_gen = latest + 1

        if start_gen >= self._num_generations:
            logger.info(
                "All %d generations already complete; nothing to resume.",
                self._num_generations,
            )
            return

        logger.info(
            "Resuming from generation %d (of %d)",
            start_gen,
            self._num_generations,
        )

        # Re-run the full loop from start_gen onward.
        # Temporarily adjust config and delegate to run().
        original = self._num_generations
        saved_config = dict(self._config)

        for gen in range(start_gen, original):
            self.run_generation(gen)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_synthetic(
        model_path: str,
        tokenizer,
        gen_config,
        seed: int,
    ):
        """Load a model and generate a synthetic corpus."""
        from transformers import AutoModelForCausalLM

        from src.data.synthetic import SyntheticGenerator

        logger.info("Loading model for generation: %s", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Move to GPU if available.
        try:
            import torch

            if torch.cuda.is_available():
                model = model.cuda()
        except ImportError:
            pass

        model.eval()
        generator = SyntheticGenerator(model, tokenizer, gen_config)
        corpus = generator.generate_corpus(n=gen_config.num_samples, seed=seed)

        # Free the generation model.
        del model
        LineageOrchestrator._cleanup_gpu(None)

        return corpus

    @staticmethod
    def _cleanup_gpu(model) -> None:
        """Free GPU memory."""
        try:
            import torch

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
