"""Single-generation model trainer using HuggingFace Trainer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Parameters for a single fine-tuning run."""

    epochs: int = 1
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_steps: int = -1
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    from_pretrained_each_generation: bool = True


class GenerationTrainer:
    """Fine-tune a causal language model on a (possibly mixed) dataset.

    Supports full fine-tuning and LoRA (via ``peft``) for larger models.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer,
        config: TrainingConfig,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._tokenizer = tokenizer
        self._config = config
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, dataset: "Dataset", output_dir: str | Path) -> None:
        """Fine-tune the model on *dataset* and save to *output_dir*.

        Args:
            dataset: HuggingFace Dataset with ``input_ids`` (and optionally
                ``attention_mask``) columns.  A ``labels`` column is created
                automatically for causal LM training.
            output_dir: Directory to save the trained model and tokenizer.
        """
        from transformers import (
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self._load_model()
        cfg = self._config

        # Prepare dataset: ensure input_ids exist, remove non-tensor columns.
        train_dataset = self._prepare_dataset(dataset)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            max_steps=cfg.max_steps,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            bf16=self._bf16_available(),
            dataloader_pin_memory=True,
            remove_unused_columns=True,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        logger.info("Starting training: %s", output_dir)
        trainer.train()

        # Save model + tokenizer.
        self._save_model(model, output_dir)
        self._tokenizer.save_pretrained(str(output_dir))
        logger.info("Model saved to %s", output_dir)

    def get_model(self):
        """Return the loaded (possibly LoRA-wrapped) model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the base model, optionally wrapping with LoRA."""
        from transformers import AutoModelForCausalLM

        logger.info("Loading model: %s", self._model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=True,
        )

        if self._config.use_lora:
            model = self._apply_lora(model)

        self._model = model
        return model

    def _apply_lora(self, model):
        """Wrap *model* with a LoRA adapter via ``peft``."""
        from peft import LoraConfig, TaskType, get_peft_model

        cfg = self._config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "LoRA applied: %d / %d trainable parameters (%.2f%%)",
            trainable,
            total,
            100.0 * trainable / total if total > 0 else 0,
        )
        return model

    def _save_model(self, model, output_dir: Path) -> None:
        """Save model, handling both full and LoRA models."""
        if self._config.use_lora:
            # Save only the adapter weights.
            model.save_pretrained(str(output_dir))
        else:
            model.save_pretrained(str(output_dir))

    def _prepare_dataset(self, dataset: "Dataset") -> "Dataset":
        """Ensure the dataset has the columns needed for causal LM training."""
        # If tokenized columns are missing, tokenize now.
        if "input_ids" not in dataset.column_names:
            cfg_max_len = 512  # sensible default
            dataset = dataset.map(
                lambda examples: self._tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=cfg_max_len,
                    padding=False,
                    return_attention_mask=False,
                ),
                batched=True,
                desc="Tokenizing for training",
            )

        # Drop non-tensor columns that Trainer can't handle.
        cols_to_remove = [
            c
            for c in dataset.column_names
            if c not in ("input_ids", "attention_mask", "labels")
        ]
        if cols_to_remove:
            dataset = dataset.remove_columns(cols_to_remove)

        return dataset

    @staticmethod
    def _bf16_available() -> bool:
        """Check if bf16 training is available."""
        try:
            import torch

            return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        except Exception:
            return False
