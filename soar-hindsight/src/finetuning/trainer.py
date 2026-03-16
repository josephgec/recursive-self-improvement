"""Fine-tuning orchestrator that delegates to backend-specific trainers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.finetuning.data_loader import DataLoader
from src.finetuning.openai_ft import OpenAIFineTuner
from src.finetuning.local_ft import LocalFineTuner
from src.finetuning.model_registry import ModelRegistry
from src.synthesis.synthesizer import TrainingPair


class Trainer:
    """Orchestrates the fine-tuning process.

    Delegates to OpenAI or local fine-tuning backends based on configuration.
    """

    def __init__(
        self,
        backend: str = "openai",
        config: Optional[Dict[str, Any]] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        self.backend = backend
        self.config = config or {}
        self.registry = registry or ModelRegistry()
        self._result: Optional[Dict[str, Any]] = None

    def train(
        self,
        pairs: List[TrainingPair],
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run fine-tuning with the configured backend."""
        loader = DataLoader()
        loader.load_pairs(pairs)
        train_data, val_data, _ = loader.split()

        if self.backend == "openai":
            tuner = OpenAIFineTuner(**self.config.get("openai", {}))
            result = tuner.fine_tune(train_data, val_data)
        elif self.backend == "local":
            tuner = LocalFineTuner(**self.config.get("local", {}))
            result = tuner.fine_tune(train_data, val_data)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # Register the model
        name = model_name or f"soar-ft-{self.backend}-{len(self.registry.list_models())}"
        self.registry.register(
            name=name,
            backend=self.backend,
            metrics=result.get("metrics", {}),
            config=self.config,
            training_pairs_count=len(pairs),
        )

        self._result = result
        self._result["model_name"] = name
        return self._result

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        return self._result

    def get_best_model(self) -> Optional[Dict[str, Any]]:
        """Return the best model from the registry."""
        return self.registry.get_best()
