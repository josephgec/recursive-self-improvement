"""Mock local fine-tuning backend."""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional

from src.synthesis.synthesizer import TrainingPair


class LocalFineTuner:
    """Mock local fine-tuning for offline testing.

    Simulates LoRA/QLoRA fine-tuning of a local model.
    """

    def __init__(
        self,
        base_model: str = "codellama-7b",
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
    ):
        self.base_model = base_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._model_path: Optional[str] = None

    def fine_tune(
        self,
        train_pairs: List[TrainingPair],
        val_pairs: Optional[List[TrainingPair]] = None,
    ) -> Dict[str, Any]:
        """Run mock local fine-tuning."""
        rng = random.Random(42)
        n_steps = max(1, len(train_pairs) // self.batch_size) * self.epochs

        # Simulate training curves
        train_loss = [3.0 * (0.9 ** i) + rng.gauss(0, 0.03) for i in range(n_steps)]
        val_loss = [3.2 * (0.92 ** i) + rng.gauss(0, 0.05) for i in range(n_steps)]

        model_id = uuid.uuid4().hex[:8]
        self._model_path = f"data/models/soar-local-{model_id}"

        metrics = {
            "train_loss_final": max(0.01, train_loss[-1]) if train_loss else 0.5,
            "val_loss_final": max(0.01, val_loss[-1]) if val_loss else 0.6,
            "train_loss_history": [round(x, 4) for x in train_loss],
            "val_loss_history": [round(x, 4) for x in val_loss],
            "n_train": len(train_pairs),
            "n_val": len(val_pairs) if val_pairs else 0,
            "n_steps": n_steps,
            "n_epochs": self.epochs,
        }

        return {
            "model_path": self._model_path,
            "base_model": self.base_model,
            "status": "completed",
            "metrics": metrics,
            "backend": "local",
            "config": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
            },
        }

    @property
    def model_path(self) -> Optional[str]:
        return self._model_path
