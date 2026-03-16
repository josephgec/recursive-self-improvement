"""Mock OpenAI fine-tuning API client."""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional

from src.synthesis.synthesizer import TrainingPair


class OpenAIFineTuner:
    """Mock OpenAI fine-tuning API for offline testing.

    Simulates the OpenAI fine-tuning workflow:
    1. Upload training file
    2. Create fine-tuning job
    3. Monitor job status
    4. Retrieve fine-tuned model
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini-2024-07-18",
        n_epochs: int = 3,
        batch_size: int = 4,
        learning_rate_multiplier: float = 1.8,
    ):
        self.model = model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate_multiplier = learning_rate_multiplier
        self._job_id: Optional[str] = None
        self._file_id: Optional[str] = None
        self._fine_tuned_model: Optional[str] = None

    def upload_file(self, pairs: List[TrainingPair]) -> str:
        """Mock file upload. Returns a fake file ID."""
        self._file_id = f"file-{uuid.uuid4().hex[:12]}"
        return self._file_id

    def create_job(self, file_id: Optional[str] = None) -> str:
        """Mock job creation. Returns a fake job ID."""
        self._job_id = f"ftjob-{uuid.uuid4().hex[:12]}"
        return self._job_id

    def get_job_status(self) -> Dict[str, Any]:
        """Mock job status check. Always returns 'succeeded'."""
        return {
            "id": self._job_id,
            "status": "succeeded",
            "model": self.model,
            "fine_tuned_model": self._get_ft_model_name(),
            "training_file": self._file_id,
            "hyperparameters": {
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "learning_rate_multiplier": self.learning_rate_multiplier,
            },
        }

    def fine_tune(
        self,
        train_pairs: List[TrainingPair],
        val_pairs: Optional[List[TrainingPair]] = None,
    ) -> Dict[str, Any]:
        """Run the full mock fine-tuning workflow."""
        file_id = self.upload_file(train_pairs)
        job_id = self.create_job(file_id)
        status = self.get_job_status()

        # Simulate training metrics
        rng = random.Random(42)
        n_steps = max(1, len(train_pairs) // self.batch_size) * self.n_epochs
        train_loss = [2.0 * (0.85 ** i) + rng.gauss(0, 0.05) for i in range(n_steps)]
        val_loss = [2.2 * (0.87 ** i) + rng.gauss(0, 0.08) for i in range(n_steps)]

        metrics = {
            "train_loss_final": max(0.01, train_loss[-1]) if train_loss else 0.5,
            "val_loss_final": max(0.01, val_loss[-1]) if val_loss else 0.6,
            "train_loss_history": [round(x, 4) for x in train_loss],
            "val_loss_history": [round(x, 4) for x in val_loss],
            "n_train": len(train_pairs),
            "n_val": len(val_pairs) if val_pairs else 0,
            "n_steps": n_steps,
            "n_epochs": self.n_epochs,
        }

        self._fine_tuned_model = self._get_ft_model_name()

        return {
            "job_id": job_id,
            "file_id": file_id,
            "status": "succeeded",
            "fine_tuned_model": self._fine_tuned_model,
            "metrics": metrics,
            "backend": "openai",
        }

    def _get_ft_model_name(self) -> str:
        """Generate a fine-tuned model name."""
        if self._fine_tuned_model:
            return self._fine_tuned_model
        return f"ft:{self.model}:soar:{uuid.uuid4().hex[:6]}"
