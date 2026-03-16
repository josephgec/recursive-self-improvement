"""Track model versions and their performance."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional


class ModelRegistry:
    """Registry for tracking fine-tuned model versions."""

    def __init__(self) -> None:
        self._models: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        backend: str,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        training_pairs_count: int = 0,
    ) -> str:
        """Register a new model version."""
        version_id = str(uuid.uuid4())[:8]
        self._models[name] = {
            "name": name,
            "version_id": version_id,
            "backend": backend,
            "metrics": metrics or {},
            "config": config or {},
            "training_pairs_count": training_pairs_count,
        }
        return version_id

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model info by name."""
        return self._models.get(name)

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def get_best(self, metric: str = "val_loss_final") -> Optional[Dict[str, Any]]:
        """Get the model with the best (lowest) metric value."""
        if not self._models:
            return None

        best_name = None
        best_val = float("inf")

        for name, info in self._models.items():
            val = info.get("metrics", {}).get(metric, float("inf"))
            if val < best_val:
                best_val = val
                best_name = name

        return self._models.get(best_name) if best_name else None

    def delete(self, name: str) -> bool:
        """Delete a model from the registry."""
        if name in self._models:
            del self._models[name]
            return True
        return False

    def count(self) -> int:
        return len(self._models)

    def summary(self) -> Dict[str, Any]:
        """Return summary of registered models."""
        return {
            "total_models": len(self._models),
            "models": [
                {
                    "name": name,
                    "backend": info["backend"],
                    "training_pairs": info["training_pairs_count"],
                }
                for name, info in self._models.items()
            ],
        }
