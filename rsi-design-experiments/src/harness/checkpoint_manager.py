"""Checkpoint manager: save and resume experiment state."""

import json
import os
from typing import Any, Dict, List, Optional

from src.experiments.base import ExperimentResult, ConditionResult


class CheckpointManager:
    """Manages checkpoints for experiment state persistence."""

    def __init__(self, checkpoint_dir: str = "data/experiment_results"):
        self._checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, result: ExperimentResult, filename: Optional[str] = None) -> str:
        """Save an experiment result to a checkpoint file.

        Returns the path to the saved file.
        """
        if filename is None:
            filename = f"{result.experiment_name}_checkpoint.json"

        filepath = os.path.join(self._checkpoint_dir, filename)

        data = self._serialize_result(result)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    def load(self, filename: str) -> ExperimentResult:
        """Load an experiment result from a checkpoint file."""
        filepath = os.path.join(self._checkpoint_dir, filename)
        with open(filepath, "r") as f:
            data = json.load(f)

        return self._deserialize_result(data)

    def resume(self, experiment_name: str) -> Optional[ExperimentResult]:
        """Try to resume from a previous checkpoint.

        Returns None if no checkpoint exists.
        """
        filename = f"{experiment_name}_checkpoint.json"
        filepath = os.path.join(self._checkpoint_dir, filename)

        if not os.path.exists(filepath):
            return None

        return self.load(filename)

    def list(self) -> List[str]:
        """List all checkpoint files."""
        if not os.path.exists(self._checkpoint_dir):
            return []
        return [
            f
            for f in os.listdir(self._checkpoint_dir)
            if f.endswith("_checkpoint.json")
        ]

    def _serialize_result(self, result: ExperimentResult) -> Dict[str, Any]:
        """Serialize ExperimentResult to a JSON-compatible dict."""
        per_condition = {}
        for cond_name, cond_results in result.per_condition_results.items():
            per_condition[cond_name] = [
                {
                    "condition_name": cr.condition_name,
                    "accuracy_trajectory": cr.accuracy_trajectory,
                    "final_accuracy": cr.final_accuracy,
                    "stability_score": cr.stability_score,
                    "rollback_count": cr.rollback_count,
                    "total_cost": cr.total_cost,
                    "improvement_rate": cr.improvement_rate,
                    "generalization_score": cr.generalization_score,
                    "composite_score": cr.composite_score,
                    "metadata": cr.metadata,
                }
                for cr in cond_results
            ]

        return {
            "experiment_name": result.experiment_name,
            "conditions": result.conditions,
            "per_condition_results": per_condition,
            "repetitions": result.repetitions,
            "metadata": result.metadata,
        }

    def _deserialize_result(self, data: Dict[str, Any]) -> ExperimentResult:
        """Deserialize a dict back to ExperimentResult."""
        per_condition = {}
        for cond_name, cond_list in data.get("per_condition_results", {}).items():
            per_condition[cond_name] = [
                ConditionResult(
                    condition_name=cr["condition_name"],
                    accuracy_trajectory=cr.get("accuracy_trajectory", []),
                    final_accuracy=cr.get("final_accuracy", 0.0),
                    stability_score=cr.get("stability_score", 0.0),
                    rollback_count=cr.get("rollback_count", 0),
                    total_cost=cr.get("total_cost", 0.0),
                    improvement_rate=cr.get("improvement_rate", 0.0),
                    generalization_score=cr.get("generalization_score", 0.0),
                    composite_score=cr.get("composite_score", 0.0),
                    metadata=cr.get("metadata", {}),
                )
                for cr in cond_list
            ]

        return ExperimentResult(
            experiment_name=data["experiment_name"],
            conditions=data.get("conditions", []),
            per_condition_results=per_condition,
            repetitions=data.get("repetitions", 1),
            metadata=data.get("metadata", {}),
        )
