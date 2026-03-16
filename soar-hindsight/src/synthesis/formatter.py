"""Format training pairs for fine-tuning APIs."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from src.synthesis.synthesizer import TrainingPair


class Formatter:
    """Format training pairs as OpenAI JSONL or HuggingFace datasets format."""

    def __init__(self, format_type: str = "openai_jsonl"):
        if format_type not in ("openai_jsonl", "huggingface"):
            raise ValueError(f"Unsupported format: {format_type}")
        self.format_type = format_type

    def format_pair(self, pair: TrainingPair) -> Dict[str, Any]:
        """Format a single training pair."""
        if self.format_type == "openai_jsonl":
            return self._format_openai(pair)
        else:
            return self._format_huggingface(pair)

    def format_all(self, pairs: List[TrainingPair]) -> List[Dict[str, Any]]:
        """Format all training pairs."""
        return [self.format_pair(p) for p in pairs]

    def write_jsonl(self, pairs: List[TrainingPair], filepath: str) -> int:
        """Write training pairs to a JSONL file, return count written."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        formatted = self.format_all(pairs)
        with open(filepath, "w") as f:
            for item in formatted:
                f.write(json.dumps(item) + "\n")
        return len(formatted)

    def read_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """Read training data from a JSONL file."""
        items = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def _format_openai(self, pair: TrainingPair) -> Dict[str, Any]:
        """Format as OpenAI fine-tuning JSONL format."""
        return {
            "messages": [
                {"role": "system", "content": "You are a helpful programming assistant."},
                {"role": "user", "content": pair.prompt},
                {"role": "assistant", "content": pair.completion},
            ]
        }

    def _format_huggingface(self, pair: TrainingPair) -> Dict[str, Any]:
        """Format as HuggingFace dataset format."""
        return {
            "instruction": pair.prompt,
            "output": pair.completion,
            "metadata": {
                "pair_id": pair.pair_id,
                "strategy": pair.strategy,
                "quality_score": pair.quality_score,
            },
        }
