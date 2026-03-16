"""Context manager: manages context for RLM sessions."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.pipeline.state import PipelineState


class ContextManager:
    """Manages context loading for RLM sessions."""

    def __init__(self, max_tokens: int = 100000):
        self._max_tokens = max_tokens
        self._codebase: str = ""
        self._dataset: str = ""
        self._history: List[Dict[str, Any]] = []

    def load_codebase(self, code: str) -> int:
        """Load codebase into context. Returns token count estimate."""
        self._codebase = code
        return self._estimate_tokens(code)

    def load_dataset(self, data: str) -> int:
        """Load dataset into context. Returns token count estimate."""
        self._dataset = data
        return self._estimate_tokens(data)

    def load_history(self, history: List[Dict[str, Any]]) -> int:
        """Load iteration history into context. Returns token count estimate."""
        self._history = list(history)
        import json
        text = json.dumps(history)
        return self._estimate_tokens(text)

    def get_context_size(self) -> int:
        """Get total context size in estimated tokens."""
        import json
        total = self._estimate_tokens(self._codebase)
        total += self._estimate_tokens(self._dataset)
        total += self._estimate_tokens(json.dumps(self._history))
        return total

    def get_context(self) -> Dict[str, Any]:
        """Get the full context for an RLM session."""
        return {
            "codebase": self._codebase,
            "dataset": self._dataset,
            "history": self._history,
            "token_count": self.get_context_size(),
        }

    def fits_in_context(self) -> bool:
        """Check if current context fits within max token limit."""
        return self.get_context_size() <= self._max_tokens

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count (roughly 4 chars per token)."""
        return len(text) // 4
