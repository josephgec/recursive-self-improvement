"""ContextLoader: load context into a REPL namespace."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.token_counter import estimate_tokens


@dataclass
class ContextMeta:
    """Metadata about the loaded context."""
    context_type: str  # "str", "list", "dict", "file"
    size_chars: int
    size_tokens: int
    num_lines: int
    num_chunks: int = 1
    source: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class ContextLoader:
    """Loads context into a REPL namespace so LLM-generated code can access it."""

    def __init__(self, max_chunk_size: int = 4000, overlap: int = 200) -> None:
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_into_repl(
        self,
        context: Union[str, list, dict, Path],
        repl: Dict[str, Any],
    ) -> ContextMeta:
        """Load *context* into *repl* namespace and return metadata."""
        text, ctx_type, source = self._normalize(context)
        meta = self._build_meta(text, ctx_type, source)

        repl["CONTEXT"] = text
        repl["CONTEXT_LENGTH"] = meta.size_chars
        repl["CONTEXT_TYPE"] = meta.context_type
        repl["CONTEXT_SIZE_TOKENS"] = meta.size_tokens
        repl["CONTEXT_META"] = meta

        return meta

    def load_from_file(self, path: Union[str, Path], repl: Dict[str, Any]) -> ContextMeta:
        """Load context from a file path."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        return self.load_into_repl(text, repl)

    def load_chunked(
        self,
        context: Union[str, list, dict, Path],
        repl: Dict[str, Any],
    ) -> ContextMeta:
        """Load context and also set CONTEXT_CHUNKS in the REPL."""
        meta = self.load_into_repl(context, repl)
        text = repl["CONTEXT"]
        chunks = self._chunk_text(text, self.max_chunk_size, self.overlap)
        repl["CONTEXT_CHUNKS"] = chunks
        meta.num_chunks = len(chunks)
        return meta

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Convenience wrapper around the token estimator."""
        return estimate_tokens(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, context: Union[str, list, dict, Path]) -> tuple[str, str, Optional[str]]:
        """Convert the context to a string and determine its type."""
        if isinstance(context, Path):
            text = context.read_text(encoding="utf-8")
            return text, "file", str(context)
        if isinstance(context, dict):
            return json.dumps(context, indent=2), "dict", None
        if isinstance(context, list):
            return "\n".join(str(item) for item in context), "list", None
        return str(context), "str", None

    @staticmethod
    def _build_meta(text: str, ctx_type: str, source: Optional[str]) -> ContextMeta:
        lines = text.split("\n")
        return ContextMeta(
            context_type=ctx_type,
            size_chars=len(text),
            size_tokens=estimate_tokens(text),
            num_lines=len(lines),
            source=source,
        )

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        effective_overlap = min(overlap, chunk_size - 1) if chunk_size > 1 else 0
        effective_overlap = max(0, effective_overlap)
        advance = chunk_size - effective_overlap
        if advance <= 0:
            advance = 1
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += advance
        return chunks
