"""Helper: chunk — split context into pieces for map-reduce style work."""

from __future__ import annotations

from typing import Any, Dict, List


def chunk(
    repl: Dict[str, Any],
    chunk_size: int = 4000,
    overlap: int = 200,
) -> List[str]:
    """Split CONTEXT into overlapping chunks of *chunk_size* characters."""
    ctx: str = repl.get("CONTEXT", "")
    if len(ctx) <= chunk_size:
        return [ctx]
    # Ensure overlap doesn't exceed chunk_size to prevent infinite loop
    effective_overlap = min(overlap, chunk_size - 1) if chunk_size > 1 else 0
    effective_overlap = max(0, effective_overlap)
    chunks: List[str] = []
    start = 0
    while start < len(ctx):
        end = start + chunk_size
        chunks.append(ctx[start:end])
        advance = chunk_size - effective_overlap
        if advance <= 0:
            advance = 1
        start += advance
    return chunks


def count_lines(repl: Dict[str, Any]) -> int:
    """Return the number of lines in CONTEXT."""
    ctx: str = repl.get("CONTEXT", "")
    if not ctx:
        return 0
    return len(ctx.split("\n"))


def make_chunk(repl: Dict[str, Any]):
    """Return chunk and count_lines functions bound to *repl*."""

    def _chunk(chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        return chunk(repl, chunk_size, overlap)

    def _count_lines() -> int:
        return count_lines(repl)

    return _chunk, _count_lines
