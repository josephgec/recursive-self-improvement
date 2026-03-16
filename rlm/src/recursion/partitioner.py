"""ContextPartitioner: split context into pieces for recursive processing."""

from __future__ import annotations

import re
from typing import List

from src.utils.token_counter import estimate_tokens


class ContextPartitioner:
    """Various strategies for splitting context into sub-pieces."""

    @staticmethod
    def equal_chunks(text: str, num_chunks: int) -> List[str]:
        """Split text into *num_chunks* roughly equal parts."""
        if num_chunks <= 0:
            return [text]
        chunk_size = max(1, len(text) // num_chunks)
        chunks: List[str] = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < num_chunks - 1 else len(text)
            chunks.append(text[start:end])
        return [c for c in chunks if c]

    @staticmethod
    def line_based(text: str, lines_per_chunk: int = 100) -> List[str]:
        """Split text by line count."""
        lines = text.split("\n")
        chunks: List[str] = []
        for i in range(0, len(lines), lines_per_chunk):
            chunk = "\n".join(lines[i: i + lines_per_chunk])
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text]

    @staticmethod
    def paragraph_based(text: str) -> List[str]:
        """Split text by double-newlines (paragraphs)."""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    @staticmethod
    def semantic_sections(text: str) -> List[str]:
        """Split text by markdown-style headings or obvious section breaks."""
        # Split on markdown headings
        parts = re.split(r"(?m)^(#{1,6}\s+.+)$", text)
        if len(parts) <= 1:
            # Try splitting on "Section", "Chapter", etc.
            parts = re.split(r"(?m)^((?:Section|Chapter|Part)\s+\d+.*?)$", text, flags=re.IGNORECASE)
        if len(parts) <= 1:
            return [text]
        # Rejoin heading with following content
        sections: List[str] = []
        current = ""
        for part in parts:
            part_stripped = part.strip()
            if not part_stripped:
                continue
            if re.match(r"^#{1,6}\s+", part_stripped) or re.match(
                r"^(?:Section|Chapter|Part)\s+\d+", part_stripped, re.IGNORECASE
            ):
                if current.strip():
                    sections.append(current.strip())
                current = part_stripped + "\n"
            else:
                current += part_stripped + "\n"
        if current.strip():
            sections.append(current.strip())
        return sections if sections else [text]

    @staticmethod
    def token_budget_chunks(text: str, token_budget: int = 4000) -> List[str]:
        """Split text so each chunk is within *token_budget* tokens."""
        lines = text.split("\n")
        chunks: List[str] = []
        current_lines: List[str] = []
        current_tokens = 0
        for line in lines:
            line_tokens = estimate_tokens(line)
            if current_tokens + line_tokens > token_budget and current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_tokens = 0
            current_lines.append(line)
            current_tokens += line_tokens
        if current_lines:
            chunks.append("\n".join(current_lines))
        return chunks if chunks else [text]

    @staticmethod
    def by_structure(data: dict | list) -> List[str]:
        """Split structured data (dict/list) into one chunk per top-level key or element."""
        import json

        if isinstance(data, dict):
            return [json.dumps({k: v}, indent=2) for k, v in data.items()]
        if isinstance(data, list):
            return [json.dumps(item, indent=2) for item in data]
        return [str(data)]
