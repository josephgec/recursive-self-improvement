"""Tests for ContextPartitioner."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.recursion.partitioner import ContextPartitioner


class TestEqualChunks:
    def test_equal_split(self):
        text = "A" * 100
        chunks = ContextPartitioner.equal_chunks(text, 4)
        assert len(chunks) == 4
        assert all(len(c) == 25 for c in chunks)

    def test_single_chunk(self):
        text = "hello"
        chunks = ContextPartitioner.equal_chunks(text, 1)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_zero_chunks(self):
        text = "hello"
        chunks = ContextPartitioner.equal_chunks(text, 0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_more_chunks_than_chars(self):
        text = "abc"
        chunks = ContextPartitioner.equal_chunks(text, 10)
        assert len(chunks) > 0
        assert "".join(chunks) == text


class TestLineBased:
    def test_line_based(self):
        text = "\n".join(f"line {i}" for i in range(20))
        chunks = ContextPartitioner.line_based(text, lines_per_chunk=5)
        assert len(chunks) == 4

    def test_single_line(self):
        chunks = ContextPartitioner.line_based("hello", lines_per_chunk=10)
        assert len(chunks) == 1

    def test_empty_text(self):
        chunks = ContextPartitioner.line_based("", lines_per_chunk=5)
        assert len(chunks) == 1


class TestParagraphBased:
    def test_paragraphs(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = ContextPartitioner.paragraph_based(text)
        assert len(chunks) == 3

    def test_no_paragraphs(self):
        text = "Just one block of text."
        chunks = ContextPartitioner.paragraph_based(text)
        assert len(chunks) == 1

    def test_multiple_blank_lines(self):
        text = "A\n\n\n\nB"
        chunks = ContextPartitioner.paragraph_based(text)
        assert len(chunks) == 2


class TestSemanticSections:
    def test_markdown_headings(self):
        text = "# Section 1\nContent 1\n# Section 2\nContent 2\n# Section 3\nContent 3"
        chunks = ContextPartitioner.semantic_sections(text)
        assert len(chunks) >= 2

    def test_no_headings(self):
        text = "Just plain text without any headings."
        chunks = ContextPartitioner.semantic_sections(text)
        assert len(chunks) >= 1

    def test_chapter_headings(self):
        text = "Chapter 1: Intro\nContent\nChapter 2: Middle\nContent\nChapter 3: End\nContent"
        chunks = ContextPartitioner.semantic_sections(text)
        assert len(chunks) >= 2


class TestTokenBudgetChunks:
    def test_within_budget(self):
        text = "short text"
        chunks = ContextPartitioner.token_budget_chunks(text, token_budget=100)
        assert len(chunks) == 1

    def test_exceeds_budget(self):
        text = "\n".join(["word " * 50] * 20)
        chunks = ContextPartitioner.token_budget_chunks(text, token_budget=100)
        assert len(chunks) > 1

    def test_empty_text(self):
        chunks = ContextPartitioner.token_budget_chunks("", token_budget=100)
        assert len(chunks) == 1


class TestByStructure:
    def test_dict(self):
        data = {"a": 1, "b": 2, "c": 3}
        chunks = ContextPartitioner.by_structure(data)
        assert len(chunks) == 3
        for c in chunks:
            parsed = json.loads(c)
            assert isinstance(parsed, dict)

    def test_list(self):
        data = [1, 2, 3]
        chunks = ContextPartitioner.by_structure(data)
        assert len(chunks) == 3

    def test_non_iterable(self):
        chunks = ContextPartitioner.by_structure("plain string")
        assert len(chunks) == 1
