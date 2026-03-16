"""Tests for ContextLoader."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.context_loader import ContextLoader, ContextMeta


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestContextLoaderString:
    def test_load_string(self):
        loader = ContextLoader()
        repl: dict = {}
        meta = loader.load_into_repl("Hello, world!", repl)

        assert repl["CONTEXT"] == "Hello, world!"
        assert repl["CONTEXT_TYPE"] == "str"
        assert repl["CONTEXT_LENGTH"] == 13
        assert meta.context_type == "str"
        assert meta.size_chars == 13
        assert meta.size_tokens > 0

    def test_load_empty_string(self):
        loader = ContextLoader()
        repl: dict = {}
        meta = loader.load_into_repl("", repl)

        assert repl["CONTEXT"] == ""
        assert meta.size_chars == 0

    def test_load_multiline_string(self):
        loader = ContextLoader()
        repl: dict = {}
        text = "line1\nline2\nline3"
        meta = loader.load_into_repl(text, repl)

        assert meta.num_lines == 3
        assert repl["CONTEXT"] == text


class TestContextLoaderList:
    def test_load_list(self):
        loader = ContextLoader()
        repl: dict = {}
        data = ["item1", "item2", "item3"]
        meta = loader.load_into_repl(data, repl)

        assert meta.context_type == "list"
        assert repl["CONTEXT_TYPE"] == "list"
        assert "item1" in repl["CONTEXT"]
        assert "item2" in repl["CONTEXT"]

    def test_load_list_of_numbers(self):
        loader = ContextLoader()
        repl: dict = {}
        data = [1, 2, 3, 4, 5]
        meta = loader.load_into_repl(data, repl)

        assert meta.context_type == "list"
        assert "1" in repl["CONTEXT"]


class TestContextLoaderDict:
    def test_load_dict(self):
        loader = ContextLoader()
        repl: dict = {}
        data = {"key": "value", "number": 42}
        meta = loader.load_into_repl(data, repl)

        assert meta.context_type == "dict"
        assert repl["CONTEXT_TYPE"] == "dict"
        parsed = json.loads(repl["CONTEXT"])
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_load_nested_dict(self):
        loader = ContextLoader()
        repl: dict = {}
        data = {"a": {"b": {"c": 1}}}
        meta = loader.load_into_repl(data, repl)

        assert meta.context_type == "dict"
        parsed = json.loads(repl["CONTEXT"])
        assert parsed["a"]["b"]["c"] == 1


class TestContextLoaderFile:
    def test_load_from_path(self):
        loader = ContextLoader()
        repl: dict = {}
        path = FIXTURES_DIR / "small_context.txt"
        meta = loader.load_into_repl(path, repl)

        assert meta.context_type == "file"
        assert meta.source == str(path)
        assert "FOUND_IT_42" in repl["CONTEXT"]

    def test_load_from_file_method(self):
        loader = ContextLoader()
        repl: dict = {}
        path = FIXTURES_DIR / "small_context.txt"
        meta = loader.load_from_file(path, repl)

        assert "FOUND_IT_42" in repl["CONTEXT"]
        assert meta.size_chars > 0


class TestContextLoaderChunked:
    def test_load_chunked_small(self):
        loader = ContextLoader(max_chunk_size=100, overlap=20)
        repl: dict = {}
        text = "A" * 50
        meta = loader.load_chunked(text, repl)

        assert "CONTEXT_CHUNKS" in repl
        assert len(repl["CONTEXT_CHUNKS"]) == 1

    def test_load_chunked_large(self):
        loader = ContextLoader(max_chunk_size=100, overlap=20)
        repl: dict = {}
        text = "A" * 300
        meta = loader.load_chunked(text, repl)

        assert len(repl["CONTEXT_CHUNKS"]) > 1
        assert meta.num_chunks == len(repl["CONTEXT_CHUNKS"])


class TestContextMeta:
    def test_meta_fields(self):
        loader = ContextLoader()
        repl: dict = {}
        meta = loader.load_into_repl("test content here", repl)

        assert isinstance(meta, ContextMeta)
        assert meta.context_type == "str"
        assert meta.size_chars == len("test content here")
        assert meta.size_tokens > 0
        assert meta.num_lines >= 1

    def test_repl_metadata_variables(self):
        loader = ContextLoader()
        repl: dict = {}
        loader.load_into_repl("test", repl)

        assert "CONTEXT_META" in repl
        assert isinstance(repl["CONTEXT_META"], ContextMeta)
        assert repl["CONTEXT_SIZE_TOKENS"] > 0


class TestEstimateTokens:
    def test_estimate_tokens(self):
        assert ContextLoader.estimate_tokens("") == 0
        assert ContextLoader.estimate_tokens("hello") > 0
        assert ContextLoader.estimate_tokens("a" * 100) == 25
