"""Test fixtures: MockLLM, MockREPL, sample contexts."""

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Ensure the project root is importable
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class MockLLM:
    """Deterministic mock LLM that generates Python code to explore context.

    Behavior:
    - Iteration 1: peek at the context to understand its structure
    - Iteration 2: grep/search for relevant information based on the query
    - Iteration 3+: produce FINAL with the answer

    For sub-queries, it processes the slice directly and calls FINAL immediately.
    """

    def __init__(self, max_before_final: int = 3) -> None:
        self._call_count = 0
        self.max_before_final = max_before_final
        self.history: List[Dict[str, str]] = []

    def chat(self, messages: List[Dict[str, str]], context_meta: Any = None) -> str:
        """Generate a deterministic code response."""
        self._call_count += 1
        self.history = messages

        # Extract the query from user messages
        query = ""
        for m in messages:
            if m["role"] == "user" and not m["content"].startswith("["):
                query = m["content"]
                break

        # Check if we got feedback indicating we should finalize
        last_msg = messages[-1]["content"] if messages else ""
        has_output = last_msg.startswith("[stdout]") or last_msg.startswith("[error]")

        # Check for budget warning
        has_budget_warning = any(
            "WARNING" in m["content"] and "iterations left" in m["content"]
            for m in messages
            if m["role"] == "system"
        )

        # Determine context type from meta or messages
        is_json = False
        for m in messages:
            if "dict" in m.get("content", "") or "json" in m.get("content", "").lower():
                is_json = True
                break

        # Strategy: peek -> grep -> FINAL
        if self._call_count == 1 and not has_budget_warning:
            return self._gen_peek_code(query)
        elif self._call_count == 2 and not has_budget_warning:
            return self._gen_search_code(query, is_json)
        else:
            return self._gen_final_code(query, last_msg, is_json)

    def complete(self, prompt: str, context_meta: Any = None) -> str:
        """Simple completion interface — delegates to chat."""
        return self.chat([{"role": "user", "content": prompt}], context_meta)

    def reset(self) -> None:
        """Reset call counter."""
        self._call_count = 0
        self.history = []

    # ------------------------------------------------------------------
    # Code generation helpers
    # ------------------------------------------------------------------

    def _gen_peek_code(self, query: str) -> str:
        return (
            "```python\n"
            "# Peek at context structure\n"
            "preview = peek(0, 300)\n"
            "print(preview)\n"
            "print(f'Total lines: {count_lines()}')\n"
            "```"
        )

    def _gen_search_code(self, query: str, is_json: bool) -> str:
        # Extract a search term from the query
        search_term = self._extract_search_term(query)

        if is_json:
            return (
                "```python\n"
                "import json\n"
                "data = json.loads(CONTEXT)\n"
                f"results = grep('{search_term}')\n"
                "for r in results[:5]:\n"
                "    print(r)\n"
                "```"
            )
        return (
            "```python\n"
            f"results = grep('{search_term}')\n"
            "for r in results[:10]:\n"
            "    print(r)\n"
            "```"
        )

    def _gen_final_code(self, query: str, last_output: str, is_json: bool) -> str:
        # Try to extract a useful answer from the last output
        if last_output.startswith("[stdout]"):
            output_text = last_output.replace("[stdout]\n", "")
            # Try to find a specific value in the output
            extracted = self._extract_answer(output_text, query)
            if extracted:
                return (
                    "```python\n"
                    f'FINAL("{extracted}")\n'
                    "```"
                )

        # Fallback: search and produce answer
        search_term = self._extract_search_term(query)
        return (
            "```python\n"
            f"results = search('{search_term}')\n"
            "if results:\n"
            "    answer = results[0].split(': ', 1)[-1] if ': ' in results[0] else results[0]\n"
            "else:\n"
            "    answer = 'not found'\n"
            'FINAL_VAR("answer")\n'
            "```"
        )

    def _extract_search_term(self, query: str) -> str:
        """Extract a useful search term from the query."""
        # Look for quoted strings
        quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", query)
        if quoted:
            return quoted[0][0] or quoted[0][1]

        # Look for specific patterns
        patterns = [
            r"(?:find|search|locate|look for)\s+(?:the\s+)?(.+?)(?:\s+and|\s+in|\.|$)",
            r"(?:containing|named|called)\s+['\"]?(\w+)['\"]?",
            r"SECRET\w*",
            r"NEEDLE\w*",
            r"specific_marker",
        ]
        for pat in patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                return m.group(0) if not m.groups() else m.group(1)

        # Fall back to significant words from the query
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "with", "and", "or", "how", "many", "what",
            "which", "find", "this", "that", "from", "does", "do", "count",
        }
        words = [w.strip("?.,!") for w in query.lower().split()]
        meaningful = [w for w in words if w not in stop_words and len(w) > 2]
        if meaningful:
            return meaningful[0]
        return "result"

    def _extract_answer(self, output: str, query: str) -> Optional[str]:
        """Try to extract a concise answer from execution output."""
        lines = output.strip().split("\n")
        if not lines:
            return None

        # If it's a single short line, use it
        if len(lines) == 1 and len(lines[0]) < 200:
            return lines[0].strip()

        # Look for lines with specific patterns (values, numbers, etc.)
        for line in lines:
            # Check for key: value patterns
            if ": " in line:
                parts = line.split(": ", 1)
                if len(parts[1].strip()) < 100:
                    return parts[1].strip()

        # Return first meaningful line
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and len(stripped) < 200:
                return stripped

        return lines[0][:100] if lines else None


class MockLLMImmediate(MockLLM):
    """Mock LLM that produces FINAL on the first call."""

    def chat(self, messages: List[Dict[str, str]], context_meta: Any = None) -> str:
        return (
            "```python\n"
            "results = search('secret')\n"
            "answer = results[0] if results else 'none'\n"
            'FINAL_VAR("answer")\n'
            "```"
        )


class MockLLMNeverFinal(MockLLM):
    """Mock LLM that never produces FINAL (for budget enforcement testing)."""

    def chat(self, messages: List[Dict[str, str]], context_meta: Any = None) -> str:
        self._call_count += 1
        return (
            "```python\n"
            f"# Iteration {self._call_count}\n"
            "result = peek(0, 100)\n"
            "print(result)\n"
            "```"
        )


class MockLLMChunking(MockLLM):
    """Mock LLM that uses chunk-based (map-reduce) strategy."""

    def chat(self, messages: List[Dict[str, str]], context_meta: Any = None) -> str:
        self._call_count += 1
        if self._call_count == 1:
            return (
                "```python\n"
                "chunks = chunk(2000)\n"
                "print(f'Split into {len(chunks)} chunks')\n"
                "counts = []\n"
                "for c in chunks:\n"
                "    counts.append(len(c.split()))\n"
                "result = sum(counts)\n"
                "print(f'Total words: {result}')\n"
                "```"
            )
        return (
            "```python\n"
            'FINAL_VAR("result")\n'
            "```"
        )


class MockLLMSubQuery(MockLLM):
    """Mock LLM that uses rlm_sub_query for hierarchical strategy."""

    def __init__(self) -> None:
        super().__init__()
        self._is_child = False

    def chat(self, messages: List[Dict[str, str]], context_meta: Any = None) -> str:
        self._call_count += 1
        # Check if this is a sub-query context
        for m in messages:
            if "Sub-Task" in m.get("content", ""):
                self._is_child = True

        if self._is_child:
            return (
                "```python\n"
                "result = search('secret')\n"
                "answer = result[0] if result else 'none'\n"
                'FINAL_VAR("answer")\n'
                "```"
            )

        if self._call_count == 1:
            return (
                "```python\n"
                "r = rlm_sub_query(query='find the secret', context=CONTEXT[:2000])\n"
                "result = r\n"
                "print(f'Sub-query result: {result}')\n"
                "```"
            )
        return (
            "```python\n"
            'FINAL_VAR("result")\n'
            "```"
        )


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_llm_immediate():
    return MockLLMImmediate()


@pytest.fixture
def mock_llm_never_final():
    return MockLLMNeverFinal()


@pytest.fixture
def mock_llm_chunking():
    return MockLLMChunking()


@pytest.fixture
def mock_llm_sub_query():
    return MockLLMSubQuery()


@pytest.fixture
def small_context():
    return (FIXTURES_DIR / "small_context.txt").read_text()


@pytest.fixture
def medium_context():
    return (FIXTURES_DIR / "medium_context.txt").read_text()


@pytest.fixture
def large_context():
    return json.loads((FIXTURES_DIR / "large_context.json").read_text())


@pytest.fixture
def large_context_str():
    return (FIXTURES_DIR / "large_context.json").read_text()


@pytest.fixture
def sample_haystack():
    from src.utils.context_generators import generate_haystack
    return generate_haystack(
        needle="The answer is: NEEDLE_42",
        haystack_size=5000,
        position=0.5,
    )
