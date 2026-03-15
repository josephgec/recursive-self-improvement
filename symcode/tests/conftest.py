"""Shared fixtures for SymCode tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def prompts_dir() -> Path:
    """Path to the prompts config directory."""
    return Path(__file__).resolve().parents[1] / "configs" / "prompts"


@pytest.fixture
def sample_problems() -> list[dict[str, Any]]:
    """A small set of sample math problems for testing."""
    return [
        {
            "problem": "What is 2 + 3?",
            "answer": "5",
            "subject": "prealgebra",
        },
        {
            "problem": "Solve for x: x^2 - 4 = 0",
            "answer": "[-2, 2]",
            "subject": "algebra",
        },
        {
            "problem": "Find the area of a circle with radius 5.",
            "answer": "25*pi",
            "subject": "geometry",
        },
        {
            "problem": "What is the remainder when 17^100 is divided by 5?",
            "answer": "2",
            "subject": "number_theory",
        },
        {
            "problem": "What is the derivative of x^3?",
            "answer": "3*x**2",
            "subject": "precalculus",
        },
        {
            "problem": "If a fair coin is flipped 3 times, what is the probability of getting exactly 2 heads?",
            "answer": "3/8",
            "subject": "counting_and_probability",
        },
    ]


@pytest.fixture
def mock_openai_client():
    """A mock OpenAI client that returns a code block."""

    def make_mock(code: str = "", answer: str = "42"):
        if not code:
            code = (
                "from sympy import *\n\n"
                f"answer = {answer}\n"
                f'print(f"Answer: {{answer}}")\n'
            )
        response_text = f"```python\n{code}```"

        client = MagicMock()
        choice = MagicMock()
        choice.message.content = response_text
        response = MagicMock()
        response.choices = [choice]
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        client.chat.completions.create.return_value = response
        return client

    return make_mock


@pytest.fixture
def mock_llm_response():
    """Helper to build a mock LLM response dict."""

    def _make(code: str, answer: str = "42"):
        return {
            "content": f"```python\n{code}\n```",
            "prompt_tokens": 100,
            "completion_tokens": 50,
        }

    return _make
