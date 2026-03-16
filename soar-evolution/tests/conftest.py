"""Shared test fixtures and utilities."""

import sys
import os
import json
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.arc.grid import Grid, ARCExample, ARCTask
from src.arc.loader import ARCLoader, BUILTIN_TASKS
from src.population.individual import Individual

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def simple_grid():
    """A simple 3x3 grid."""
    return Grid.from_list([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@pytest.fixture
def small_grid():
    """A small 2x2 grid."""
    return Grid.from_list([[1, 2], [3, 4]])


@pytest.fixture
def zero_grid():
    """A 3x3 grid of zeros."""
    return Grid.zeros(3, 3)


@pytest.fixture
def color_swap_task():
    """The simple_color_swap task."""
    loader = ARCLoader()
    return loader.load_task("simple_color_swap")


@pytest.fixture
def pattern_fill_task():
    """The pattern_fill task."""
    loader = ARCLoader()
    return loader.load_task("pattern_fill")


@pytest.fixture
def grid_transform_task():
    """The grid_transform task."""
    loader = ARCLoader()
    return loader.load_task("grid_transform")


@pytest.fixture
def all_tasks():
    """All built-in tasks."""
    loader = ARCLoader()
    return loader.load_all()


@pytest.fixture
def sample_individual():
    """A sample individual with identity transform."""
    return Individual(
        code="def transform(input_grid):\n    return [row[:] for row in input_grid]\n",
        generation=0,
        operator="test",
    )


@pytest.fixture
def good_color_swap_individual():
    """An individual that correctly solves the color swap task."""
    code = """def transform(input_grid):
    result = []
    for row in input_grid:
        new_row = []
        for cell in row:
            if cell == 1:
                new_row.append(2)
            else:
                new_row.append(cell)
        result.append(new_row)
    return result
"""
    return Individual(code=code, generation=1, operator="test_good")


@pytest.fixture
def bad_individual():
    """An individual with a compile error."""
    return Individual(
        code="def transform(input_grid):\n    return None\n",
        generation=0,
        operator="test_bad",
    )


@pytest.fixture
def error_individual():
    """An individual with a runtime error."""
    return Individual(
        code="def transform(input_grid):\n    return input_grid[999]\n",
        generation=0,
        operator="test_error",
    )


@pytest.fixture
def mock_llm():
    """A mock LLM function for testing."""
    def _mock(prompt: str) -> str:
        return '''def transform(input_grid):
    result = []
    for row in input_grid:
        new_row = []
        for cell in row:
            if cell == 1:
                new_row.append(2)
            else:
                new_row.append(cell)
        result.append(new_row)
    return result
'''
    return _mock


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def loader(fixtures_dir):
    """An ARCLoader configured with the fixtures directory."""
    return ARCLoader(data_dir=str(fixtures_dir))
