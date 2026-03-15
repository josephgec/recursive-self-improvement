"""Shared fixtures for CTM Integration tests."""

import os
import sys
import tempfile

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bdm.ctm_table import CTMTable
from src.bdm.scorer import BDMScorer
from src.bdm.block_decomposer import BlockDecomposer
from src.synthesis.candidate_generator import CandidateGenerator, IOExample
from src.synthesis.empirical_verifier import EmpiricalVerifier
from src.library.rule import VerifiedRule
from src.library.store import RuleStore


@pytest.fixture
def ctm_table():
    """A CTM table with fallback values only (fast)."""
    return CTMTable.with_fallback_only()


@pytest.fixture
def built_ctm_table():
    """A CTM table built from 1-state 2-symbol TMs (small but real)."""
    table = CTMTable()
    table.build(max_states=1, max_symbols=2, max_steps=20, block_size=8)
    return table


@pytest.fixture
def scorer(ctm_table):
    """BDM scorer with fallback CTM table."""
    return BDMScorer(ctm_table=ctm_table, block_size=4)


@pytest.fixture
def decomposer(ctm_table):
    """Block decomposer with fallback CTM table."""
    return BlockDecomposer(ctm_table=ctm_table, block_size=4)


@pytest.fixture
def generator():
    """Candidate generator with mock LLM."""
    return CandidateGenerator(domain="math")


@pytest.fixture
def verifier():
    """Empirical verifier."""
    return EmpiricalVerifier()


@pytest.fixture
def linear_examples():
    """I/O examples for y = 2x + 1."""
    return [
        IOExample(input=0, output=1, domain="math"),
        IOExample(input=1, output=3, domain="math"),
        IOExample(input=2, output=5, domain="math"),
        IOExample(input=3, output=7, domain="math"),
        IOExample(input=5, output=11, domain="math"),
    ]


@pytest.fixture
def double_examples():
    """I/O examples for y = 2x."""
    return [
        IOExample(input=1, output=2, domain="math"),
        IOExample(input=2, output=4, domain="math"),
        IOExample(input=3, output=6, domain="math"),
        IOExample(input=5, output=10, domain="math"),
        IOExample(input=10, output=20, domain="math"),
    ]


@pytest.fixture
def sum_examples():
    """I/O examples for sum(list)."""
    return [
        IOExample(input=[1, 2, 3], output=6, domain="math"),
        IOExample(input=[10, 20], output=30, domain="math"),
        IOExample(input=[5], output=5, domain="math"),
        IOExample(input=[1, 1, 1, 1], output=4, domain="math"),
    ]


@pytest.fixture
def quadratic_examples():
    """I/O examples for y = x^2."""
    return [
        IOExample(input=0, output=0, domain="math"),
        IOExample(input=1, output=1, domain="math"),
        IOExample(input=2, output=4, domain="math"),
        IOExample(input=3, output=9, domain="math"),
        IOExample(input=4, output=16, domain="math"),
    ]


@pytest.fixture
def tmp_store(tmp_path):
    """Temporary rule store."""
    store_path = str(tmp_path / "test_rules.json")
    return RuleStore(path=store_path)


@pytest.fixture
def sample_rules():
    """Sample verified rules for testing."""
    return [
        VerifiedRule(
            rule_id="rule_double",
            domain="math",
            description="Doubles a number",
            source_code="def rule(x):\n    return x * 2\n",
            accuracy=1.0,
            bdm_score=15.0,
            mdl_score=15.0,
            tags=["math", "arithmetic", "double"],
        ),
        VerifiedRule(
            rule_id="rule_square",
            domain="math",
            description="Squares a number",
            source_code="def rule(x):\n    return x ** 2\n",
            accuracy=1.0,
            bdm_score=16.0,
            mdl_score=16.0,
            tags=["math", "arithmetic", "square", "power"],
        ),
        VerifiedRule(
            rule_id="rule_reverse",
            domain="string",
            description="Reverses a string",
            source_code="def rule(x):\n    return x[::-1]\n",
            accuracy=1.0,
            bdm_score=18.0,
            mdl_score=18.0,
            tags=["string", "reverse"],
        ),
        VerifiedRule(
            rule_id="rule_sum",
            domain="math",
            description="Sums a list of numbers",
            source_code="def rule(x):\n    return sum(x)\n",
            accuracy=1.0,
            bdm_score=14.0,
            mdl_score=14.0,
            tags=["math", "list", "sum", "aggregate"],
        ),
    ]


@pytest.fixture
def populated_store(tmp_store, sample_rules):
    """Store populated with sample rules."""
    for rule in sample_rules:
        tmp_store.add(rule)
    return tmp_store
