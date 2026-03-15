"""Tests for rule library: store CRUD, deduplication, index retrieval."""

import json
import os

import pytest

from src.library.rule import VerifiedRule
from src.library.store import RuleStore
from src.library.index import RuleIndex


class TestRuleStore:
    """Tests for RuleStore CRUD operations."""

    def test_add_and_get(self, tmp_store, sample_rules):
        """Adding a rule should make it retrievable."""
        rule = sample_rules[0]
        tmp_store.add(rule)
        retrieved = tmp_store.get(rule.rule_id)
        assert retrieved is not None
        assert retrieved.rule_id == rule.rule_id
        assert retrieved.source_code == rule.source_code

    def test_size(self, populated_store, sample_rules):
        """Size should reflect number of added rules."""
        assert populated_store.size == len(sample_rules)

    def test_remove(self, populated_store, sample_rules):
        """Removing a rule should decrease size."""
        rule_id = sample_rules[0].rule_id
        initial_size = populated_store.size
        result = populated_store.remove(rule_id)
        assert result is True
        assert populated_store.size == initial_size - 1
        assert populated_store.get(rule_id) is None

    def test_remove_nonexistent(self, tmp_store):
        """Removing non-existent rule should return False."""
        result = tmp_store.remove("nonexistent")
        assert result is False

    def test_list_all(self, populated_store, sample_rules):
        """list_all should return all rules."""
        all_rules = populated_store.list_all()
        assert len(all_rules) == len(sample_rules)

    def test_list_by_domain(self, populated_store):
        """list_by_domain should filter correctly."""
        math_rules = populated_store.list_by_domain("math")
        string_rules = populated_store.list_by_domain("string")
        assert len(math_rules) == 3  # double, square, sum
        assert len(string_rules) == 1  # reverse

    def test_persistence(self, tmp_path, sample_rules):
        """Rules should persist across store instances."""
        path = str(tmp_path / "persist_test.json")

        # Write
        store1 = RuleStore(path=path)
        for rule in sample_rules:
            store1.add(rule)
        assert os.path.exists(path)

        # Read
        store2 = RuleStore(path=path)
        assert store2.size == len(sample_rules)
        for rule in sample_rules:
            retrieved = store2.get(rule.rule_id)
            assert retrieved is not None

    def test_in_memory_store(self, sample_rules):
        """Store without path should work in-memory."""
        store = RuleStore()  # no path
        for rule in sample_rules:
            store.add(rule)
        assert store.size == len(sample_rules)


class TestDeduplication:
    """Tests for rule deduplication."""

    def test_deduplicate_identical_rules(self, tmp_store):
        """Duplicate rules should be deduplicated."""
        rule1 = VerifiedRule(
            rule_id="r1",
            domain="math",
            description="Double",
            source_code="def rule(x):\n    return x * 2\n",
            accuracy=0.8,
        )
        rule2 = VerifiedRule(
            rule_id="r2",
            domain="math",
            description="Also double",
            source_code="def rule(x):\n    return x * 2\n",  # same code
            accuracy=1.0,
        )
        tmp_store.add(rule1)
        tmp_store.add(rule2)
        assert tmp_store.size == 2

        removed = tmp_store.deduplicate()
        assert removed == 1
        assert tmp_store.size == 1

        # Should keep the higher accuracy one
        remaining = tmp_store.list_all()[0]
        assert remaining.accuracy == 1.0

    def test_deduplicate_unique_rules(self, populated_store):
        """Unique rules should not be affected by deduplication."""
        initial_size = populated_store.size
        removed = populated_store.deduplicate()
        assert removed == 0
        assert populated_store.size == initial_size


class TestRuleIndex:
    """Tests for keyword-based rule retrieval."""

    def test_build_index(self, populated_store):
        """Building index should not raise."""
        index = RuleIndex(store=populated_store)
        assert len(index._index) > 0

    def test_retrieve_by_keyword(self, populated_store):
        """Retrieval should find relevant rules."""
        index = RuleIndex(store=populated_store)

        results = index.retrieve("arithmetic double number", k=3)
        assert len(results) > 0
        # "double" should match the doubling rule
        rule_ids = [r.rule_id for r in results]
        assert "rule_double" in rule_ids

    def test_retrieve_domain_match(self, populated_store):
        """Retrieval should prefer domain matches."""
        index = RuleIndex(store=populated_store)

        results = index.retrieve("string reverse", k=3)
        assert len(results) > 0
        # String reverse rule should rank high
        rule_ids = [r.rule_id for r in results]
        assert "rule_reverse" in rule_ids

    def test_retrieve_limited_k(self, populated_store):
        """Retrieval should respect the k limit."""
        index = RuleIndex(store=populated_store)
        results = index.retrieve("math number", k=2)
        assert len(results) <= 2

    def test_retrieve_no_matches(self, populated_store):
        """Query with no matches should return empty list."""
        index = RuleIndex(store=populated_store)
        results = index.retrieve("zzznonexistenttopic", k=3)
        assert len(results) == 0

    def test_empty_store_index(self):
        """Index on empty store should work without errors."""
        store = RuleStore()
        index = RuleIndex(store=store)
        results = index.retrieve("anything", k=3)
        assert results == []


class TestVerifiedRule:
    """Tests for the VerifiedRule dataclass."""

    def test_to_dict_and_back(self, sample_rules):
        """Serialization round-trip should preserve data."""
        for rule in sample_rules:
            d = rule.to_dict()
            restored = VerifiedRule.from_dict(d)
            assert restored.rule_id == rule.rule_id
            assert restored.source_code == rule.source_code
            assert restored.accuracy == rule.accuracy

    def test_code_hash_consistent(self, sample_rules):
        """Code hash should be deterministic."""
        rule = sample_rules[0]
        h1 = rule.code_hash
        h2 = rule.code_hash
        assert h1 == h2

    def test_code_hash_different_for_different_code(self, sample_rules):
        """Different code should produce different hashes."""
        hashes = set(r.code_hash for r in sample_rules)
        assert len(hashes) == len(sample_rules)
