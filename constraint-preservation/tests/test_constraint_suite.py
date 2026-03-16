"""Tests for ConstraintSuite, ConstraintRunner, SuiteVerdict."""

import pytest
from tests.conftest import MockAgent
from src.checker.suite import ConstraintSuite
from src.checker.runner import ConstraintRunner
from src.checker.verdict import SuiteVerdict
from src.checker.cache import ConstraintCache
from src.constraints.base import CheckContext, ConstraintResult
from src.constraints.custom import CustomConstraint


class TestConstraintSuite:
    """Tests for ConstraintSuite."""

    def test_default_suite_builds_all_constraints(self):
        """Default config enables all 7 built-in constraints."""
        suite = ConstraintSuite()
        assert len(suite) == 7

    def test_get_by_name(self):
        """Retrieve a constraint by name."""
        suite = ConstraintSuite()
        acc = suite.get_by_name("accuracy_floor")
        assert acc is not None
        assert acc.name == "accuracy_floor"

    def test_get_by_name_missing(self):
        """Missing name returns None."""
        suite = ConstraintSuite()
        assert suite.get_by_name("nonexistent") is None

    def test_get_by_category(self):
        """Filter constraints by category."""
        suite = ConstraintSuite()
        quality = suite.get_by_category("quality")
        assert len(quality) == 6  # all except safety_eval

        safety = suite.get_by_category("safety")
        assert len(safety) == 1
        assert safety[0].name == "safety_eval"

    def test_immutable_tuple(self):
        """Constraints are stored as an immutable tuple."""
        suite = ConstraintSuite()
        assert isinstance(suite.constraints, tuple)

    def test_iter(self):
        """Suite is iterable."""
        suite = ConstraintSuite()
        names = [c.name for c in suite]
        assert "accuracy_floor" in names

    def test_disabled_constraint(self):
        """Disabled constraints are excluded."""
        config = {
            "constraints": {
                "accuracy_floor": {"enabled": False},
            }
        }
        suite = ConstraintSuite(config)
        assert suite.get_by_name("accuracy_floor") is None
        assert len(suite) == 6

    def test_add_custom(self):
        """Adding a custom constraint returns a new suite."""
        suite = ConstraintSuite()
        original_len = len(suite)

        custom = CustomConstraint(
            name="custom_test",
            description="test",
            category="custom",
            threshold=0.5,
            check_fn=lambda a, c: ConstraintResult(True, 1.0, 0.5, 0.5),
        )
        new_suite = suite.add_custom(custom)

        assert len(new_suite) == original_len + 1
        assert new_suite.get_by_name("custom_test") is not None
        # Original is unchanged
        assert len(suite) == original_len


class TestSuiteVerdict:
    """Tests for SuiteVerdict."""

    def test_all_pass(self):
        """All-pass verdict."""
        results = {
            "a": ConstraintResult(True, 0.9, 0.8, 0.1),
            "b": ConstraintResult(True, 0.5, 0.4, 0.1),
        }
        v = SuiteVerdict(passed=True, results=results)
        assert v.passed is True
        assert len(v.violations) == 0

    def test_one_fail(self):
        """One failure makes the verdict fail."""
        results = {
            "a": ConstraintResult(True, 0.9, 0.8, 0.1),
            "b": ConstraintResult(False, 0.3, 0.4, -0.1),
        }
        v = SuiteVerdict(passed=False, results=results)
        assert v.passed is False
        assert len(v.violations) == 1
        assert "b" in v.violations

    def test_closest_to_violation(self):
        """Closest-to-violation is the passing constraint with smallest headroom."""
        results = {
            "a": ConstraintResult(True, 0.85, 0.8, 0.05),
            "b": ConstraintResult(True, 0.90, 0.8, 0.10),
            "c": ConstraintResult(True, 0.81, 0.8, 0.01),
        }
        v = SuiteVerdict(passed=True, results=results)
        closest = v.closest_to_violation
        assert closest is not None
        assert closest[0] == "c"

    def test_closest_to_violation_none_when_all_fail(self):
        """No closest when everything fails (none are satisfied)."""
        results = {
            "a": ConstraintResult(False, 0.7, 0.8, -0.1),
        }
        v = SuiteVerdict(passed=False, results=results)
        assert v.closest_to_violation is None

    def test_summary(self):
        """Summary is a non-empty string."""
        results = {
            "a": ConstraintResult(True, 0.9, 0.8, 0.1),
        }
        v = SuiteVerdict(passed=True, results=results)
        s = v.summary()
        assert "PASSED" in s

    def test_summary_failed(self):
        """Failed summary includes violation details."""
        results = {
            "a": ConstraintResult(False, 0.7, 0.8, -0.1),
        }
        v = SuiteVerdict(passed=False, results=results)
        s = v.summary()
        assert "FAILED" in s
        assert "a" in s


class TestConstraintRunner:
    """Tests for ConstraintRunner."""

    def test_run_all_pass(self, check_context):
        """All constraints pass with a good agent."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)
        agent = MockAgent()

        verdict = runner.run(agent, check_context)
        assert verdict.passed is True
        assert len(verdict.results) == 7

    def test_run_one_fails(self, check_context):
        """Failing one constraint fails the verdict."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)
        agent = MockAgent(accuracy=0.50)

        verdict = runner.run(agent, check_context)
        assert verdict.passed is False
        assert "accuracy_floor" in verdict.violations

    def test_parallel_run(self, check_context):
        """Parallel runner produces same results."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=True, max_workers=2)
        agent = MockAgent()

        verdict = runner.run(agent, check_context)
        assert verdict.passed is True
        assert len(verdict.results) == 7

    def test_default_context(self):
        """Runner creates a default context if none provided."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)
        agent = MockAgent()

        verdict = runner.run(agent)
        assert verdict.passed is True

    def test_run_with_cache(self, check_context):
        """Cached results are used on second run."""
        suite = ConstraintSuite()
        cache = ConstraintCache(ttl_seconds=60)
        runner = ConstraintRunner(suite, parallel=False, cache=cache)
        agent = MockAgent()

        # First run populates cache
        verdict1 = runner.run(agent, check_context)
        assert len(cache) == 7

        # Second run uses cache
        verdict2 = runner.run(agent, check_context)
        assert verdict2.passed == verdict1.passed

    def test_run_all_on_fail(self, check_context):
        """Even if one fails, all constraints are still evaluated."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)
        agent = MockAgent(accuracy=0.50)

        verdict = runner.run(agent, check_context)
        assert len(verdict.results) == 7  # all evaluated


class TestConstraintCache:
    """Tests for ConstraintCache."""

    def test_put_get(self):
        cache = ConstraintCache(ttl_seconds=60)
        result = ConstraintResult(True, 0.9, 0.8, 0.1)
        cache.put("test", result)
        assert cache.get("test") is result

    def test_miss(self):
        cache = ConstraintCache(ttl_seconds=60)
        assert cache.get("missing") is None

    def test_invalidate_key(self):
        cache = ConstraintCache(ttl_seconds=60)
        result = ConstraintResult(True, 0.9, 0.8, 0.1)
        cache.put("test", result)
        cache.invalidate("test")
        assert cache.get("test") is None

    def test_invalidate_all(self):
        cache = ConstraintCache(ttl_seconds=60)
        cache.put("a", ConstraintResult(True, 0.9, 0.8, 0.1))
        cache.put("b", ConstraintResult(True, 0.9, 0.8, 0.1))
        cache.invalidate()
        assert len(cache) == 0

    def test_len(self):
        cache = ConstraintCache(ttl_seconds=60)
        assert len(cache) == 0
        cache.put("a", ConstraintResult(True, 0.9, 0.8, 0.1))
        assert len(cache) == 1

    def test_keys(self):
        cache = ConstraintCache(ttl_seconds=60)
        cache.put("x", ConstraintResult(True, 0.9, 0.8, 0.1))
        assert "x" in cache.keys()

    def test_ttl_expiry(self):
        """Expired entries are not returned."""
        cache = ConstraintCache(ttl_seconds=0)  # instant expiry
        result = ConstraintResult(True, 0.9, 0.8, 0.1)
        cache.put("test", result)
        # With ttl=0, the entry is already expired
        assert cache.get("test") is None
