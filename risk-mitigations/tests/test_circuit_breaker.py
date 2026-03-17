"""Tests for CircuitBreaker - sub-query limit, token limit, kill."""

import pytest
from src.cost.circuit_breaker import CircuitBreaker


class TestSubQueryLimit:
    """Tests for sub-query circuit breaking."""

    def test_under_limit_ok(self):
        cb = CircuitBreaker(max_sub_queries=50)
        cb.register_query("q1")
        for _ in range(49):
            assert cb.record_sub_query() is True

    def test_trips_at_50_sub_queries(self):
        cb = CircuitBreaker(max_sub_queries=50)
        cb.register_query("q1")
        for _ in range(49):
            cb.record_sub_query()
        result = cb.record_sub_query()  # 50th
        assert result is False
        assert cb.is_tripped is True

    def test_trips_reason_sub_queries(self):
        cb = CircuitBreaker(max_sub_queries=5)
        cb.register_query("q1")
        for _ in range(5):
            cb.record_sub_query()
        assert cb.state.trip_reason == "max_sub_queries_exceeded"

    def test_remains_tripped_after_trip(self):
        cb = CircuitBreaker(max_sub_queries=3)
        cb.register_query("q1")
        for _ in range(3):
            cb.record_sub_query()
        assert cb.check() is False  # Should remain tripped


class TestTokenLimit:
    """Tests for token-based circuit breaking."""

    def test_under_token_limit_ok(self):
        cb = CircuitBreaker(max_tokens=1_000_000)
        cb.register_query("q1")
        assert cb.record_tokens(500_000) is True

    def test_trips_at_token_limit(self):
        cb = CircuitBreaker(max_tokens=1_000_000)
        cb.register_query("q1")
        cb.record_tokens(500_000)
        result = cb.record_tokens(500_000)  # Exactly at limit
        assert result is False
        assert cb.is_tripped is True

    def test_trips_reason_tokens(self):
        cb = CircuitBreaker(max_tokens=100)
        cb.register_query("q1")
        cb.record_tokens(101)
        assert cb.state.trip_reason == "max_tokens_exceeded"


class TestCostLimit:
    """Tests for cost-based circuit breaking."""

    def test_trips_at_cost_limit(self):
        cb = CircuitBreaker(max_cost_per_query=10.0)
        cb.register_query("q1")
        cb.record_sub_query(cost=5.0)
        result = cb.record_sub_query(cost=5.0)
        assert result is False

    def test_combined_cost_tracking(self):
        cb = CircuitBreaker(max_cost_per_query=10.0)
        cb.register_query("q1")
        cb.record_sub_query(cost=3.0)
        cb.record_tokens(100, cost=3.0)
        assert cb.state.cost == pytest.approx(6.0)


class TestKill:
    """Tests for kill functionality."""

    def test_kill_returns_details(self):
        cb = CircuitBreaker(max_sub_queries=5)
        cb.register_query("q1")
        for _ in range(5):
            cb.record_sub_query(cost=1.0)
        result = cb.kill()
        assert result["status"] == "killed"
        assert result["query_id"] == "q1"

    def test_manual_kill(self):
        cb = CircuitBreaker()
        cb.register_query("q1")
        result = cb.kill()
        assert result["status"] == "killed"
        assert cb.is_tripped is True

    def test_check_with_kill_flag(self):
        cb = CircuitBreaker(max_sub_queries=2)
        cb.register_query("q1")
        cb.record_sub_query()
        cb.record_sub_query()
        result = cb.check(kill=True)
        assert result is False


class TestQueryRegistration:
    """Tests for query registration and reset."""

    def test_register_resets_counters(self):
        cb = CircuitBreaker(max_sub_queries=5)
        cb.register_query("q1")
        for _ in range(5):
            cb.record_sub_query()
        assert cb.is_tripped is True

        cb.register_query("q2")  # Reset
        assert cb.is_tripped is False
        assert cb.state.sub_queries == 0

    def test_trip_history_preserved(self):
        cb = CircuitBreaker(max_sub_queries=2)
        cb.register_query("q1")
        cb.record_sub_query()
        cb.record_sub_query()  # Trip
        cb.register_query("q2")  # Reset
        assert len(cb.get_trip_history()) == 1
        assert cb.get_trip_history()[0].query_id == "q1"
