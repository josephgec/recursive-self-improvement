"""Tests for CostForecaster - predict exhaustion, alert thresholds."""

import pytest
from src.cost.cost_forecaster import CostForecaster, CostForecast


class TestForecastBasic:
    """Basic forecasting tests."""

    def test_empty_history_returns_zero(self):
        forecaster = CostForecaster()
        forecast = forecaster.forecast([], 10, 100.0)
        assert forecast.burn_rate_per_iteration == 0.0
        assert forecast.budget_sufficient is True

    def test_burn_rate_calculation(self, cost_history):
        forecaster = CostForecaster()
        forecast = forecaster.forecast(cost_history, 50, 1000.0)
        expected_rate = sum(cost_history) / len(cost_history)
        assert forecast.burn_rate_per_iteration == pytest.approx(expected_rate)

    def test_remaining_cost_estimate(self, cost_history):
        forecaster = CostForecaster()
        remaining_iters = 20
        forecast = forecaster.forecast(cost_history, remaining_iters, 1000.0)
        expected_rate = sum(cost_history) / len(cost_history)
        assert forecast.estimated_remaining_cost == pytest.approx(expected_rate * remaining_iters)


class TestExhaustionPrediction:
    """Tests for budget exhaustion prediction."""

    def test_predicts_exhaustion_correctly(self):
        forecaster = CostForecaster()
        history = [10.0] * 10  # $10 per iteration
        forecast = forecaster.forecast(
            history,
            remaining_iterations=100,
            total_budget=200.0,
            spent_so_far=100.0,
        )
        # Budget remaining = 100, rate = 10/iter, so 10 iterations until exhaustion
        assert forecast.iterations_until_exhaustion == 10

    def test_budget_insufficient(self):
        forecaster = CostForecaster()
        history = [10.0] * 10
        forecast = forecaster.forecast(
            history,
            remaining_iterations=50,
            total_budget=200.0,
            spent_so_far=100.0,
        )
        # Need 50 * 10 = 500, but only 100 remaining
        assert forecast.budget_sufficient is False

    def test_budget_sufficient(self):
        forecaster = CostForecaster()
        history = [1.0] * 10
        forecast = forecaster.forecast(
            history,
            remaining_iterations=10,
            total_budget=1000.0,
            spent_so_far=10.0,
        )
        assert forecast.budget_sufficient is True

    def test_no_exhaustion_when_rate_zero(self):
        forecaster = CostForecaster()
        history = [0.0] * 5
        forecast = forecaster.forecast(history, 100, 1000.0)
        assert forecast.iterations_until_exhaustion is None


class TestAlertThresholds:
    """Tests for alert thresholds at 20% remaining."""

    def test_alert_at_20_percent_remaining(self):
        forecaster = CostForecaster(alert_threshold=0.20)
        history = [10.0] * 10
        forecast = forecaster.forecast(
            history,
            remaining_iterations=100,
            total_budget=100.0,
            spent_so_far=85.0,  # 15% remaining
        )
        assert forecaster.should_alert(forecast) is True

    def test_no_alert_when_sufficient(self):
        forecaster = CostForecaster(alert_threshold=0.20)
        history = [1.0] * 10
        forecast = forecaster.forecast(
            history,
            remaining_iterations=10,
            total_budget=1000.0,
            spent_so_far=100.0,  # 90% remaining
        )
        assert forecaster.should_alert(forecast) is False

    def test_alert_when_budget_insufficient(self):
        forecaster = CostForecaster(alert_threshold=0.20)
        history = [50.0] * 10
        forecast = forecaster.forecast(
            history,
            remaining_iterations=100,
            total_budget=200.0,
            spent_so_far=150.0,
        )
        assert forecaster.should_alert(forecast) is True

    def test_critical_alert_level(self):
        forecaster = CostForecaster(alert_threshold=0.20)
        history = [10.0] * 10
        forecast = forecaster.forecast(
            history,
            remaining_iterations=100,
            total_budget=100.0,
            spent_so_far=96.0,  # 4% remaining
        )
        assert forecast.alert_level == "critical"

    def test_warning_alert_level(self):
        forecaster = CostForecaster(alert_threshold=0.20)
        history = [1.0] * 10
        forecast = forecaster.forecast(
            history,
            remaining_iterations=10,
            total_budget=100.0,
            spent_so_far=85.0,  # 15% remaining
        )
        assert forecast.alert_level == "warning"

    def test_no_alert_level(self):
        forecaster = CostForecaster(alert_threshold=0.20)
        history = [1.0] * 10
        forecast = forecaster.forecast(
            history,
            remaining_iterations=10,
            total_budget=1000.0,
            spent_so_far=10.0,
        )
        assert forecast.alert_level == "none"


class TestForecastHistory:
    """Tests for forecast history tracking."""

    def test_history_recorded(self, cost_history):
        forecaster = CostForecaster()
        forecaster.forecast(cost_history, 10, 1000.0)
        forecaster.forecast(cost_history, 20, 1000.0)
        assert len(forecaster.get_history()) == 2

    def test_fraction_remaining_property(self):
        forecaster = CostForecaster()
        history = [10.0] * 10
        forecast = forecaster.forecast(history, 10, 200.0, spent_so_far=100.0)
        assert forecast.fraction_remaining > 0

    def test_confidence_increases_with_data(self):
        forecaster = CostForecaster()
        f1 = forecaster.forecast([1.0], 10, 100.0)
        f2 = forecaster.forecast([1.0] * 20, 10, 100.0)
        assert f2.confidence > f1.confidence
