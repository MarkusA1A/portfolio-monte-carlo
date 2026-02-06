"""
Tests for chart/visualization functions.
"""
import numpy as np
import pytest
import plotly.graph_objects as go

from src.simulation.monte_carlo import SimulationResults
from src.visualization.charts import (
    plot_simulation_paths,
    plot_distribution,
    plot_var_cone,
    plot_correlation_heatmap,
    plot_portfolio_weights,
    plot_drawdown,
    plot_success_rate_gauge,
    plot_optimal_weights,
)


@pytest.fixture
def mock_results():
    rng = np.random.default_rng(42)
    n_sims = 50
    time_horizon = 20

    portfolio_values = np.zeros((n_sims, time_horizon + 1), dtype=np.float32)
    portfolio_values[:, 0] = 100000
    for t in range(time_horizon):
        daily_ret = rng.normal(0.0005, 0.015, n_sims)
        portfolio_values[:, t + 1] = portfolio_values[:, t] * (1 + daily_ret)

    final_values = portfolio_values[:, -1]
    returns = (final_values - 100000) / 100000

    return SimulationResults(
        portfolio_values=portfolio_values,
        final_values=final_values,
        returns=returns,
        time_horizon=time_horizon,
        num_simulations=n_sims
    )


@pytest.fixture
def empty_results():
    return SimulationResults(
        portfolio_values=np.empty((0, 1)),
        final_values=np.array([]),
        returns=np.array([]),
        time_horizon=0,
        num_simulations=0
    )


class TestPlotSimulationPaths:
    def test_returns_figure(self, mock_results):
        fig = plot_simulation_paths(mock_results, 100000)
        assert isinstance(fig, go.Figure)

    def test_empty_results(self, empty_results):
        fig = plot_simulation_paths(empty_results, 100000)
        assert isinstance(fig, go.Figure)


class TestPlotDistribution:
    def test_returns_figure(self, mock_results):
        fig = plot_distribution(mock_results.final_values, 100000)
        assert isinstance(fig, go.Figure)

    def test_with_var_cvar(self, mock_results):
        fig = plot_distribution(mock_results.final_values, 100000, var_level=95000, cvar_level=90000)
        assert isinstance(fig, go.Figure)

    def test_empty_results(self):
        fig = plot_distribution(np.array([]), 100000)
        assert isinstance(fig, go.Figure)


class TestPlotVarCone:
    def test_returns_figure(self, mock_results):
        fig = plot_var_cone(mock_results)
        assert isinstance(fig, go.Figure)


class TestPlotCorrelationHeatmap:
    def test_returns_figure(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        tickers = ["A", "B"]
        fig = plot_correlation_heatmap(corr, tickers)
        assert isinstance(fig, go.Figure)

    def test_dimension_mismatch(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        tickers = ["A", "B", "C"]  # 3 tickers but 2x2 matrix
        fig = plot_correlation_heatmap(corr, tickers)
        assert isinstance(fig, go.Figure)

    def test_single_asset(self):
        corr = np.array([[1.0]])
        tickers = ["A"]
        fig = plot_correlation_heatmap(corr, tickers)
        assert isinstance(fig, go.Figure)


class TestPlotPortfolioWeights:
    def test_returns_figure(self):
        weights = np.array([0.4, 0.3, 0.3])
        tickers = ["A", "B", "C"]
        fig = plot_portfolio_weights(weights, tickers)
        assert isinstance(fig, go.Figure)


class TestPlotDrawdown:
    def test_returns_figure(self, mock_results):
        median_path = np.median(mock_results.portfolio_values, axis=0)
        fig = plot_drawdown(median_path)
        assert isinstance(fig, go.Figure)


class TestPlotSuccessRateGauge:
    def test_high_success(self):
        fig = plot_success_rate_gauge(0.95)
        assert isinstance(fig, go.Figure)

    def test_low_success(self):
        fig = plot_success_rate_gauge(0.30)
        assert isinstance(fig, go.Figure)

    def test_zero_success(self):
        fig = plot_success_rate_gauge(0.0)
        assert isinstance(fig, go.Figure)


class TestPlotOptimalWeights:
    def test_returns_figure(self):
        weights = np.array([0.4, 0.35, 0.25])
        tickers = ["SPY", "AGG", "GLD"]
        fig = plot_optimal_weights(weights, tickers)
        assert isinstance(fig, go.Figure)
