"""
Tests for Monte Carlo Simulation Engine
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.simulation.monte_carlo import MonteCarloSimulator, SimulationResults
from src.portfolio.portfolio import Portfolio, Asset
from src.risk.var import calculate_var, calculate_cvar
from src.risk.metrics import calculate_sharpe_ratio, calculate_max_drawdown


def create_test_portfolio() -> Portfolio:
    """Create a simple test portfolio."""
    assets = [
        Asset(
            ticker="TEST1",
            name="Test Asset 1",
            current_price=100.0,
            mean_return=0.0005,  # ~12.5% annual
            std_return=0.02     # ~31.6% annual vol
        ),
        Asset(
            ticker="TEST2",
            name="Test Asset 2",
            current_price=50.0,
            mean_return=0.0003,  # ~7.5% annual
            std_return=0.015    # ~23.7% annual vol
        )
    ]
    return Portfolio(
        assets=assets,
        weights=np.array([0.6, 0.4]),
        initial_value=100000,
        correlation_matrix=np.array([[1.0, 0.5], [0.5, 1.0]])
    )


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""

    def test_initialization(self):
        simulator = MonteCarloSimulator(num_simulations=1000, time_horizon=252)
        assert simulator.num_simulations == 1000
        assert simulator.time_horizon == 252

    def test_run_simulation(self):
        portfolio = create_test_portfolio()
        simulator = MonteCarloSimulator(
            num_simulations=100,
            time_horizon=50,
            random_seed=42
        )
        results = simulator.run_simulation(portfolio)

        assert isinstance(results, SimulationResults)
        assert results.num_simulations == 100
        assert results.time_horizon == 50
        assert results.portfolio_values.shape == (100, 51)  # +1 for initial
        assert results.final_values.shape == (100,)

    def test_results_properties(self):
        portfolio = create_test_portfolio()
        simulator = MonteCarloSimulator(
            num_simulations=500,
            time_horizon=252,
            random_seed=42
        )
        results = simulator.run_simulation(portfolio)

        assert results.mean_final_value > 0
        assert results.median_final_value > 0
        assert results.std_final_value > 0
        assert results.min_value < results.max_value

    def test_percentile(self):
        portfolio = create_test_portfolio()
        simulator = MonteCarloSimulator(
            num_simulations=1000,
            time_horizon=252,
            random_seed=42
        )
        results = simulator.run_simulation(portfolio)

        p25 = results.percentile(25)
        p50 = results.percentile(50)
        p75 = results.percentile(75)

        assert p25 < p50 < p75


class TestPortfolio:
    """Tests for Portfolio class."""

    def test_weights_must_sum_to_one(self):
        assets = [
            Asset("A", "Asset A", 100, 0.001, 0.02),
            Asset("B", "Asset B", 50, 0.001, 0.02)
        ]
        with pytest.raises(ValueError):
            Portfolio(assets=assets, weights=np.array([0.5, 0.3]), initial_value=100000)

    def test_expected_return(self):
        portfolio = create_test_portfolio()
        ret = portfolio.expected_return()
        # Should be weighted average: 0.6 * 0.0005 + 0.4 * 0.0003 = 0.00042
        assert np.isclose(ret, 0.00042)

    def test_covariance_matrix(self):
        portfolio = create_test_portfolio()
        cov = portfolio.get_covariance_matrix()
        assert cov.shape == (2, 2)
        # Should be symmetric
        assert np.allclose(cov, cov.T)


class TestRiskMetrics:
    """Tests for risk metrics calculations."""

    def test_var_calculation(self):
        returns = np.random.normal(0.001, 0.02, 1000)
        var = calculate_var(returns, confidence=0.95)
        # VaR should be negative (representing loss)
        assert var < 0

    def test_cvar_less_than_var(self):
        returns = np.random.normal(0.001, 0.02, 1000)
        var = calculate_var(returns, confidence=0.95)
        cvar = calculate_cvar(returns, confidence=0.95)
        # CVaR should be more negative than VaR (worse)
        assert cvar <= var

    def test_sharpe_ratio(self):
        # Positive returns should give positive Sharpe
        returns = np.random.normal(0.001, 0.01, 252)
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        assert isinstance(sharpe, float)

    def test_max_drawdown(self):
        # Create values with a known drawdown
        values = np.array([100, 110, 120, 100, 90, 95, 100])
        max_dd, peak_idx, trough_idx = calculate_max_drawdown(values)
        # Max drawdown should be 30/120 = 0.25
        assert np.isclose(max_dd, 0.25)
        assert peak_idx == 2
        assert trough_idx == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
