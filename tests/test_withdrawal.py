"""
Tests for Withdrawal Simulation (Entnahme-Simulation).
"""
import numpy as np
import pytest

from src.simulation.withdrawal import (
    WithdrawalSimulator,
    WithdrawalResults,
    calculate_required_capital
)


class TestWithdrawalSimulator:
    """Tests for WithdrawalSimulator class."""

    def test_initialization(self):
        """Test simulator initialization."""
        simulator = WithdrawalSimulator(n_simulations=1000, random_seed=42)
        assert simulator.n_simulations == 1000
        assert simulator.random_seed == 42

    def test_basic_simulation(self):
        """Test basic withdrawal simulation."""
        simulator = WithdrawalSimulator(n_simulations=100, random_seed=42)
        results = simulator.simulate(
            initial_value=500000,
            monthly_withdrawal=2000,
            expected_annual_return=0.06,
            annual_volatility=0.15,
            years=30,
            inflation_rate=0.02,
            adjust_for_inflation=False
        )

        assert isinstance(results, WithdrawalResults)
        assert results.n_simulations == 100
        assert results.initial_value == 500000
        assert results.monthly_withdrawal == 2000
        assert results.total_periods == 360  # 30 years * 12 months

    def test_portfolio_paths_shape(self):
        """Test that portfolio paths have correct shape."""
        simulator = WithdrawalSimulator(n_simulations=50, random_seed=42)
        results = simulator.simulate(
            initial_value=100000,
            monthly_withdrawal=500,
            expected_annual_return=0.05,
            annual_volatility=0.10,
            years=10
        )

        assert results.portfolio_paths.shape == (50, 121)  # 10 years * 12 + 1
        assert results.withdrawal_paths.shape == (50, 120)

    def test_success_rate_bounds(self):
        """Test that success rate is between 0 and 1."""
        simulator = WithdrawalSimulator(n_simulations=100, random_seed=42)
        results = simulator.simulate(
            initial_value=500000,
            monthly_withdrawal=2000,
            expected_annual_return=0.06,
            annual_volatility=0.15,
            years=30
        )

        assert 0 <= results.success_rate <= 1
        assert results.failure_rate == 1 - results.success_rate

    def test_high_withdrawal_low_success(self):
        """Test that very high withdrawal rates lead to low success rates."""
        simulator = WithdrawalSimulator(n_simulations=500, random_seed=42)

        # Very aggressive withdrawal (10% of capital per month)
        results = simulator.simulate(
            initial_value=100000,
            monthly_withdrawal=10000,  # 10% per month = 120% per year
            expected_annual_return=0.06,
            annual_volatility=0.15,
            years=20
        )

        # With such high withdrawal, almost all simulations should fail
        assert results.success_rate < 0.1

    def test_low_withdrawal_high_success(self):
        """Test that very low withdrawal rates lead to high success rates."""
        simulator = WithdrawalSimulator(n_simulations=500, random_seed=42)

        # Very conservative withdrawal (1% of capital per year)
        results = simulator.simulate(
            initial_value=1000000,
            monthly_withdrawal=833,  # ~1% per year
            expected_annual_return=0.06,
            annual_volatility=0.15,
            years=30
        )

        # With such low withdrawal, most simulations should succeed
        assert results.success_rate > 0.9

    def test_inflation_adjustment_increases_withdrawals(self):
        """Test that inflation adjustment increases total withdrawals."""
        simulator = WithdrawalSimulator(n_simulations=100, random_seed=42)

        # Without inflation adjustment
        results_no_inflation = simulator.simulate(
            initial_value=500000,
            monthly_withdrawal=2000,
            expected_annual_return=0.06,
            annual_volatility=0.15,
            years=20,
            inflation_rate=0.03,
            adjust_for_inflation=False
        )

        # With inflation adjustment
        results_with_inflation = simulator.simulate(
            initial_value=500000,
            monthly_withdrawal=2000,
            expected_annual_return=0.06,
            annual_volatility=0.15,
            years=20,
            inflation_rate=0.03,
            adjust_for_inflation=True
        )

        # Inflation-adjusted withdrawals should be higher
        assert results_with_inflation.total_withdrawn_mean > results_no_inflation.total_withdrawn_mean

    def test_depletion_times(self):
        """Test depletion time tracking."""
        simulator = WithdrawalSimulator(n_simulations=200, random_seed=42)

        # Moderate withdrawal that should cause some failures
        results = simulator.simulate(
            initial_value=300000,
            monthly_withdrawal=3000,  # ~12% per year
            expected_annual_return=0.05,
            annual_volatility=0.20,
            years=30
        )

        # Check depletion times array
        assert len(results.depletion_times) == 200

        # Infinite for successful simulations
        successful_count = np.sum(results.depletion_times == np.inf)
        assert np.isclose(successful_count / 200, results.success_rate, atol=0.01)

    def test_percentile_ordering(self):
        """Test that percentiles are correctly ordered."""
        simulator = WithdrawalSimulator(n_simulations=500, random_seed=42)
        results = simulator.simulate(
            initial_value=500000,
            monthly_withdrawal=1500,
            expected_annual_return=0.06,
            annual_volatility=0.15,
            years=25
        )

        assert results.percentile_5 <= results.percentile_25
        assert results.percentile_25 <= results.percentile_50
        assert results.percentile_50 <= results.percentile_75
        assert results.percentile_75 <= results.percentile_95

    def test_withdrawal_rate_property(self):
        """Test withdrawal rate calculation."""
        simulator = WithdrawalSimulator(n_simulations=50, random_seed=42)
        results = simulator.simulate(
            initial_value=500000,
            monthly_withdrawal=2000,
            expected_annual_return=0.06,
            annual_volatility=0.15,
            years=20
        )

        expected_rate = (2000 * 12) / 500000  # 4.8%
        assert np.isclose(results.withdrawal_rate, expected_rate)


class TestFindSafeWithdrawalRate:
    """Tests for safe withdrawal rate finder."""

    def test_swr_basic(self):
        """Test basic SWR calculation."""
        simulator = WithdrawalSimulator(n_simulations=500, random_seed=42)
        result = simulator.find_safe_withdrawal_rate(
            initial_value=1000000,
            expected_annual_return=0.06,
            annual_volatility=0.15,
            years=30,
            target_success_rate=0.95
        )

        assert 'monthly_withdrawal' in result
        assert 'withdrawal_rate_pct' in result
        assert result['success_rate'] >= 0.94  # Should be close to target
        assert result['withdrawal_rate_pct'] > 0

    def test_swr_higher_target_lower_rate(self):
        """Test that higher success target leads to lower withdrawal rate."""
        simulator = WithdrawalSimulator(n_simulations=500, random_seed=42)

        result_90 = simulator.find_safe_withdrawal_rate(
            initial_value=1000000,
            expected_annual_return=0.06,
            annual_volatility=0.15,
            years=30,
            target_success_rate=0.90
        )

        result_99 = simulator.find_safe_withdrawal_rate(
            initial_value=1000000,
            expected_annual_return=0.06,
            annual_volatility=0.15,
            years=30,
            target_success_rate=0.99
        )

        # Higher success target should mean lower withdrawal rate
        assert result_99['withdrawal_rate_pct'] < result_90['withdrawal_rate_pct']


class TestCalculateRequiredCapital:
    """Tests for required capital calculation."""

    def test_required_capital_basic(self):
        """Test basic required capital calculation."""
        result = calculate_required_capital(
            monthly_withdrawal=3000,
            expected_annual_return=0.06,
            annual_volatility=0.15,
            years=30,
            target_success_rate=0.95,
            n_simulations=500
        )

        assert 'required_capital' in result
        assert 'implied_withdrawal_rate_pct' in result
        assert result['required_capital'] > 0
        assert result['success_rate'] >= 0.94

    def test_higher_withdrawal_needs_more_capital(self):
        """Test that higher withdrawal needs more capital."""
        result_low = calculate_required_capital(
            monthly_withdrawal=2000,
            expected_annual_return=0.06,
            annual_volatility=0.15,
            years=30,
            target_success_rate=0.95,
            n_simulations=300
        )

        result_high = calculate_required_capital(
            monthly_withdrawal=4000,
            expected_annual_return=0.06,
            annual_volatility=0.15,
            years=30,
            target_success_rate=0.95,
            n_simulations=300
        )

        assert result_high['required_capital'] > result_low['required_capital']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
