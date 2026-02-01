"""
Tests for Savings Plan (Sparplan) Simulation.
"""
import numpy as np
import pytest

from src.simulation.savings_plan import SavingsPlanSimulator, SavingsPlanResults


class TestSavingsPlanSimulator:
    """Tests for SavingsPlanSimulator class."""

    def test_initialization(self):
        """Test simulator initialization."""
        simulator = SavingsPlanSimulator(
            num_simulations=1000,
            time_horizon_years=10,
            random_seed=42
        )
        assert simulator.num_simulations == 1000
        assert simulator.time_horizon_years == 10
        assert simulator.time_horizon_months == 120

    def test_basic_simulation(self, simple_portfolio):
        """Test basic savings plan simulation."""
        simulator = SavingsPlanSimulator(
            num_simulations=100,
            time_horizon_years=5,
            random_seed=42
        )
        results = simulator.run_simulation(
            portfolio=simple_portfolio,
            monthly_contribution=500,
            initial_investment=10000
        )

        assert isinstance(results, SavingsPlanResults)
        assert results.num_simulations == 100
        assert results.time_horizon == 60  # 5 years * 12 months
        assert results.monthly_contribution == 500

    def test_portfolio_values_shape(self, simple_portfolio):
        """Test that portfolio values have correct shape."""
        simulator = SavingsPlanSimulator(
            num_simulations=50,
            time_horizon_years=3,
            random_seed=42
        )
        results = simulator.run_simulation(
            portfolio=simple_portfolio,
            monthly_contribution=300,
            initial_investment=5000
        )

        assert results.portfolio_values.shape == (50, 37)  # 3 years * 12 + 1
        assert results.final_values.shape == (50,)

    def test_total_invested_calculation(self, simple_portfolio):
        """Test total invested calculation."""
        simulator = SavingsPlanSimulator(
            num_simulations=50,
            time_horizon_years=5,
            random_seed=42
        )
        results = simulator.run_simulation(
            portfolio=simple_portfolio,
            monthly_contribution=500,
            initial_investment=10000
        )

        expected_total = 10000 + (500 * 60)  # Initial + 60 months
        assert results.total_invested == expected_total

    def test_contributions_tracking(self, simple_portfolio):
        """Test that contributions are tracked correctly."""
        simulator = SavingsPlanSimulator(
            num_simulations=50,
            time_horizon_years=2,
            random_seed=42
        )
        results = simulator.run_simulation(
            portfolio=simple_portfolio,
            monthly_contribution=1000,
            initial_investment=5000
        )

        # First contribution should be initial investment
        assert results.total_contributions[0] == 5000

        # Each subsequent month adds 1000
        for i in range(1, len(results.total_contributions)):
            assert results.total_contributions[i] == 5000 + i * 1000

    def test_mean_final_value_positive(self, simple_portfolio):
        """Test that mean final value is positive."""
        simulator = SavingsPlanSimulator(
            num_simulations=100,
            time_horizon_years=10,
            random_seed=42
        )
        results = simulator.run_simulation(
            portfolio=simple_portfolio,
            monthly_contribution=500,
            initial_investment=10000
        )

        assert results.mean_final_value > 0
        assert results.median_final_value > 0

    def test_profit_calculation(self, simple_portfolio):
        """Test profit calculation."""
        simulator = SavingsPlanSimulator(
            num_simulations=200,
            time_horizon_years=10,
            random_seed=42
        )
        results = simulator.run_simulation(
            portfolio=simple_portfolio,
            monthly_contribution=500,
            initial_investment=10000
        )

        # Profit should be final value minus total invested
        expected_profit = results.mean_final_value - results.total_invested
        assert np.isclose(results.mean_profit, expected_profit)

    def test_return_calculation(self, simple_portfolio):
        """Test return calculation."""
        simulator = SavingsPlanSimulator(
            num_simulations=200,
            time_horizon_years=10,
            random_seed=42
        )
        results = simulator.run_simulation(
            portfolio=simple_portfolio,
            monthly_contribution=500,
            initial_investment=10000
        )

        # Return should be profit / total invested
        expected_return = results.mean_profit / results.total_invested
        assert np.isclose(results.mean_return, expected_return)

    def test_percentile_ordering(self, simple_portfolio):
        """Test that percentiles are correctly ordered."""
        simulator = SavingsPlanSimulator(
            num_simulations=500,
            time_horizon_years=10,
            random_seed=42
        )
        results = simulator.run_simulation(
            portfolio=simple_portfolio,
            monthly_contribution=500,
            initial_investment=10000
        )

        p25 = results.percentile(25)
        p50 = results.percentile(50)
        p75 = results.percentile(75)

        assert p25 <= p50 <= p75

    def test_zero_initial_investment(self, simple_portfolio):
        """Test simulation with zero initial investment."""
        simulator = SavingsPlanSimulator(
            num_simulations=100,
            time_horizon_years=5,
            random_seed=42
        )
        results = simulator.run_simulation(
            portfolio=simple_portfolio,
            monthly_contribution=500,
            initial_investment=0
        )

        assert results.total_invested == 500 * 60
        # With 0 initial investment, the first month's contribution (500) is added at start
        # The implementation adds monthly contribution at the beginning of each period
        assert results.total_contributions[0] == 0  # Initial tracking is 0

    def test_higher_contribution_higher_final_value(self, simple_portfolio):
        """Test that higher contribution leads to higher final value."""
        simulator = SavingsPlanSimulator(
            num_simulations=200,
            time_horizon_years=10,
            random_seed=42
        )

        results_low = simulator.run_simulation(
            portfolio=simple_portfolio,
            monthly_contribution=200,
            initial_investment=0
        )

        results_high = simulator.run_simulation(
            portfolio=simple_portfolio,
            monthly_contribution=1000,
            initial_investment=0
        )

        # Higher contribution should lead to higher final value on average
        assert results_high.mean_final_value > results_low.mean_final_value

    def test_reproducibility_with_seed(self, simple_portfolio):
        """Test that results are reproducible with same seed."""
        simulator1 = SavingsPlanSimulator(
            num_simulations=100,
            time_horizon_years=5,
            random_seed=42
        )
        results1 = simulator1.run_simulation(
            portfolio=simple_portfolio,
            monthly_contribution=500,
            initial_investment=10000
        )

        simulator2 = SavingsPlanSimulator(
            num_simulations=100,
            time_horizon_years=5,
            random_seed=42
        )
        results2 = simulator2.run_simulation(
            portfolio=simple_portfolio,
            monthly_contribution=500,
            initial_investment=10000
        )

        assert np.allclose(results1.final_values, results2.final_values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
