"""
Tests for tax and transaction cost calculations.
"""
import numpy as np
import pytest

from src.simulation.tax_costs import (
    TaxConfig,
    TransactionCostConfig,
    TaxCostCalculator,
    TaxCostResults,
)


class TestTaxConfig:
    def test_default_kest_rate(self):
        config = TaxConfig()
        assert config.tax_rate == 0.275

    def test_custom_rate(self):
        config = TaxConfig(tax_rate=0.25)
        assert config.tax_rate == 0.25

    def test_invalid_rate_above_1(self):
        with pytest.raises(ValueError, match="zwischen 0 und 1"):
            TaxConfig(tax_rate=1.5)

    def test_invalid_rate_below_0(self):
        with pytest.raises(ValueError, match="zwischen 0 und 1"):
            TaxConfig(tax_rate=-0.1)

    def test_zero_rate(self):
        config = TaxConfig(tax_rate=0.0)
        assert config.tax_rate == 0.0

    def test_full_rate(self):
        config = TaxConfig(tax_rate=1.0)
        assert config.tax_rate == 1.0


class TestTransactionCostConfig:
    def test_default_config(self):
        config = TransactionCostConfig()
        assert config.use_percentage is True
        assert config.percentage_fee == 0.001
        assert config.flat_fee_per_trade == 5.0

    def test_negative_percentage_fee(self):
        with pytest.raises(ValueError, match="nicht negativ"):
            TransactionCostConfig(percentage_fee=-0.01)

    def test_negative_flat_fee(self):
        with pytest.raises(ValueError, match="nicht negativ"):
            TransactionCostConfig(flat_fee_per_trade=-1.0)

    def test_zero_fees(self):
        config = TransactionCostConfig(percentage_fee=0.0, flat_fee_per_trade=0.0)
        assert config.percentage_fee == 0.0
        assert config.flat_fee_per_trade == 0.0


class TestTaxCostResults:
    @pytest.fixture
    def sample_results(self):
        n = 100
        return TaxCostResults(
            total_taxes_paid=np.full(n, 1000.0),
            total_transaction_costs=np.full(n, 50.0),
            rebalancing_events=np.full(n, 4, dtype=np.int32),
            final_value_before_tax=np.full(n, 120000.0),
            final_value_after_tax=np.full(n, 115000.0),
            realized_gains=np.full(n, 5000.0),
            unrealized_gains=np.full(n, 10000.0),
            tax_rate=0.275
        )

    def test_mean_properties(self, sample_results):
        assert sample_results.mean_taxes_paid == 1000.0
        assert sample_results.mean_transaction_costs == 50.0
        assert sample_results.mean_rebalancing_events == 4.0
        assert sample_results.mean_final_before_tax == 120000.0
        assert sample_results.mean_final_after_tax == 115000.0
        assert sample_results.mean_realized_gains == 5000.0
        assert sample_results.mean_unrealized_gains == 10000.0

    def test_total_cost_impact(self, sample_results):
        expected = (120000.0 - 115000.0) / 120000.0
        assert abs(sample_results.total_cost_impact - expected) < 1e-10

    def test_total_cost_impact_zero_before(self):
        results = TaxCostResults(
            total_taxes_paid=np.zeros(10),
            total_transaction_costs=np.zeros(10),
            rebalancing_events=np.zeros(10, dtype=np.int32),
            final_value_before_tax=np.zeros(10),
            final_value_after_tax=np.zeros(10),
            realized_gains=np.zeros(10),
            unrealized_gains=np.zeros(10),
        )
        assert results.total_cost_impact == 0.0

    def test_effective_tax_rate(self, sample_results):
        expected = 1000.0 / 5000.0
        assert abs(sample_results.effective_tax_rate - expected) < 1e-10

    def test_effective_tax_rate_no_gains(self):
        results = TaxCostResults(
            total_taxes_paid=np.zeros(10),
            total_transaction_costs=np.zeros(10),
            rebalancing_events=np.zeros(10, dtype=np.int32),
            final_value_before_tax=np.full(10, 100000.0),
            final_value_after_tax=np.full(10, 100000.0),
            realized_gains=np.zeros(10),
            unrealized_gains=np.zeros(10),
        )
        assert results.effective_tax_rate == 0.0

    def test_median_and_percentile(self, sample_results):
        assert sample_results.median_final_after_tax == 115000.0
        assert sample_results.get_percentile_after_tax(50) == 115000.0


class TestTaxCostCalculator:
    @pytest.fixture
    def calculator(self):
        return TaxCostCalculator(
            tax_config=TaxConfig(tax_rate=0.275),
            cost_config=TransactionCostConfig(use_percentage=True, percentage_fee=0.001),
            num_simulations=10,
            num_assets=2,
            initial_value=100000.0
        )

    def test_initialize_cost_basis(self, calculator):
        weights = np.array([0.6, 0.4])
        calculator.initialize_cost_basis(weights)
        expected = np.tile([60000.0, 40000.0], (10, 1))
        np.testing.assert_array_almost_equal(calculator.cost_basis, expected, decimal=0)

    def test_calculate_transaction_costs_percentage(self, calculator):
        old_w = np.tile([0.65, 0.35], (10, 1))
        new_w = np.tile([0.60, 0.40], (10, 1))
        values = np.full(10, 110000.0)
        costs = calculator.calculate_transaction_costs(old_w, new_w, values)
        # turnover = sum(|0.05, 0.05|) / 2 = 0.05
        # cost = 0.05 * 110000 * 0.001 = 5.5
        np.testing.assert_array_almost_equal(costs, np.full(10, 5.5), decimal=0)

    def test_calculate_transaction_costs_flat_fee(self):
        calc = TaxCostCalculator(
            tax_config=TaxConfig(),
            cost_config=TransactionCostConfig(use_percentage=False, flat_fee_per_trade=10.0),
            num_simulations=5,
            num_assets=2,
            initial_value=100000.0
        )
        old_w = np.tile([0.65, 0.35], (5, 1))
        new_w = np.tile([0.60, 0.40], (5, 1))
        values = np.full(5, 110000.0)
        costs = calc.calculate_transaction_costs(old_w, new_w, values)
        # 2 trades (both weights change > 0.001), so 2 * 10 = 20
        np.testing.assert_array_almost_equal(costs, np.full(5, 20.0), decimal=0)

    def test_process_rebalancing_accumulates(self, calculator):
        weights = np.array([0.6, 0.4])
        calculator.initialize_cost_basis(weights)

        old_w = np.tile([0.65, 0.35], (10, 1))
        new_w = np.tile([0.60, 0.40], (10, 1))
        values = np.full(10, 110000.0)

        total_costs = calculator.process_rebalancing(old_w, new_w, values)
        assert np.all(total_costs >= 0)
        assert np.all(calculator.rebalancing_events == 1)

        # Process again
        calculator.process_rebalancing(old_w, new_w, values)
        assert np.all(calculator.rebalancing_events == 2)

    def test_get_results(self, calculator):
        weights = np.array([0.6, 0.4])
        calculator.initialize_cost_basis(weights)

        before = np.full(10, 120000.0)
        after = np.full(10, 115000.0)
        results = calculator.get_results(before, after)

        assert isinstance(results, TaxCostResults)
        assert results.mean_final_before_tax == 120000.0
        assert results.mean_final_after_tax == 115000.0
