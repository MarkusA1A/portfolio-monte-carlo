"""
Tests for export/report generation.
"""
import numpy as np
import pytest

from src.portfolio.portfolio import Portfolio, Asset
from src.simulation.monte_carlo import SimulationResults
from src.simulation.tax_costs import TaxCostResults
from src.export.reports import (
    create_excel_report,
    create_csv_report,
    format_currency,
    format_percentage,
)


@pytest.fixture
def mock_portfolio():
    assets = [
        Asset(ticker="AAA", name="Asset A", current_price=100.0,
              mean_return=0.0005, std_return=0.02),
        Asset(ticker="BBB", name="Asset B", current_price=50.0,
              mean_return=0.0003, std_return=0.015),
    ]
    return Portfolio(
        assets=assets,
        weights=np.array([0.6, 0.4]),
        initial_value=100000,
        correlation_matrix=np.array([[1.0, 0.3], [0.3, 1.0]])
    )


@pytest.fixture
def mock_results():
    n = 100
    final_vals = np.random.default_rng(42).normal(110000, 15000, n)
    returns = (final_vals - 100000) / 100000
    portfolio_values = np.column_stack([
        np.full(n, 100000),
        final_vals
    ])
    return SimulationResults(
        portfolio_values=portfolio_values,
        final_values=final_vals,
        returns=returns,
        time_horizon=252,
        num_simulations=n
    )


@pytest.fixture
def mock_tax_results():
    n = 100
    return TaxCostResults(
        total_taxes_paid=np.full(n, 500.0),
        total_transaction_costs=np.full(n, 25.0),
        rebalancing_events=np.full(n, 4, dtype=np.int32),
        final_value_before_tax=np.full(n, 115000.0),
        final_value_after_tax=np.full(n, 112000.0),
        realized_gains=np.full(n, 2000.0),
        unrealized_gains=np.full(n, 8000.0),
        tax_rate=0.275
    )


class TestExcelReport:
    def test_creates_bytes(self, mock_portfolio, mock_results):
        result = create_excel_report(
            portfolio=mock_portfolio,
            results=mock_results,
            initial_value=100000,
            var_value=-5000,
            cvar_value=-8000,
            confidence_level=0.95,
            metrics={'sharpe': 0.5, 'sortino': 0.6, 'max_drawdown': -0.15, 'volatility': 0.2}
        )
        assert isinstance(result, bytes)
        assert len(result) > 0
        # Excel files start with PK (zip format)
        assert result[:2] == b'PK'

    def test_with_tax_results(self, mock_portfolio, mock_results, mock_tax_results):
        result = create_excel_report(
            portfolio=mock_portfolio,
            results=mock_results,
            initial_value=100000,
            var_value=-5000,
            cvar_value=-8000,
            confidence_level=0.95,
            metrics={'sharpe': 0.5, 'sortino': 0.6, 'max_drawdown': -0.15, 'volatility': 0.2},
            tax_cost_results=mock_tax_results
        )
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestCsvReport:
    def test_creates_string(self, mock_portfolio, mock_results):
        result = create_csv_report(
            portfolio=mock_portfolio,
            results=mock_results,
            initial_value=100000
        )
        assert isinstance(result, str)
        assert "Monte Carlo" in result
        assert "AAA" in result
        assert "BBB" in result
        assert "Perzentile" in result

    def test_with_tax_results(self, mock_portfolio, mock_results, mock_tax_results):
        result = create_csv_report(
            portfolio=mock_portfolio,
            results=mock_results,
            initial_value=100000,
            tax_cost_results=mock_tax_results
        )
        assert "Steuern & Kosten" in result
        assert "KESt" in result

    def test_contains_percentiles(self, mock_portfolio, mock_results):
        result = create_csv_report(
            portfolio=mock_portfolio,
            results=mock_results,
            initial_value=100000
        )
        assert "5%" in result
        assert "50%" in result
        assert "95%" in result


class TestFormatHelpers:
    def test_format_currency_default(self):
        assert format_currency(1234.56) == "€1,234.56"

    def test_format_currency_custom(self):
        assert format_currency(1000.0, "$") == "$1,000.00"

    def test_format_percentage(self):
        assert format_percentage(0.1234) == "12.34%"

    def test_format_percentage_negative(self):
        assert format_percentage(-0.05) == "-5.00%"
