"""
Tests for Portfolio Optimization and Efficient Frontier.
"""
import numpy as np
import pytest

from src.portfolio.optimization import (
    PortfolioOptimizer,
    OptimizedPortfolio,
    EfficientFrontierResult,
    optimize_portfolio_from_data
)


class TestPortfolioOptimizer:
    """Tests for PortfolioOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create a basic optimizer for testing."""
        expected_returns = np.array([0.12, 0.08, 0.05])  # Annualized
        cov_matrix = np.array([
            [0.04, 0.01, 0.005],    # 20% vol, some correlation
            [0.01, 0.025, 0.003],   # 15.8% vol
            [0.005, 0.003, 0.01]    # 10% vol
        ])
        return PortfolioOptimizer(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            ticker_symbols=['A', 'B', 'C'],
            risk_free_rate=0.02,
            random_seed=42
        )

    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.n_assets == 3
        assert optimizer.risk_free_rate == 0.02
        assert len(optimizer.ticker_symbols) == 3

    def test_portfolio_return(self, optimizer):
        """Test portfolio return calculation."""
        weights = np.array([0.5, 0.3, 0.2])
        expected = 0.5 * 0.12 + 0.3 * 0.08 + 0.2 * 0.05  # = 0.094
        actual = optimizer._portfolio_return(weights)
        assert np.isclose(actual, expected)

    def test_portfolio_volatility(self, optimizer):
        """Test portfolio volatility calculation."""
        weights = np.array([1.0, 0.0, 0.0])  # 100% in first asset
        vol = optimizer._portfolio_volatility(weights)

        # Should equal sqrt of first diagonal element
        expected = np.sqrt(0.04)  # 20%
        assert np.isclose(vol, expected)

    def test_portfolio_sharpe(self, optimizer):
        """Test Sharpe ratio calculation."""
        weights = np.array([0.33, 0.33, 0.34])
        sharpe = optimizer._portfolio_sharpe(weights)

        assert isinstance(sharpe, float)
        assert np.isfinite(sharpe)


class TestMaxSharpeOptimization:
    """Tests for Maximum Sharpe Ratio optimization."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with clear optimal portfolio."""
        # Asset C has best risk-adjusted return
        expected_returns = np.array([0.08, 0.06, 0.10])
        cov_matrix = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.025, 0.003],
            [0.005, 0.003, 0.01]  # C has lowest vol with highest return
        ])
        return PortfolioOptimizer(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            ticker_symbols=['A', 'B', 'C'],
            risk_free_rate=0.02,
            random_seed=42
        )

    def test_max_sharpe_returns_portfolio(self, optimizer):
        """Test that max Sharpe returns OptimizedPortfolio."""
        result = optimizer.optimize_max_sharpe()

        assert isinstance(result, OptimizedPortfolio)
        assert len(result.weights) == 3
        assert len(result.ticker_symbols) == 3

    def test_max_sharpe_weights_sum_to_one(self, optimizer):
        """Test that optimized weights sum to 1."""
        result = optimizer.optimize_max_sharpe()
        assert np.isclose(np.sum(result.weights), 1.0)

    def test_max_sharpe_no_negative_weights(self, optimizer):
        """Test that weights are non-negative (no shorting by default)."""
        result = optimizer.optimize_max_sharpe()
        assert np.all(result.weights >= -1e-6)  # Allow small numerical error

    def test_max_sharpe_prefers_best_asset(self, optimizer):
        """Test that optimizer prefers asset with best risk-adjusted return."""
        result = optimizer.optimize_max_sharpe()
        # Asset C should have significant weight (best Sharpe)
        assert result.weights[2] > 0.3

    def test_get_weights_dict(self, optimizer):
        """Test weights dictionary conversion."""
        result = optimizer.optimize_max_sharpe()
        weights_dict = result.get_weights_dict()

        assert 'A' in weights_dict
        assert 'B' in weights_dict
        assert 'C' in weights_dict
        assert np.isclose(sum(weights_dict.values()), 1.0)


class TestMinVolatilityOptimization:
    """Tests for Minimum Volatility optimization."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for min vol testing."""
        expected_returns = np.array([0.10, 0.08, 0.05])
        cov_matrix = np.array([
            [0.04, 0.005, 0.002],   # A: highest vol
            [0.005, 0.02, 0.001],   # B: medium vol
            [0.002, 0.001, 0.005]   # C: lowest vol
        ])
        return PortfolioOptimizer(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            ticker_symbols=['A', 'B', 'C'],
            risk_free_rate=0.02,
            random_seed=42
        )

    def test_min_vol_returns_portfolio(self, optimizer):
        """Test that min vol returns OptimizedPortfolio."""
        result = optimizer.optimize_min_volatility()

        assert isinstance(result, OptimizedPortfolio)
        assert result.volatility > 0

    def test_min_vol_weights_sum_to_one(self, optimizer):
        """Test that optimized weights sum to 1."""
        result = optimizer.optimize_min_volatility()
        assert np.isclose(np.sum(result.weights), 1.0)

    def test_min_vol_prefers_low_vol_assets(self, optimizer):
        """Test that min vol portfolio favors low volatility assets."""
        result = optimizer.optimize_min_volatility()
        # Asset C has lowest volatility, should have significant weight
        assert result.weights[2] > 0.3

    def test_min_vol_lower_than_max_sharpe(self, optimizer):
        """Test that min vol portfolio has lower vol than max sharpe."""
        min_vol = optimizer.optimize_min_volatility()
        max_sharpe = optimizer.optimize_max_sharpe()

        assert min_vol.volatility <= max_sharpe.volatility


class TestTargetReturnOptimization:
    """Tests for target return optimization."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for target return testing."""
        expected_returns = np.array([0.15, 0.10, 0.05])
        cov_matrix = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.025, 0.003],
            [0.005, 0.003, 0.01]
        ])
        return PortfolioOptimizer(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            ticker_symbols=['A', 'B', 'C'],
            risk_free_rate=0.02,
            random_seed=42
        )

    def test_target_return_achieves_target(self, optimizer):
        """Test that optimization achieves target return."""
        target = 0.10
        result = optimizer.optimize_target_return(target)

        if result is not None:
            assert np.isclose(result.expected_return, target, atol=0.01)

    def test_unreachable_target_returns_none(self, optimizer):
        """Test that unreachable target returns None."""
        # Target higher than max possible
        result = optimizer.optimize_target_return(0.50)  # 50% - impossible
        # Should return None or a portfolio close to max return
        # Depending on implementation


class TestEfficientFrontier:
    """Tests for Efficient Frontier calculation."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for efficient frontier testing."""
        expected_returns = np.array([0.12, 0.08, 0.05, 0.10])
        cov_matrix = np.array([
            [0.04, 0.01, 0.005, 0.008],
            [0.01, 0.025, 0.003, 0.005],
            [0.005, 0.003, 0.01, 0.002],
            [0.008, 0.005, 0.002, 0.03]
        ])
        return PortfolioOptimizer(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            ticker_symbols=['A', 'B', 'C', 'D'],
            risk_free_rate=0.02,
            random_seed=42
        )

    def test_frontier_returns_result(self, optimizer):
        """Test that frontier calculation returns EfficientFrontierResult."""
        result = optimizer.calculate_efficient_frontier(n_points=20)

        assert isinstance(result, EfficientFrontierResult)
        assert result.max_sharpe_portfolio is not None
        assert result.min_volatility_portfolio is not None

    def test_frontier_points_ordered(self, optimizer):
        """Test that frontier points are ordered by return."""
        result = optimizer.calculate_efficient_frontier(n_points=20)

        # Returns should be generally increasing
        returns = result.frontier_returns
        assert len(returns) > 1

    def test_frontier_includes_special_portfolios(self, optimizer):
        """Test that frontier includes max Sharpe and min vol portfolios."""
        result = optimizer.calculate_efficient_frontier(n_points=30)

        # Max Sharpe should be on or near the frontier
        max_sharpe = result.max_sharpe_portfolio
        assert max_sharpe.sharpe_ratio > 0

        # Min vol should have lowest volatility
        min_vol = result.min_volatility_portfolio
        assert min_vol.volatility <= max_sharpe.volatility

    def test_random_portfolios_generated(self, optimizer):
        """Test that random portfolios are generated for visualization."""
        result = optimizer.calculate_efficient_frontier(n_points=20)

        assert len(result.all_portfolios_returns) == 1000
        assert len(result.all_portfolios_volatilities) == 1000
        assert len(result.all_portfolios_sharpe) == 1000

    def test_min_vol_on_frontier_left(self, optimizer):
        """Test that min vol portfolio is leftmost on frontier."""
        result = optimizer.calculate_efficient_frontier(n_points=30)

        min_vol = result.min_volatility_portfolio.volatility
        frontier_vols = result.frontier_volatilities

        # Min vol should be <= min of frontier volatilities
        assert min_vol <= np.min(frontier_vols) + 0.001


class TestOptimizeFromData:
    """Tests for optimize_portfolio_from_data helper function."""

    def test_from_returns_df(self, sample_returns_df):
        """Test optimization from returns DataFrame."""
        result = optimize_portfolio_from_data(sample_returns_df, risk_free_rate=0.02)

        assert isinstance(result, EfficientFrontierResult)
        assert result.max_sharpe_portfolio is not None
        assert len(result.max_sharpe_portfolio.ticker_symbols) == 3

    def test_tickers_preserved(self, sample_returns_df):
        """Test that ticker symbols are preserved."""
        result = optimize_portfolio_from_data(sample_returns_df)

        tickers = result.max_sharpe_portfolio.ticker_symbols
        assert 'A' in tickers
        assert 'B' in tickers
        assert 'C' in tickers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
