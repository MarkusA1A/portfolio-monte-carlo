"""
Tests for Market Data Provider with mocked yfinance.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.data.market_data import MarketDataProvider, AssetStatistics


class TestMarketDataProvider:
    """Tests for MarketDataProvider class."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = MarketDataProvider(period="5y")
        assert provider.period == "5y"
        assert provider._cache == {}

    def test_initialization_different_periods(self):
        """Test initialization with different periods."""
        periods = ["1y", "2y", "5y", "10y"]
        for period in periods:
            provider = MarketDataProvider(period=period)
            assert provider.period == period

    @patch('src.data.market_data.yf.download')
    def test_fetch_historical_data_single_ticker(self, mock_download):
        """Test fetching data for single ticker."""
        # Create mock response
        dates = pd.date_range('2020-01-01', periods=100)
        mock_data = pd.DataFrame({
            'Close': np.random.uniform(100, 150, 100)
        }, index=dates)
        mock_download.return_value = mock_data

        provider = MarketDataProvider()
        result = provider.fetch_historical_data(['AAPL'], period="1y")

        mock_download.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    @patch('src.data.market_data.yf.download')
    def test_fetch_historical_data_multiple_tickers(self, mock_download):
        """Test fetching data for multiple tickers."""
        dates = pd.date_range('2020-01-01', periods=100)
        mock_data = pd.DataFrame({
            ('Close', 'AAPL'): np.random.uniform(100, 150, 100),
            ('Close', 'MSFT'): np.random.uniform(200, 300, 100)
        }, index=dates)
        mock_data.columns = pd.MultiIndex.from_tuples([('Close', 'AAPL'), ('Close', 'MSFT')])
        mock_download.return_value = mock_data

        provider = MarketDataProvider()
        result = provider.fetch_historical_data(['AAPL', 'MSFT'], period="1y")

        mock_download.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    @patch('src.data.market_data.yf.download')
    def test_caching(self, mock_download):
        """Test that data is cached."""
        dates = pd.date_range('2020-01-01', periods=100)
        mock_data = pd.DataFrame({
            'Close': np.random.uniform(100, 150, 100)
        }, index=dates)
        mock_download.return_value = mock_data

        provider = MarketDataProvider()

        # First call
        result1 = provider.fetch_historical_data(['AAPL'], period="1y")
        # Second call with same params
        result2 = provider.fetch_historical_data(['AAPL'], period="1y")

        # Should only call download once due to caching
        assert mock_download.call_count == 1

    def test_clear_cache(self):
        """Test cache clearing."""
        provider = MarketDataProvider()
        provider._cache['test_key'] = 'test_value'

        provider.clear_cache()

        assert provider._cache == {}


class TestCalculateReturns:
    """Tests for returns calculation."""

    def test_calculate_returns(self):
        """Test daily returns calculation."""
        provider = MarketDataProvider()

        prices = pd.DataFrame({
            'A': [100, 105, 110, 108, 115],
            'B': [50, 52, 51, 53, 55]
        })

        returns = provider.calculate_returns(prices)

        # Should have one less row (first row dropped)
        assert len(returns) == 4

        # Check first return calculation for A: (105-100)/100 = 0.05
        assert np.isclose(returns['A'].iloc[0], 0.05)

    def test_calculate_returns_handles_missing(self):
        """Test that returns calculation handles missing data."""
        provider = MarketDataProvider()

        prices = pd.DataFrame({
            'A': [100, np.nan, 110, 108, 115],
            'B': [50, 52, np.nan, 53, 55]
        })

        returns = provider.calculate_returns(prices)

        # dropna should remove rows with NaN
        assert not returns.isna().any().any()


class TestCalculateStatistics:
    """Tests for statistics calculation."""

    @patch('src.data.market_data.yf.Ticker')
    @patch('src.data.market_data.yf.download')
    def test_calculate_statistics(self, mock_download, mock_ticker):
        """Test statistics calculation."""
        # Mock price data - needs 'Close' column for single ticker
        dates = pd.date_range('2020-01-01', periods=252)
        prices = pd.DataFrame({
            'Close': 100 * (1 + np.random.normal(0.001, 0.02, 252)).cumprod()
        }, index=dates)
        mock_download.return_value = prices

        # Mock ticker info
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'shortName': 'Apple Inc.'}
        mock_ticker.return_value = mock_ticker_instance

        provider = MarketDataProvider()
        stats = provider.calculate_statistics(['AAPL'], period="1y")

        assert 'AAPL' in stats
        assert isinstance(stats['AAPL'], AssetStatistics)
        assert stats['AAPL'].ticker == 'AAPL'
        assert stats['AAPL'].current_price > 0

    @patch('src.data.market_data.yf.Ticker')
    @patch('src.data.market_data.yf.download')
    def test_annualized_statistics(self, mock_download, mock_ticker):
        """Test that statistics are properly annualized."""
        dates = pd.date_range('2020-01-01', periods=252)
        # Create data with known daily return - needs 'Close' column
        daily_return = 0.001  # 0.1% daily
        prices = pd.DataFrame({
            'Close': 100 * ((1 + daily_return) ** np.arange(252))
        }, index=dates)
        mock_download.return_value = prices

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'shortName': 'Test Stock'}
        mock_ticker.return_value = mock_ticker_instance

        provider = MarketDataProvider()
        stats = provider.calculate_statistics(['TEST'])

        # Annualized return should be approximately (1+0.001)^252 - 1 ≈ 0.284
        assert stats['TEST'].annualized_return > 0


class TestCorrelationMatrix:
    """Tests for correlation matrix calculation."""

    @patch('src.data.market_data.yf.download')
    def test_correlation_matrix_shape(self, mock_download):
        """Test correlation matrix has correct shape."""
        dates = pd.date_range('2020-01-01', periods=100)
        mock_data = pd.DataFrame({
            ('Close', 'A'): np.random.normal(100, 10, 100),
            ('Close', 'B'): np.random.normal(50, 5, 100),
            ('Close', 'C'): np.random.normal(200, 20, 100)
        }, index=dates)
        mock_data.columns = pd.MultiIndex.from_tuples([
            ('Close', 'A'), ('Close', 'B'), ('Close', 'C')
        ])
        mock_download.return_value = mock_data

        provider = MarketDataProvider()
        corr = provider.calculate_correlation_matrix(['A', 'B', 'C'])

        assert corr.shape == (3, 3)

    @patch('src.data.market_data.yf.download')
    def test_correlation_matrix_diagonal_is_one(self, mock_download):
        """Test that diagonal elements are 1."""
        dates = pd.date_range('2020-01-01', periods=100)
        mock_data = pd.DataFrame({
            ('Close', 'A'): np.random.normal(100, 10, 100),
            ('Close', 'B'): np.random.normal(50, 5, 100)
        }, index=dates)
        mock_data.columns = pd.MultiIndex.from_tuples([
            ('Close', 'A'), ('Close', 'B')
        ])
        mock_download.return_value = mock_data

        provider = MarketDataProvider()
        corr = provider.calculate_correlation_matrix(['A', 'B'])

        # Diagonal should be 1
        assert np.allclose(np.diag(corr.values), 1.0)

    @patch('src.data.market_data.yf.download')
    def test_correlation_matrix_symmetric(self, mock_download):
        """Test that correlation matrix is symmetric."""
        dates = pd.date_range('2020-01-01', periods=100)
        mock_data = pd.DataFrame({
            ('Close', 'A'): np.random.normal(100, 10, 100),
            ('Close', 'B'): np.random.normal(50, 5, 100)
        }, index=dates)
        mock_data.columns = pd.MultiIndex.from_tuples([
            ('Close', 'A'), ('Close', 'B')
        ])
        mock_download.return_value = mock_data

        provider = MarketDataProvider()
        corr = provider.calculate_correlation_matrix(['A', 'B'])

        assert np.allclose(corr.values, corr.values.T)


class TestCreatePortfolio:
    """Tests for portfolio creation."""

    @patch('src.data.market_data.yf.Ticker')
    @patch('src.data.market_data.yf.download')
    def test_create_portfolio(self, mock_download, mock_ticker):
        """Test portfolio creation from market data."""
        dates = pd.date_range('2020-01-01', periods=100)
        mock_data = pd.DataFrame({
            ('Close', 'A'): 100 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod(),
            ('Close', 'B'): 50 * (1 + np.random.normal(0.001, 0.015, 100)).cumprod()
        }, index=dates)
        mock_data.columns = pd.MultiIndex.from_tuples([
            ('Close', 'A'), ('Close', 'B')
        ])
        mock_download.return_value = mock_data

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'shortName': 'Test'}
        mock_ticker.return_value = mock_ticker_instance

        provider = MarketDataProvider()
        portfolio = provider.create_portfolio(
            tickers=['A', 'B'],
            weights=[0.6, 0.4],
            initial_value=100000
        )

        assert portfolio.num_assets == 2
        assert np.isclose(sum(portfolio.weights), 1.0)
        assert portfolio.initial_value == 100000

    @patch('src.data.market_data.yf.Ticker')
    @patch('src.data.market_data.yf.download')
    def test_portfolio_has_statistics(self, mock_download, mock_ticker):
        """Test that created portfolio has computed statistics."""
        dates = pd.date_range('2020-01-01', periods=100)
        mock_data = pd.DataFrame({
            ('Close', 'A'): 100 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod(),
            ('Close', 'B'): 50 * (1 + np.random.normal(0.001, 0.015, 100)).cumprod()
        }, index=dates)
        mock_data.columns = pd.MultiIndex.from_tuples([
            ('Close', 'A'), ('Close', 'B')
        ])
        mock_download.return_value = mock_data

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'shortName': 'Test'}
        mock_ticker.return_value = mock_ticker_instance

        provider = MarketDataProvider()
        portfolio = provider.create_portfolio(
            tickers=['A', 'B'],
            weights=[0.5, 0.5],
            initial_value=100000
        )

        # Portfolio should have computed expected return and volatility
        assert portfolio.expected_return() is not None
        assert portfolio.expected_volatility() > 0


class TestAssetStatistics:
    """Tests for AssetStatistics dataclass."""

    def test_asset_statistics_creation(self):
        """Test AssetStatistics dataclass."""
        returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])

        stats = AssetStatistics(
            ticker='TEST',
            name='Test Asset',
            current_price=100.0,
            mean_daily_return=0.001,
            std_daily_return=0.02,
            annualized_return=0.28,
            annualized_volatility=0.32,
            historical_returns=returns
        )

        assert stats.ticker == 'TEST'
        assert stats.current_price == 100.0
        assert len(stats.historical_returns) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
