"""
Market Data Provider using Yahoo Finance.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf

from src.portfolio.portfolio import Asset, Portfolio


@dataclass
class AssetStatistics:
    """Statistics calculated from historical data."""

    ticker: str
    name: str
    current_price: float
    mean_daily_return: float
    std_daily_return: float
    annualized_return: float
    annualized_volatility: float
    historical_returns: pd.Series


class MarketDataProvider:
    """
    Fetches and processes market data from Yahoo Finance.
    """

    def __init__(self, period: str = "5y"):
        """
        Initialize the market data provider.

        Args:
            period: Historical data period (e.g., "1y", "2y", "5y", "10y")
        """
        self.period = period
        self._cache: dict[str, pd.DataFrame] = {}

    def fetch_historical_data(
        self,
        tickers: list[str],
        period: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical adjusted close prices for given tickers.

        Args:
            tickers: List of stock tickers
            period: Optional override for data period

        Returns:
            DataFrame with adjusted close prices
        """
        period = period or self.period

        # Check cache
        cache_key = f"{','.join(sorted(tickers))}_{period}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Fetch data
        data = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False
        )

        # Handle single vs multiple tickers
        if len(tickers) == 1:
            prices = data[['Close']].rename(columns={'Close': tickers[0]})
        else:
            prices = data['Close']

        self._cache[cache_key] = prices
        return prices

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from prices."""
        return prices.pct_change().dropna()

    def calculate_statistics(
        self,
        tickers: list[str],
        period: Optional[str] = None
    ) -> dict[str, AssetStatistics]:
        """
        Calculate statistics for multiple assets.

        Args:
            tickers: List of stock tickers
            period: Optional data period

        Returns:
            Dict mapping ticker to AssetStatistics
        """
        prices = self.fetch_historical_data(tickers, period)
        returns = self.calculate_returns(prices)

        stats = {}
        for ticker in tickers:
            ticker_returns = returns[ticker].dropna()
            ticker_info = yf.Ticker(ticker).info

            mean_daily = ticker_returns.mean()
            std_daily = ticker_returns.std()

            stats[ticker] = AssetStatistics(
                ticker=ticker,
                name=ticker_info.get('shortName', ticker),
                current_price=prices[ticker].iloc[-1],
                mean_daily_return=mean_daily,
                std_daily_return=std_daily,
                annualized_return=(1 + mean_daily) ** 252 - 1,
                annualized_volatility=std_daily * np.sqrt(252),
                historical_returns=ticker_returns
            )

        return stats

    def calculate_correlation_matrix(
        self,
        tickers: list[str],
        period: Optional[str] = None
    ) -> pd.DataFrame:
        """Calculate correlation matrix for given tickers."""
        prices = self.fetch_historical_data(tickers, period)
        returns = self.calculate_returns(prices)
        return returns.corr()

    def calculate_covariance_matrix(
        self,
        tickers: list[str],
        period: Optional[str] = None
    ) -> pd.DataFrame:
        """Calculate covariance matrix for given tickers."""
        prices = self.fetch_historical_data(tickers, period)
        returns = self.calculate_returns(prices)
        return returns.cov()

    def create_portfolio(
        self,
        tickers: list[str],
        weights: list[float],
        initial_value: float,
        period: Optional[str] = None
    ) -> Portfolio:
        """
        Create a Portfolio object from tickers and weights.

        Args:
            tickers: List of stock tickers
            weights: List of portfolio weights
            initial_value: Initial portfolio value
            period: Historical data period

        Returns:
            Configured Portfolio object
        """
        stats = self.calculate_statistics(tickers, period)
        corr_matrix = self.calculate_correlation_matrix(tickers, period)

        assets = [
            Asset(
                ticker=s.ticker,
                name=s.name,
                current_price=s.current_price,
                mean_return=s.mean_daily_return,
                std_return=s.std_daily_return,
                historical_returns=s.historical_returns
            )
            for s in stats.values()
        ]

        return Portfolio(
            assets=assets,
            weights=np.array(weights),
            initial_value=initial_value,
            correlation_matrix=corr_matrix.values
        )

    def get_benchmark_data(
        self,
        benchmark: str = "SPY",
        period: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch benchmark data for comparison."""
        return self.fetch_historical_data([benchmark], period)

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
