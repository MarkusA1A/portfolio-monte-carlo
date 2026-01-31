"""
Portfolio and Asset classes for investment management.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class Asset:
    """Represents a single asset in the portfolio."""

    ticker: str
    name: str
    current_price: float
    mean_return: float  # Daily mean return
    std_return: float   # Daily standard deviation
    historical_returns: Optional[pd.Series] = None

    def annualized_return(self) -> float:
        """Calculate annualized return from daily return."""
        return (1 + self.mean_return) ** 252 - 1

    def annualized_volatility(self) -> float:
        """Calculate annualized volatility from daily std."""
        return self.std_return * np.sqrt(252)


@dataclass
class Portfolio:
    """
    Portfolio containing multiple assets with weights.

    Attributes:
        assets: List of Asset objects
        weights: Array of portfolio weights (must sum to 1)
        initial_value: Initial portfolio value in currency
    """

    assets: list[Asset]
    weights: np.ndarray
    initial_value: float
    correlation_matrix: Optional[np.ndarray] = None
    _covariance_matrix: Optional[np.ndarray] = None

    def __post_init__(self):
        self.weights = np.array(self.weights)
        if not np.isclose(np.sum(self.weights), 1.0):
            raise ValueError(f"Weights must sum to 1, got {np.sum(self.weights)}")
        if len(self.assets) != len(self.weights):
            raise ValueError("Number of assets must match number of weights")

    @property
    def tickers(self) -> list[str]:
        return [a.ticker for a in self.assets]

    @property
    def num_assets(self) -> int:
        return len(self.assets)

    def get_current_prices(self) -> np.ndarray:
        """Get current prices for all assets."""
        return np.array([a.current_price for a in self.assets])

    def get_mean_returns(self) -> np.ndarray:
        """Get mean daily returns for all assets."""
        return np.array([a.mean_return for a in self.assets])

    def get_std_returns(self) -> np.ndarray:
        """Get standard deviation of daily returns for all assets."""
        return np.array([a.std_return for a in self.assets])

    def get_covariance_matrix(self) -> np.ndarray:
        """Calculate or return cached covariance matrix."""
        if self._covariance_matrix is not None:
            return self._covariance_matrix

        std_returns = self.get_std_returns()

        if self.correlation_matrix is not None:
            # Build covariance from correlation
            D = np.diag(std_returns)
            self._covariance_matrix = D @ self.correlation_matrix @ D
        else:
            # Use historical returns if available
            returns_data = self._get_historical_returns_matrix()
            if returns_data is not None:
                self._covariance_matrix = np.cov(returns_data.T)
            else:
                # Assume no correlation (diagonal covariance)
                self._covariance_matrix = np.diag(std_returns ** 2)

        return self._covariance_matrix

    def _get_historical_returns_matrix(self) -> Optional[np.ndarray]:
        """Get matrix of historical returns if available."""
        if all(a.historical_returns is not None for a in self.assets):
            returns_df = pd.DataFrame({
                a.ticker: a.historical_returns for a in self.assets
            })
            return returns_df.dropna().values
        return None

    def get_correlation_matrix(self) -> np.ndarray:
        """Calculate correlation matrix from covariance matrix."""
        cov = self.get_covariance_matrix()
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        return corr

    def expected_return(self) -> float:
        """Calculate expected portfolio return (daily)."""
        return float(np.dot(self.weights, self.get_mean_returns()))

    def expected_volatility(self) -> float:
        """Calculate expected portfolio volatility (daily)."""
        cov = self.get_covariance_matrix()
        variance = self.weights @ cov @ self.weights
        return float(np.sqrt(variance))

    def annualized_expected_return(self) -> float:
        """Calculate annualized expected return."""
        daily = self.expected_return()
        return (1 + daily) ** 252 - 1

    def annualized_expected_volatility(self) -> float:
        """Calculate annualized expected volatility."""
        return self.expected_volatility() * np.sqrt(252)

    def copy(self) -> "Portfolio":
        """Create a copy of the portfolio."""
        return Portfolio(
            assets=[Asset(
                ticker=a.ticker,
                name=a.name,
                current_price=a.current_price,
                mean_return=a.mean_return,
                std_return=a.std_return,
                historical_returns=a.historical_returns.copy() if a.historical_returns is not None else None
            ) for a in self.assets],
            weights=self.weights.copy(),
            initial_value=self.initial_value,
            correlation_matrix=self.correlation_matrix.copy() if self.correlation_matrix is not None else None
        )

    def adjust_statistics(self, return_adjustment: float, volatility_multiplier: float):
        """Adjust portfolio statistics for scenario analysis."""
        for asset in self.assets:
            asset.mean_return += return_adjustment / 252  # Convert annual to daily
            asset.std_return *= volatility_multiplier
        self._covariance_matrix = None  # Invalidate cache

    def to_dataframe(self) -> pd.DataFrame:
        """Convert portfolio to a DataFrame for display."""
        return pd.DataFrame({
            'Ticker': [a.ticker for a in self.assets],
            'Name': [a.name for a in self.assets],
            'Weight': self.weights,
            'Price': [a.current_price for a in self.assets],
            'Ann. Return': [a.annualized_return() for a in self.assets],
            'Ann. Volatility': [a.annualized_volatility() for a in self.assets],
        })
