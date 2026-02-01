"""
Shared test fixtures for all test modules.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.portfolio.portfolio import Portfolio, Asset


@pytest.fixture
def simple_portfolio() -> Portfolio:
    """Create a simple 2-asset test portfolio."""
    assets = [
        Asset(
            ticker="TEST1",
            name="Test Asset 1",
            current_price=100.0,
            mean_return=0.0005,  # ~12.5% annual
            std_return=0.02      # ~31.6% annual vol
        ),
        Asset(
            ticker="TEST2",
            name="Test Asset 2",
            current_price=50.0,
            mean_return=0.0003,  # ~7.5% annual
            std_return=0.015     # ~23.7% annual vol
        )
    ]
    return Portfolio(
        assets=assets,
        weights=np.array([0.6, 0.4]),
        initial_value=100000,
        correlation_matrix=np.array([[1.0, 0.5], [0.5, 1.0]])
    )


@pytest.fixture
def multi_asset_portfolio() -> Portfolio:
    """Create a 4-asset test portfolio for optimization tests."""
    assets = [
        Asset(ticker="A", name="Asset A", current_price=100.0, mean_return=0.0008, std_return=0.025),
        Asset(ticker="B", name="Asset B", current_price=75.0, mean_return=0.0005, std_return=0.018),
        Asset(ticker="C", name="Asset C", current_price=120.0, mean_return=0.0003, std_return=0.012),
        Asset(ticker="D", name="Asset D", current_price=50.0, mean_return=0.0006, std_return=0.022),
    ]
    corr = np.array([
        [1.0, 0.4, 0.2, 0.3],
        [0.4, 1.0, 0.5, 0.25],
        [0.2, 0.5, 1.0, 0.15],
        [0.3, 0.25, 0.15, 1.0]
    ])
    return Portfolio(
        assets=assets,
        weights=np.array([0.25, 0.25, 0.25, 0.25]),
        initial_value=500000,
        correlation_matrix=corr
    )


@pytest.fixture
def sample_returns() -> np.ndarray:
    """Generate sample daily returns for risk metrics tests."""
    np.random.seed(42)
    return np.random.normal(0.0004, 0.015, 252)


@pytest.fixture
def sample_portfolio_values() -> np.ndarray:
    """Generate sample portfolio values with known drawdown."""
    # Values with clear peak at index 10 and trough at index 15
    base = [100000, 102000, 105000, 108000, 110000, 115000,
            120000, 125000, 130000, 135000, 140000,  # Peak at index 10
            135000, 128000, 120000, 110000, 98000,   # Trough at index 15
            105000, 112000, 120000, 130000]
    return np.array(base, dtype=float)


@pytest.fixture
def sample_returns_df() -> pd.DataFrame:
    """Create sample returns DataFrame for optimization tests."""
    np.random.seed(42)
    n_days = 252
    returns = {
        'A': np.random.normal(0.0008, 0.025, n_days),
        'B': np.random.normal(0.0005, 0.018, n_days),
        'C': np.random.normal(0.0003, 0.012, n_days),
    }
    return pd.DataFrame(returns)


@pytest.fixture
def benchmark_returns() -> np.ndarray:
    """Generate sample benchmark returns."""
    np.random.seed(123)
    return np.random.normal(0.0003, 0.012, 252)
