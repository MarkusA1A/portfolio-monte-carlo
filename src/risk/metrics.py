"""
Additional risk and performance metrics.
"""
import numpy as np
from typing import Optional


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe Ratio.

    Sharpe Ratio = (Expected Return - Risk Free Rate) / Standard Deviation

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) == 0:
        return 0.0

    # Convert to same frequency
    period_rf = risk_free_rate / periods_per_year

    excess_returns = returns - period_rf
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(returns, ddof=1)

    if std_returns == 0:
        return 0.0

    # Annualize
    sharpe = mean_excess / std_returns * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
    target_return: Optional[float] = None
) -> float:
    """
    Calculate the Sortino Ratio.

    Similar to Sharpe but uses downside deviation instead of total volatility.

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        target_return: Target return for downside calculation (defaults to rf)

    Returns:
        Annualized Sortino Ratio
    """
    if len(returns) == 0:
        return 0.0

    period_rf = risk_free_rate / periods_per_year
    target = target_return if target_return is not None else period_rf

    excess_returns = returns - period_rf
    mean_excess = np.mean(excess_returns)

    # Calculate downside deviation
    downside_returns = returns[returns < target] - target
    if len(downside_returns) == 0:
        return float('inf') if mean_excess > 0 else 0.0

    downside_std = np.sqrt(np.mean(downside_returns ** 2))

    if downside_std == 0:
        return float('inf') if mean_excess > 0 else 0.0

    sortino = mean_excess / downside_std * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_max_drawdown(values: np.ndarray) -> tuple[float, int, int]:
    """
    Calculate Maximum Drawdown.

    Maximum Drawdown measures the largest peak-to-trough decline.

    Args:
        values: Array of portfolio values over time

    Returns:
        Tuple of (max_drawdown_pct, peak_idx, trough_idx)
    """
    if len(values) == 0:
        return 0.0, 0, 0

    peak = values[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_peak_idx = 0
    max_dd_trough_idx = 0

    for i, value in enumerate(values):
        if value > peak:
            peak = value
            peak_idx = i

        drawdown = (peak - value) / peak if peak != 0 else 0.0
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_peak_idx = peak_idx
            max_dd_trough_idx = i

    return float(max_dd), max_dd_peak_idx, max_dd_trough_idx


def calculate_calmar_ratio(
    returns: np.ndarray,
    values: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar Ratio.

    Calmar Ratio = Annualized Return / Maximum Drawdown

    Args:
        returns: Array of period returns
        values: Array of portfolio values
        periods_per_year: Periods per year

    Returns:
        Calmar Ratio
    """
    max_dd, _, _ = calculate_max_drawdown(values)
    if max_dd == 0:
        return float('inf')

    annualized_return = np.mean(returns) * periods_per_year
    return float(annualized_return / max_dd)


def calculate_volatility(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Array of period returns
        periods_per_year: Periods per year

    Returns:
        Annualized volatility
    """
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def calculate_skewness(returns: np.ndarray) -> float:
    """
    Calculate return distribution skewness.

    Positive skew = more extreme positive returns
    Negative skew = more extreme negative returns

    Args:
        returns: Array of returns

    Returns:
        Skewness value
    """
    if len(returns) < 3:
        return 0.0

    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0

    n = len(returns)
    skew = (n / ((n - 1) * (n - 2))) * np.sum(((returns - mean) / std) ** 3)
    return float(skew)


def calculate_kurtosis(returns: np.ndarray) -> float:
    """
    Calculate excess kurtosis of returns.

    Kurtosis > 0: fat tails (more extreme events than normal)
    Kurtosis < 0: thin tails

    Args:
        returns: Array of returns

    Returns:
        Excess kurtosis value
    """
    if len(returns) < 4:
        return 0.0

    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0

    n = len(returns)
    m4 = np.mean((returns - mean) ** 4)
    excess_kurt = (m4 / (std ** 4)) - 3
    return float(excess_kurt)


def calculate_information_ratio(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information Ratio.

    IR = Active Return / Tracking Error

    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series
        periods_per_year: Periods per year

    Returns:
        Annualized Information Ratio
    """
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Return series must have same length")

    active_returns = portfolio_returns - benchmark_returns
    tracking_error = np.std(active_returns, ddof=1)

    if tracking_error == 0:
        return 0.0

    ir = np.mean(active_returns) / tracking_error * np.sqrt(periods_per_year)
    return float(ir)


def calculate_beta(
    portfolio_returns: np.ndarray,
    market_returns: np.ndarray
) -> float:
    """
    Calculate portfolio beta relative to market.

    Beta measures systematic risk relative to the market.

    Args:
        portfolio_returns: Portfolio return series
        market_returns: Market return series

    Returns:
        Beta coefficient
    """
    if len(portfolio_returns) != len(market_returns):
        raise ValueError("Return series must have same length")

    covariance = np.cov(portfolio_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns, ddof=1)

    if market_variance == 0:
        return 0.0

    return float(covariance / market_variance)


def calculate_alpha(
    portfolio_returns: np.ndarray,
    market_returns: np.ndarray,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Jensen's Alpha.

    Alpha = Portfolio Return - [Rf + Beta * (Market Return - Rf)]

    Args:
        portfolio_returns: Portfolio return series
        market_returns: Market return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year

    Returns:
        Annualized Alpha
    """
    beta = calculate_beta(portfolio_returns, market_returns)
    period_rf = risk_free_rate / periods_per_year

    portfolio_mean = np.mean(portfolio_returns)
    market_mean = np.mean(market_returns)

    alpha = portfolio_mean - (period_rf + beta * (market_mean - period_rf))
    return float(alpha * periods_per_year)
