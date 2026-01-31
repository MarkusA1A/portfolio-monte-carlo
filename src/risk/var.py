"""
Value-at-Risk (VaR) and Conditional VaR (CVaR) calculations.
"""
import numpy as np
from scipy import stats


def calculate_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    initial_value: float = 1.0
) -> float:
    """
    Calculate Historical Value-at-Risk.

    VaR represents the maximum expected loss at a given confidence level.

    Args:
        returns: Array of portfolio returns
        confidence: Confidence level (e.g., 0.95 for 95%)
        initial_value: Initial portfolio value for absolute VaR

    Returns:
        VaR as absolute value (negative = loss)
    """
    percentile = (1 - confidence) * 100
    var_return = np.percentile(returns, percentile)
    return var_return * initial_value


def calculate_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
    initial_value: float = 1.0
) -> float:
    """
    Calculate Conditional Value-at-Risk (Expected Shortfall).

    CVaR is the expected loss given that the loss exceeds VaR.

    Args:
        returns: Array of portfolio returns
        confidence: Confidence level
        initial_value: Initial portfolio value

    Returns:
        CVaR as absolute value (negative = loss)
    """
    var_return = calculate_var(returns, confidence, initial_value=1.0)
    # CVaR is the mean of returns below VaR
    tail_returns = returns[returns <= var_return]
    if len(tail_returns) == 0:
        return var_return * initial_value
    return float(np.mean(tail_returns)) * initial_value


def calculate_parametric_var(
    mean: float,
    std: float,
    confidence: float = 0.95,
    initial_value: float = 1.0,
    time_horizon: int = 1
) -> float:
    """
    Calculate Parametric (Gaussian) VaR.

    Assumes returns are normally distributed.

    Args:
        mean: Expected return (for time horizon)
        std: Standard deviation of returns
        confidence: Confidence level
        initial_value: Initial portfolio value
        time_horizon: Time horizon in days

    Returns:
        Parametric VaR
    """
    # Scale parameters for time horizon
    scaled_mean = mean * time_horizon
    scaled_std = std * np.sqrt(time_horizon)

    # Get z-score for confidence level
    z_score = stats.norm.ppf(1 - confidence)

    var_return = scaled_mean + z_score * scaled_std
    return var_return * initial_value


def calculate_parametric_cvar(
    mean: float,
    std: float,
    confidence: float = 0.95,
    initial_value: float = 1.0,
    time_horizon: int = 1
) -> float:
    """
    Calculate Parametric CVaR for normal distribution.

    Args:
        mean: Expected return
        std: Standard deviation
        confidence: Confidence level
        initial_value: Initial portfolio value
        time_horizon: Time horizon in days

    Returns:
        Parametric CVaR
    """
    scaled_mean = mean * time_horizon
    scaled_std = std * np.sqrt(time_horizon)

    alpha = 1 - confidence
    z_score = stats.norm.ppf(alpha)
    pdf_z = stats.norm.pdf(z_score)

    cvar_return = scaled_mean - scaled_std * pdf_z / alpha
    return cvar_return * initial_value


def calculate_marginal_var(
    portfolio_returns: np.ndarray,
    asset_returns: np.ndarray,
    confidence: float = 0.95
) -> float:
    """
    Calculate Marginal VaR for an asset.

    Marginal VaR measures how VaR changes with small changes in position.

    Args:
        portfolio_returns: Portfolio return series
        asset_returns: Individual asset return series
        confidence: Confidence level

    Returns:
        Marginal VaR contribution
    """
    portfolio_var = calculate_var(portfolio_returns, confidence)
    cov = np.cov(portfolio_returns, asset_returns)[0, 1]
    portfolio_std = np.std(portfolio_returns)

    beta = cov / (portfolio_std ** 2)
    return beta * portfolio_var


def calculate_component_var(
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
    confidence: float = 0.95,
    initial_value: float = 1.0
) -> np.ndarray:
    """
    Calculate Component VaR for each asset.

    Component VaR shows each asset's contribution to total portfolio VaR.

    Args:
        weights: Portfolio weights
        covariance_matrix: Asset covariance matrix
        confidence: Confidence level
        initial_value: Portfolio value

    Returns:
        Array of component VaR values
    """
    portfolio_variance = weights @ covariance_matrix @ weights
    portfolio_std = np.sqrt(portfolio_variance)

    z_score = stats.norm.ppf(1 - confidence)

    marginal_var = z_score * (covariance_matrix @ weights) / portfolio_std
    component_var = weights * marginal_var * initial_value

    return component_var
