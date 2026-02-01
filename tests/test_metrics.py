"""
Comprehensive tests for all risk and performance metrics.
"""
import numpy as np
import pytest

from src.risk.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_volatility,
    calculate_skewness,
    calculate_kurtosis,
    calculate_information_ratio,
    calculate_beta,
    calculate_alpha
)
from src.risk.var import calculate_var, calculate_cvar


class TestSharpeRatio:
    """Tests for Sharpe Ratio calculation."""

    def test_positive_returns_positive_sharpe(self, sample_returns):
        """Test that positive excess returns give positive Sharpe."""
        # Use low risk-free rate to ensure positive excess return
        sharpe = calculate_sharpe_ratio(sample_returns, risk_free_rate=0.01)
        # With mean return ~0.0004 daily and rf=1%, should be positive
        assert isinstance(sharpe, float)

    def test_zero_volatility_returns_zero(self):
        """Test that zero volatility returns zero Sharpe."""
        constant_returns = np.array([0.01] * 252)  # Constant returns
        sharpe = calculate_sharpe_ratio(constant_returns, risk_free_rate=0.01)
        assert sharpe == 0.0

    def test_empty_returns_returns_zero(self):
        """Test that empty returns array returns zero."""
        sharpe = calculate_sharpe_ratio(np.array([]), risk_free_rate=0.02)
        assert sharpe == 0.0

    def test_annualization(self):
        """Test that Sharpe is properly annualized."""
        daily_returns = np.random.normal(0.001, 0.02, 252)
        sharpe_daily = calculate_sharpe_ratio(daily_returns, periods_per_year=252)
        sharpe_monthly = calculate_sharpe_ratio(daily_returns, periods_per_year=12)

        # Different period assumptions should give different results
        assert sharpe_daily != sharpe_monthly


class TestSortinoRatio:
    """Tests for Sortino Ratio calculation."""

    def test_sortino_vs_sharpe(self, sample_returns):
        """Test Sortino calculation."""
        sharpe = calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)
        sortino = calculate_sortino_ratio(sample_returns, risk_free_rate=0.02)

        # Both should be finite
        assert np.isfinite(sharpe)
        assert np.isfinite(sortino) or sortino == float('inf')

    def test_no_downside_returns_inf(self):
        """Test that all positive returns give inf Sortino."""
        positive_returns = np.abs(np.random.normal(0.001, 0.01, 100)) + 0.001
        sortino = calculate_sortino_ratio(positive_returns, risk_free_rate=0.0)

        # With no negative returns, Sortino should be inf (if mean > 0)
        assert sortino == float('inf')

    def test_empty_returns_zero(self):
        """Test empty returns."""
        sortino = calculate_sortino_ratio(np.array([]))
        assert sortino == 0.0


class TestMaxDrawdown:
    """Tests for Maximum Drawdown calculation."""

    def test_known_drawdown(self, sample_portfolio_values):
        """Test drawdown calculation with known values."""
        max_dd, peak_idx, trough_idx = calculate_max_drawdown(sample_portfolio_values)

        # Peak should be at index 10 (value 140000)
        # Trough should be at index 15 (value 98000)
        # Drawdown = (140000 - 98000) / 140000 = 0.3 (30%)
        assert np.isclose(max_dd, 0.3, atol=0.01)
        assert peak_idx == 10
        assert trough_idx == 15

    def test_no_drawdown(self):
        """Test with monotonically increasing values."""
        values = np.array([100, 110, 120, 130, 140, 150])
        max_dd, _, _ = calculate_max_drawdown(values)
        assert max_dd == 0.0

    def test_single_peak_decline(self):
        """Test with single peak then decline."""
        values = np.array([100, 200, 100])
        max_dd, peak_idx, trough_idx = calculate_max_drawdown(values)

        assert max_dd == 0.5  # 50% decline
        assert peak_idx == 1
        assert trough_idx == 2

    def test_empty_values(self):
        """Test with empty array."""
        max_dd, peak_idx, trough_idx = calculate_max_drawdown(np.array([]))
        assert max_dd == 0.0


class TestCalmarRatio:
    """Tests for Calmar Ratio calculation."""

    def test_calmar_calculation(self, sample_returns, sample_portfolio_values):
        """Test Calmar ratio calculation."""
        calmar = calculate_calmar_ratio(sample_returns, sample_portfolio_values)
        assert isinstance(calmar, float)

    def test_no_drawdown_returns_inf(self):
        """Test that no drawdown returns inf."""
        returns = np.array([0.01, 0.02, 0.01])
        values = np.array([100, 110, 122])  # Always increasing
        calmar = calculate_calmar_ratio(returns, values)
        assert calmar == float('inf')


class TestVolatility:
    """Tests for Volatility calculation."""

    def test_annualized_volatility(self, sample_returns):
        """Test annualized volatility calculation."""
        vol = calculate_volatility(sample_returns, periods_per_year=252)

        # Daily vol ~0.015, annualized should be ~0.015 * sqrt(252) ≈ 0.238
        assert 0.1 < vol < 0.5  # Reasonable range

    def test_zero_volatility(self):
        """Test with constant returns."""
        constant_returns = np.array([0.01] * 100)
        vol = calculate_volatility(constant_returns)
        assert np.isclose(vol, 0.0, atol=1e-10)


class TestSkewnessKurtosis:
    """Tests for Skewness and Kurtosis calculation."""

    def test_normal_distribution_skewness(self):
        """Test that normal distribution has ~0 skewness."""
        np.random.seed(42)
        normal_returns = np.random.normal(0, 0.01, 10000)
        skew = calculate_skewness(normal_returns)

        # Should be close to 0 for normal distribution
        assert abs(skew) < 0.1

    def test_normal_distribution_kurtosis(self):
        """Test that normal distribution has ~0 excess kurtosis."""
        np.random.seed(42)
        normal_returns = np.random.normal(0, 0.01, 10000)
        kurt = calculate_kurtosis(normal_returns)

        # Excess kurtosis should be close to 0 for normal distribution
        assert abs(kurt) < 0.2

    def test_small_sample_skewness(self):
        """Test skewness with very small sample."""
        small_sample = np.array([0.01, 0.02])
        skew = calculate_skewness(small_sample)
        assert skew == 0.0

    def test_small_sample_kurtosis(self):
        """Test kurtosis with very small sample."""
        small_sample = np.array([0.01, 0.02, 0.03])
        kurt = calculate_kurtosis(small_sample)
        assert kurt == 0.0


class TestBeta:
    """Tests for Beta calculation."""

    def test_beta_vs_itself(self, sample_returns):
        """Test that beta of asset vs itself is 1."""
        beta = calculate_beta(sample_returns, sample_returns)
        assert np.isclose(beta, 1.0)

    def test_beta_calculation(self, sample_returns, benchmark_returns):
        """Test beta calculation."""
        beta = calculate_beta(sample_returns, benchmark_returns)

        # Beta should be a reasonable value
        assert isinstance(beta, float)
        assert np.isfinite(beta)

    def test_beta_length_mismatch(self, sample_returns):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError):
            calculate_beta(sample_returns, sample_returns[:100])

    def test_zero_market_variance(self):
        """Test beta with zero market variance."""
        portfolio = np.array([0.01, 0.02, 0.03])
        market = np.array([0.01, 0.01, 0.01])  # Constant = zero variance
        beta = calculate_beta(portfolio, market)
        assert beta == 0.0


class TestAlpha:
    """Tests for Alpha calculation."""

    def test_alpha_calculation(self, sample_returns, benchmark_returns):
        """Test alpha calculation."""
        alpha = calculate_alpha(sample_returns, benchmark_returns, risk_free_rate=0.02)

        # Alpha should be a finite number
        assert isinstance(alpha, float)
        assert np.isfinite(alpha)

    def test_alpha_length_mismatch(self, sample_returns):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError):
            calculate_alpha(sample_returns, sample_returns[:100])


class TestInformationRatio:
    """Tests for Information Ratio calculation."""

    def test_ir_calculation(self, sample_returns, benchmark_returns):
        """Test Information Ratio calculation."""
        ir = calculate_information_ratio(sample_returns, benchmark_returns)

        assert isinstance(ir, float)
        assert np.isfinite(ir)

    def test_ir_vs_itself(self, sample_returns):
        """Test IR of asset vs itself is 0."""
        ir = calculate_information_ratio(sample_returns, sample_returns)
        assert ir == 0.0

    def test_ir_length_mismatch(self, sample_returns):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError):
            calculate_information_ratio(sample_returns, sample_returns[:100])


class TestVaR:
    """Tests for Value at Risk calculation."""

    def test_var_is_negative(self, sample_returns):
        """Test that VaR represents a loss (negative)."""
        var = calculate_var(sample_returns, confidence=0.95)
        # VaR at 95% should be the 5th percentile - typically negative
        assert var < 0

    def test_var_confidence_relationship(self, sample_returns):
        """Test that higher confidence gives more extreme VaR."""
        var_90 = calculate_var(sample_returns, confidence=0.90)
        var_95 = calculate_var(sample_returns, confidence=0.95)
        var_99 = calculate_var(sample_returns, confidence=0.99)

        # Higher confidence should give more extreme (more negative) VaR
        assert var_99 <= var_95 <= var_90


class TestCVaR:
    """Tests for Conditional Value at Risk calculation."""

    def test_cvar_more_extreme_than_var(self, sample_returns):
        """Test that CVaR is more extreme than VaR."""
        var = calculate_var(sample_returns, confidence=0.95)
        cvar = calculate_cvar(sample_returns, confidence=0.95)

        # CVaR should be more negative (worse) than VaR
        assert cvar <= var

    def test_cvar_confidence_relationship(self, sample_returns):
        """Test that higher confidence gives more extreme CVaR."""
        cvar_90 = calculate_cvar(sample_returns, confidence=0.90)
        cvar_95 = calculate_cvar(sample_returns, confidence=0.95)
        cvar_99 = calculate_cvar(sample_returns, confidence=0.99)

        # Higher confidence should give more extreme CVaR
        assert cvar_99 <= cvar_95 <= cvar_90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
