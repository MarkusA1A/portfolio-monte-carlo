"""
Monte Carlo Simulation Engine for Portfolio Analysis
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from src.portfolio.portfolio import Portfolio


@dataclass
class SimulationResults:
    """Container for Monte Carlo simulation results."""

    price_paths: np.ndarray  # Shape: (num_simulations, time_horizon, num_assets)
    portfolio_values: np.ndarray  # Shape: (num_simulations, time_horizon)
    final_values: np.ndarray  # Shape: (num_simulations,)
    returns: np.ndarray  # Shape: (num_simulations,)
    time_horizon: int
    num_simulations: int

    @property
    def mean_final_value(self) -> float:
        return float(np.mean(self.final_values))

    @property
    def median_final_value(self) -> float:
        return float(np.median(self.final_values))

    @property
    def std_final_value(self) -> float:
        return float(np.std(self.final_values))

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns))

    @property
    def min_value(self) -> float:
        return float(np.min(self.final_values))

    @property
    def max_value(self) -> float:
        return float(np.max(self.final_values))

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.final_values, p))


class MonteCarloSimulator:
    """
    Monte Carlo Simulator for portfolio analysis using Geometric Brownian Motion.

    Uses correlated random returns to simulate multi-asset portfolio evolution.
    """

    def __init__(
        self,
        num_simulations: int = 10000,
        time_horizon: int = 252,
        random_seed: int | None = None
    ):
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon  # Trading days
        self.rng = np.random.default_rng(random_seed)

    def run_simulation(
        self,
        portfolio: "Portfolio",
        rebalancing_strategy=None
    ) -> SimulationResults:
        """
        Run Monte Carlo simulation for the given portfolio.

        Args:
            portfolio: Portfolio object with assets and weights
            rebalancing_strategy: Optional rebalancing strategy

        Returns:
            SimulationResults with all simulation data
        """
        num_assets = len(portfolio.assets)

        # Get portfolio statistics
        mean_returns = portfolio.get_mean_returns()  # Daily returns
        cov_matrix = portfolio.get_covariance_matrix()

        # Generate correlated random returns using Cholesky decomposition
        cholesky = np.linalg.cholesky(cov_matrix)

        # Generate uncorrelated standard normal random variables
        uncorrelated_randoms = self.rng.standard_normal(
            (self.num_simulations, self.time_horizon, num_assets)
        )

        # Transform to correlated returns
        correlated_randoms = np.einsum('ijk,lk->ijl', uncorrelated_randoms, cholesky)

        # Calculate daily returns: mean + volatility * random
        daily_returns = mean_returns + correlated_randoms

        # Initialize price paths
        initial_prices = portfolio.get_current_prices()
        price_paths = np.zeros((self.num_simulations, self.time_horizon + 1, num_assets))
        price_paths[:, 0, :] = initial_prices

        # Simulate price evolution using geometric returns
        for t in range(self.time_horizon):
            price_paths[:, t + 1, :] = price_paths[:, t, :] * (1 + daily_returns[:, t, :])

        # Calculate portfolio values
        weights = portfolio.weights
        initial_value = portfolio.initial_value

        if rebalancing_strategy is None:
            # Buy and hold: fixed number of shares
            shares = (initial_value * weights) / initial_prices
            portfolio_values = np.sum(price_paths * shares, axis=2)
        else:
            # With rebalancing
            portfolio_values = self._simulate_with_rebalancing(
                price_paths, weights, initial_value, rebalancing_strategy
            )

        final_values = portfolio_values[:, -1]
        total_returns = (final_values - initial_value) / initial_value

        return SimulationResults(
            price_paths=price_paths,
            portfolio_values=portfolio_values,
            final_values=final_values,
            returns=total_returns,
            time_horizon=self.time_horizon,
            num_simulations=self.num_simulations
        )

    def _simulate_with_rebalancing(
        self,
        price_paths: np.ndarray,
        target_weights: np.ndarray,
        initial_value: float,
        strategy
    ) -> np.ndarray:
        """Simulate portfolio with rebalancing strategy."""
        num_sims, num_days, num_assets = price_paths.shape
        portfolio_values = np.zeros((num_sims, num_days))

        # Initialize
        current_weights = np.tile(target_weights, (num_sims, 1))
        portfolio_values[:, 0] = initial_value

        for t in range(1, num_days):
            # Calculate returns for this period
            period_returns = price_paths[:, t, :] / price_paths[:, t - 1, :] - 1

            # Update portfolio value
            weighted_returns = np.sum(current_weights * period_returns, axis=1)
            portfolio_values[:, t] = portfolio_values[:, t - 1] * (1 + weighted_returns)

            # Update weights based on price changes
            for i in range(num_assets):
                current_weights[:, i] = (
                    current_weights[:, i] * (1 + period_returns[:, i]) / (1 + weighted_returns)
                )

            # Check if rebalancing is needed
            if strategy.should_rebalance(t, current_weights, target_weights):
                current_weights = np.tile(target_weights, (num_sims, 1))

        return portfolio_values

    def run_scenario_analysis(
        self,
        portfolio: "Portfolio",
        scenarios: dict[str, tuple[float, float]]
    ) -> dict[str, SimulationResults]:
        """
        Run simulations under different market scenarios.

        Args:
            portfolio: Portfolio to simulate
            scenarios: Dict mapping scenario name to (return_adjustment, vol_multiplier)

        Returns:
            Dict mapping scenario name to SimulationResults
        """
        results = {}

        for name, (return_adj, vol_mult) in scenarios.items():
            # Create modified portfolio statistics
            modified_portfolio = portfolio.copy()
            modified_portfolio.adjust_statistics(return_adj, vol_mult)

            results[name] = self.run_simulation(modified_portfolio)

        return results
