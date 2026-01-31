"""
Monte Carlo Simulation Engine for Portfolio Analysis

Memory-optimized version using float32 and optional price path storage.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from src.portfolio.portfolio import Portfolio


@dataclass
class SimulationResults:
    """Container for Monte Carlo simulation results."""

    portfolio_values: np.ndarray  # Shape: (num_simulations, time_horizon)
    final_values: np.ndarray  # Shape: (num_simulations,)
    returns: np.ndarray  # Shape: (num_simulations,)
    time_horizon: int
    num_simulations: int
    # Optional - only stored when needed for detailed analysis
    price_paths: np.ndarray | None = field(default=None)

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
    Memory-optimized: uses float32 and avoids storing unnecessary data.
    """

    # Memory threshold in MB - above this, use memory-efficient mode
    MEMORY_EFFICIENT_THRESHOLD_MB = 200

    def __init__(
        self,
        num_simulations: int = 10000,
        time_horizon: int = 252,
        random_seed: int | None = None,
        use_float32: bool = True
    ):
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon  # Trading days
        self.rng = np.random.default_rng(random_seed)
        self.dtype = np.float32 if use_float32 else np.float64

    def _estimate_memory_mb(self, num_assets: int, store_price_paths: bool) -> float:
        """Estimate memory usage in MB."""
        bytes_per_float = 4 if self.dtype == np.float32 else 8
        portfolio_values = self.num_simulations * (self.time_horizon + 1) * bytes_per_float
        randoms = self.num_simulations * self.time_horizon * num_assets * bytes_per_float

        if store_price_paths:
            price_paths = self.num_simulations * (self.time_horizon + 1) * num_assets * bytes_per_float
        else:
            price_paths = 0

        total_bytes = portfolio_values + randoms + price_paths
        return total_bytes / (1024 * 1024)

    def _safe_cholesky(self, cov_matrix: np.ndarray, max_tries: int = 5) -> np.ndarray:
        """
        Perform Cholesky decomposition with regularization fallback.

        If the covariance matrix is not positive definite (e.g., due to highly
        correlated assets), adds small regularization to make it decomposable.
        """
        # First try without regularization
        try:
            return np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            pass

        # Try with increasing regularization
        n = cov_matrix.shape[0]
        reg_factor = 1e-8

        for _ in range(max_tries):
            try:
                regularized = cov_matrix + np.eye(n) * reg_factor
                return np.linalg.cholesky(regularized)
            except np.linalg.LinAlgError:
                reg_factor *= 10

        # Final fallback: use eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        cov_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return np.linalg.cholesky(cov_fixed)

    def run_simulation(
        self,
        portfolio: "Portfolio",
        rebalancing_strategy=None,
        store_price_paths: bool = False
    ) -> SimulationResults:
        """
        Run Monte Carlo simulation for the given portfolio.

        Args:
            portfolio: Portfolio object with assets and weights
            rebalancing_strategy: Optional rebalancing strategy
            store_price_paths: If True, store full price paths (uses more memory)

        Returns:
            SimulationResults with simulation data
        """
        num_assets = len(portfolio.assets)

        # Check if we should use memory-efficient mode
        estimated_memory = self._estimate_memory_mb(num_assets, store_price_paths)
        use_batched = estimated_memory > self.MEMORY_EFFICIENT_THRESHOLD_MB

        if use_batched and self.num_simulations > 5000:
            return self._run_batched_simulation(portfolio, rebalancing_strategy)

        # Get portfolio statistics
        mean_returns = portfolio.get_mean_returns().astype(self.dtype)
        cov_matrix = portfolio.get_covariance_matrix().astype(np.float64)  # Keep float64 for Cholesky

        # Generate correlated random returns using Cholesky decomposition
        cholesky = self._safe_cholesky(cov_matrix).astype(self.dtype)

        # Generate uncorrelated standard normal random variables
        uncorrelated_randoms = self.rng.standard_normal(
            (self.num_simulations, self.time_horizon, num_assets)
        ).astype(self.dtype)

        # Transform to correlated returns
        correlated_randoms = np.einsum('ijk,lk->ijl', uncorrelated_randoms, cholesky)

        # Free memory
        del uncorrelated_randoms

        # Calculate daily returns: mean + volatility * random
        daily_returns = mean_returns + correlated_randoms

        # Free memory
        del correlated_randoms

        # Initialize arrays
        initial_prices = portfolio.get_current_prices().astype(self.dtype)
        weights = portfolio.weights.astype(self.dtype)
        initial_value = self.dtype(portfolio.initial_value)

        if rebalancing_strategy is None:
            # Memory-efficient buy and hold simulation
            portfolio_values, price_paths = self._simulate_buy_and_hold(
                daily_returns, initial_prices, weights, initial_value,
                store_price_paths
            )
        else:
            # With rebalancing - needs price paths internally
            portfolio_values, price_paths = self._simulate_with_rebalancing_optimized(
                daily_returns, initial_prices, weights, initial_value,
                rebalancing_strategy, store_price_paths
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

    def _simulate_buy_and_hold(
        self,
        daily_returns: np.ndarray,
        initial_prices: np.ndarray,
        weights: np.ndarray,
        initial_value: float,
        store_price_paths: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Memory-efficient buy and hold simulation."""
        num_sims, time_horizon, num_assets = daily_returns.shape

        # Calculate shares bought initially
        shares = (initial_value * weights) / initial_prices

        # Initialize portfolio values
        portfolio_values = np.zeros((num_sims, time_horizon + 1), dtype=self.dtype)
        portfolio_values[:, 0] = initial_value

        # Track current prices (overwrite each step to save memory)
        current_prices = np.tile(initial_prices, (num_sims, 1))

        # Store price paths only if requested
        if store_price_paths:
            price_paths = np.zeros((num_sims, time_horizon + 1, num_assets), dtype=self.dtype)
            price_paths[:, 0, :] = initial_prices
        else:
            price_paths = None

        # Simulate day by day
        for t in range(time_horizon):
            current_prices = current_prices * (1 + daily_returns[:, t, :])
            portfolio_values[:, t + 1] = np.sum(current_prices * shares, axis=1)

            if store_price_paths:
                price_paths[:, t + 1, :] = current_prices

        return portfolio_values, price_paths

    def _simulate_with_rebalancing_optimized(
        self,
        daily_returns: np.ndarray,
        initial_prices: np.ndarray,
        weights: np.ndarray,
        initial_value: float,
        strategy,
        store_price_paths: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Memory-efficient simulation with rebalancing."""
        num_sims, time_horizon, num_assets = daily_returns.shape

        # Initialize
        portfolio_values = np.zeros((num_sims, time_horizon + 1), dtype=self.dtype)
        portfolio_values[:, 0] = initial_value

        current_weights = np.tile(weights, (num_sims, 1))
        current_prices = np.tile(initial_prices, (num_sims, 1))

        if store_price_paths:
            price_paths = np.zeros((num_sims, time_horizon + 1, num_assets), dtype=self.dtype)
            price_paths[:, 0, :] = initial_prices
        else:
            price_paths = None

        for t in range(time_horizon):
            # Calculate returns for this period
            period_returns = daily_returns[:, t, :]

            # Update prices
            current_prices = current_prices * (1 + period_returns)

            if store_price_paths:
                price_paths[:, t + 1, :] = current_prices

            # Update portfolio value
            weighted_returns = np.sum(current_weights * period_returns, axis=1)
            portfolio_values[:, t + 1] = portfolio_values[:, t] * (1 + weighted_returns)

            # Update weights based on price changes (vectorized)
            weight_factors = (1 + period_returns) / (1 + weighted_returns[:, np.newaxis])
            current_weights = current_weights * weight_factors

            # Check if rebalancing is needed
            if strategy.should_rebalance(t + 1, current_weights, weights):
                current_weights = np.tile(weights, (num_sims, 1))

        return portfolio_values, price_paths

    def _run_batched_simulation(
        self,
        portfolio: "Portfolio",
        rebalancing_strategy=None,
        batch_size: int = 2500
    ) -> SimulationResults:
        """Run simulation in batches to reduce peak memory usage."""
        num_batches = (self.num_simulations + batch_size - 1) // batch_size

        all_portfolio_values = []
        all_final_values = []
        all_returns = []

        original_num_sims = self.num_simulations

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, original_num_sims)
            current_batch_size = end_idx - start_idx

            # Temporarily adjust num_simulations
            self.num_simulations = current_batch_size

            # Run simulation for this batch (don't store price paths in batched mode)
            batch_result = self.run_simulation(
                portfolio, rebalancing_strategy, store_price_paths=False
            )

            all_portfolio_values.append(batch_result.portfolio_values)
            all_final_values.append(batch_result.final_values)
            all_returns.append(batch_result.returns)

        # Restore original setting
        self.num_simulations = original_num_sims

        # Combine results
        portfolio_values = np.concatenate(all_portfolio_values, axis=0)
        final_values = np.concatenate(all_final_values, axis=0)
        returns = np.concatenate(all_returns, axis=0)

        return SimulationResults(
            price_paths=None,  # Not available in batched mode
            portfolio_values=portfolio_values,
            final_values=final_values,
            returns=returns,
            time_horizon=self.time_horizon,
            num_simulations=original_num_sims
        )

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
            modified_portfolio = portfolio.copy()
            modified_portfolio.adjust_statistics(return_adj, vol_mult)
            results[name] = self.run_simulation(modified_portfolio)

        return results
