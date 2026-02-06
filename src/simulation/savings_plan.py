"""
Savings Plan (Sparplan) Simulation - Regular investment contributions.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.portfolio.portfolio import Portfolio


@dataclass
class SavingsPlanResults:
    """Results from savings plan simulation."""

    portfolio_values: np.ndarray  # Shape: (num_simulations, time_horizon)
    total_contributions: np.ndarray  # Shape: (time_horizon,)
    final_values: np.ndarray  # Shape: (num_simulations,)
    total_invested: float
    time_horizon: int
    num_simulations: int
    monthly_contribution: float

    @property
    def mean_final_value(self) -> float:
        return float(np.mean(self.final_values))

    @property
    def median_final_value(self) -> float:
        return float(np.median(self.final_values))

    @property
    def mean_profit(self) -> float:
        return self.mean_final_value - self.total_invested

    @property
    def mean_return(self) -> float:
        if self.total_invested == 0:
            return 0.0
        return (self.mean_final_value - self.total_invested) / self.total_invested

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.final_values, p))


class SavingsPlanSimulator:
    """
    Monte Carlo Simulator for savings plans with regular contributions.
    """

    def __init__(
        self,
        num_simulations: int = 10000,
        time_horizon_years: int = 10,
        random_seed: int | None = None
    ):
        self.num_simulations = num_simulations
        self.time_horizon_months = time_horizon_years * 12
        self.time_horizon_years = time_horizon_years
        self.rng = np.random.default_rng(random_seed)

    def run_simulation(
        self,
        portfolio: "Portfolio",
        monthly_contribution: float,
        initial_investment: float = 0.0
    ) -> SavingsPlanResults:
        """
        Run savings plan simulation with monthly contributions.

        Args:
            portfolio: Portfolio with asset statistics
            monthly_contribution: Monthly contribution amount
            initial_investment: Optional initial lump sum

        Returns:
            SavingsPlanResults with simulation data
        """
        # Get monthly statistics (convert from daily)
        daily_mean = portfolio.expected_return()
        daily_std = portfolio.expected_volatility()

        # Convert to monthly (approx 21 trading days per month)
        monthly_mean = daily_mean * 21
        monthly_std = daily_std * np.sqrt(21)

        # Generate random monthly returns
        monthly_returns = self.rng.normal(
            monthly_mean,
            monthly_std,
            (self.num_simulations, self.time_horizon_months)
        )

        # Initialize portfolio values
        portfolio_values = np.zeros((self.num_simulations, self.time_horizon_months + 1))
        portfolio_values[:, 0] = initial_investment

        # Track total contributions
        total_contributions = np.zeros(self.time_horizon_months + 1)
        total_contributions[0] = initial_investment

        # Simulate month by month
        for month in range(self.time_horizon_months):
            # Add monthly contribution at start of month
            portfolio_values[:, month] += monthly_contribution

            # Apply monthly return
            portfolio_values[:, month + 1] = portfolio_values[:, month] * (1 + monthly_returns[:, month])

            # Track contributions
            total_contributions[month + 1] = total_contributions[month] + monthly_contribution

        total_invested = initial_investment + monthly_contribution * self.time_horizon_months

        return SavingsPlanResults(
            portfolio_values=portfolio_values,
            total_contributions=total_contributions,
            final_values=portfolio_values[:, -1],
            total_invested=total_invested,
            time_horizon=self.time_horizon_months,
            num_simulations=self.num_simulations,
            monthly_contribution=monthly_contribution
        )
