"""
Scenario Analysis for stress testing portfolios.
"""
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.simulation.monte_carlo import SimulationResults


class ScenarioType(Enum):
    """Predefined market scenarios."""
    BULL_MARKET = "bull"
    NORMAL = "normal"
    BEAR_MARKET = "bear"
    CRASH = "crash"
    HIGH_VOLATILITY = "high_vol"
    STAGFLATION = "stagflation"


@dataclass
class Scenario:
    """Market scenario definition."""

    name: str
    description: str
    return_adjustment: float  # Added to annual return (e.g., 0.05 = +5%)
    volatility_multiplier: float  # Multiplied with volatility (e.g., 1.5 = +50% vol)
    color: str  # For visualization

    def apply_to_stats(
        self,
        mean_daily_return: float,
        std_daily_return: float
    ) -> tuple[float, float]:
        """Apply scenario adjustments to daily statistics."""
        # Convert annual adjustment to daily
        adjusted_mean = mean_daily_return + self.return_adjustment / 252
        adjusted_std = std_daily_return * self.volatility_multiplier
        return adjusted_mean, adjusted_std


# Predefined scenarios
SCENARIOS = {
    ScenarioType.BULL_MARKET: Scenario(
        name="Bullenmarkt",
        description="Starkes Wirtschaftswachstum, optimistische Stimmung",
        return_adjustment=0.08,  # +8% annual
        volatility_multiplier=0.8,  # Lower volatility
        color="green"
    ),
    ScenarioType.NORMAL: Scenario(
        name="Normal",
        description="Durchschnittliche Marktbedingungen",
        return_adjustment=0.0,
        volatility_multiplier=1.0,
        color="blue"
    ),
    ScenarioType.BEAR_MARKET: Scenario(
        name="Bärenmarkt",
        description="Wirtschaftliche Abschwächung, pessimistische Stimmung",
        return_adjustment=-0.10,  # -10% annual
        volatility_multiplier=1.3,  # Higher volatility
        color="orange"
    ),
    ScenarioType.CRASH: Scenario(
        name="Crash",
        description="Schwere Marktkrise (wie 2008 oder 2020)",
        return_adjustment=-0.30,  # -30% annual
        volatility_multiplier=2.5,  # Much higher volatility
        color="red"
    ),
    ScenarioType.HIGH_VOLATILITY: Scenario(
        name="Hohe Volatilität",
        description="Unsichere Märkte mit starken Schwankungen",
        return_adjustment=-0.02,
        volatility_multiplier=2.0,
        color="purple"
    ),
    ScenarioType.STAGFLATION: Scenario(
        name="Stagflation",
        description="Niedrige Renditen bei hoher Inflation",
        return_adjustment=-0.05,
        volatility_multiplier=1.5,
        color="brown"
    ),
}


@dataclass
class CustomScenario:
    """User-defined custom scenario."""

    name: str
    return_adjustment: float
    volatility_multiplier: float

    def to_scenario(self) -> Scenario:
        return Scenario(
            name=self.name,
            description="Benutzerdefiniertes Szenario",
            return_adjustment=self.return_adjustment,
            volatility_multiplier=self.volatility_multiplier,
            color="gray"
        )


def get_scenario(scenario_type: ScenarioType) -> Scenario:
    """Get a predefined scenario."""
    return SCENARIOS[scenario_type]


def get_all_scenarios() -> dict[ScenarioType, Scenario]:
    """Get all predefined scenarios."""
    return SCENARIOS.copy()


@dataclass
class ScenarioComparisonResults:
    """Results from comparing multiple scenarios."""

    scenario_results: dict[str, "SimulationResults"]
    scenario_definitions: dict[str, Scenario]

    def get_summary_df(self, initial_value: float):
        """Get summary DataFrame for all scenarios."""
        import pandas as pd

        data = []
        for name, results in self.scenario_results.items():
            scenario = self.scenario_definitions[name]
            data.append({
                "Szenario": name,
                "Erwarteter Endwert": results.mean_final_value,
                "Median": results.median_final_value,
                "Min": results.min_value,
                "Max": results.max_value,
                "Rendite (%)": results.mean_return * 100,
                "Std.Abw.": results.std_final_value,
            })

        return pd.DataFrame(data)
