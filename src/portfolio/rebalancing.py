"""
Portfolio Rebalancing Strategies.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
import numpy as np


class RebalanceFrequency(Enum):
    """Rebalancing frequency options."""
    DAILY = 1
    WEEKLY = 5
    MONTHLY = 21
    QUARTERLY = 63
    SEMI_ANNUALLY = 126
    ANNUALLY = 252


class RebalancingStrategy(ABC):
    """Abstract base class for rebalancing strategies."""

    @abstractmethod
    def should_rebalance(
        self,
        day: int,
        current_weights: np.ndarray,
        target_weights: np.ndarray
    ) -> bool:
        """
        Determine if rebalancing should occur.

        Args:
            day: Current simulation day
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights

        Returns:
            True if rebalancing should occur
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name for display."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return strategy description."""
        pass


class NoRebalancing(RebalancingStrategy):
    """Buy and hold strategy - no rebalancing."""

    def should_rebalance(
        self,
        day: int,
        current_weights: np.ndarray,
        target_weights: np.ndarray
    ) -> bool:
        return False

    @property
    def name(self) -> str:
        return "Buy & Hold"

    @property
    def description(self) -> str:
        return "No rebalancing - maintain initial positions"


class PeriodicRebalancing(RebalancingStrategy):
    """Rebalance at fixed time intervals."""

    def __init__(self, frequency: RebalanceFrequency = RebalanceFrequency.QUARTERLY):
        self.frequency = frequency

    def should_rebalance(
        self,
        day: int,
        current_weights: np.ndarray,
        target_weights: np.ndarray
    ) -> bool:
        return day > 0 and day % self.frequency.value == 0

    @property
    def name(self) -> str:
        freq_names = {
            RebalanceFrequency.DAILY: "Daily",
            RebalanceFrequency.WEEKLY: "Weekly",
            RebalanceFrequency.MONTHLY: "Monthly",
            RebalanceFrequency.QUARTERLY: "Quarterly",
            RebalanceFrequency.SEMI_ANNUALLY: "Semi-Annual",
            RebalanceFrequency.ANNUALLY: "Annual",
        }
        return f"{freq_names[self.frequency]} Rebalancing"

    @property
    def description(self) -> str:
        return f"Rebalance every {self.frequency.value} trading days"


class ThresholdRebalancing(RebalancingStrategy):
    """Rebalance when weights deviate beyond threshold."""

    def __init__(self, threshold: float = 0.05):
        """
        Initialize threshold rebalancing.

        Args:
            threshold: Maximum allowed deviation from target (e.g., 0.05 = 5%)
        """
        self.threshold = threshold

    def should_rebalance(
        self,
        day: int,
        current_weights: np.ndarray,
        target_weights: np.ndarray
    ) -> bool:
        # Handle 2D array (multiple simulations)
        if current_weights.ndim == 2:
            max_deviation = np.max(np.abs(current_weights - target_weights), axis=1)
            return np.any(max_deviation > self.threshold)

        max_deviation = np.max(np.abs(current_weights - target_weights))
        return max_deviation > self.threshold

    @property
    def name(self) -> str:
        return f"Threshold ({self.threshold:.0%})"

    @property
    def description(self) -> str:
        return f"Rebalance when any weight deviates by more than {self.threshold:.0%}"


class BandRebalancing(RebalancingStrategy):
    """Rebalance only weights that exceed their bands."""

    def __init__(self, band_width: float = 0.05):
        """
        Initialize band rebalancing.

        Args:
            band_width: Width of allowed band around target weight
        """
        self.band_width = band_width

    def should_rebalance(
        self,
        day: int,
        current_weights: np.ndarray,
        target_weights: np.ndarray
    ) -> bool:
        deviations = np.abs(current_weights - target_weights)
        return np.any(deviations > self.band_width)

    @property
    def name(self) -> str:
        return f"Band ({self.band_width:.0%})"

    @property
    def description(self) -> str:
        return f"Rebalance assets that drift outside ±{self.band_width:.0%} band"


class HybridRebalancing(RebalancingStrategy):
    """Combine periodic and threshold rebalancing."""

    def __init__(
        self,
        frequency: RebalanceFrequency = RebalanceFrequency.QUARTERLY,
        threshold: float = 0.10
    ):
        self.periodic = PeriodicRebalancing(frequency)
        self.threshold_check = ThresholdRebalancing(threshold)
        self.frequency = frequency
        self.threshold = threshold

    def should_rebalance(
        self,
        day: int,
        current_weights: np.ndarray,
        target_weights: np.ndarray
    ) -> bool:
        # Rebalance on schedule OR when threshold exceeded
        return (
            self.periodic.should_rebalance(day, current_weights, target_weights) or
            self.threshold_check.should_rebalance(day, current_weights, target_weights)
        )

    @property
    def name(self) -> str:
        return f"Hybrid ({self.frequency.name} / {self.threshold:.0%})"

    @property
    def description(self) -> str:
        return f"Rebalance {self.frequency.name.lower()} or when deviation exceeds {self.threshold:.0%}"


def get_available_strategies() -> list[RebalancingStrategy]:
    """Return list of all available rebalancing strategies."""
    return [
        NoRebalancing(),
        PeriodicRebalancing(RebalanceFrequency.MONTHLY),
        PeriodicRebalancing(RebalanceFrequency.QUARTERLY),
        PeriodicRebalancing(RebalanceFrequency.ANNUALLY),
        ThresholdRebalancing(0.05),
        ThresholdRebalancing(0.10),
        BandRebalancing(0.05),
        HybridRebalancing(RebalanceFrequency.QUARTERLY, 0.10),
    ]


def get_strategy_by_name(name: str) -> Optional[RebalancingStrategy]:
    """Get a strategy by its name."""
    for strategy in get_available_strategies():
        if strategy.name == name:
            return strategy
    return None
