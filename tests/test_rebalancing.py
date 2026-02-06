"""
Tests for portfolio rebalancing strategies.
"""
import numpy as np
import pytest

from src.portfolio.rebalancing import (
    NoRebalancing,
    PeriodicRebalancing,
    ThresholdRebalancing,
    BandRebalancing,
    HybridRebalancing,
    RebalanceFrequency,
    get_available_strategies,
    get_strategy_by_name,
)


class TestNoRebalancing:
    def test_never_rebalances(self):
        strategy = NoRebalancing()
        weights = np.array([0.6, 0.4])
        target = np.array([0.5, 0.5])
        for day in range(1, 300):
            assert strategy.should_rebalance(day, weights, target) == False

    def test_name(self):
        assert NoRebalancing().name == "Buy & Hold"


class TestPeriodicRebalancing:
    def test_quarterly_rebalancing(self):
        strategy = PeriodicRebalancing(RebalanceFrequency.QUARTERLY)
        weights = np.array([0.6, 0.4])
        target = np.array([0.5, 0.5])

        assert strategy.should_rebalance(0, weights, target) == False
        assert strategy.should_rebalance(62, weights, target) == False
        assert strategy.should_rebalance(63, weights, target) == True
        assert strategy.should_rebalance(126, weights, target) == True

    def test_monthly_rebalancing(self):
        strategy = PeriodicRebalancing(RebalanceFrequency.MONTHLY)
        weights = np.array([0.5, 0.5])
        target = np.array([0.5, 0.5])

        assert strategy.should_rebalance(21, weights, target) == True
        assert strategy.should_rebalance(42, weights, target) == True
        assert strategy.should_rebalance(10, weights, target) == False

    def test_name(self):
        strategy = PeriodicRebalancing(RebalanceFrequency.QUARTERLY)
        assert "Quarterly" in strategy.name

    def test_description(self):
        strategy = PeriodicRebalancing(RebalanceFrequency.QUARTERLY)
        assert "63" in strategy.description


class TestThresholdRebalancing:
    def test_below_threshold_no_rebalance(self):
        strategy = ThresholdRebalancing(threshold=0.05)
        current = np.array([0.58, 0.42])
        target = np.array([0.60, 0.40])
        assert strategy.should_rebalance(1, current, target) == False

    def test_above_threshold_rebalance(self):
        strategy = ThresholdRebalancing(threshold=0.05)
        current = np.array([0.70, 0.30])
        target = np.array([0.60, 0.40])
        assert strategy.should_rebalance(1, current, target) == True

    def test_2d_weights(self):
        strategy = ThresholdRebalancing(threshold=0.05)
        # 3 simulations, 2 assets
        current = np.array([
            [0.58, 0.42],
            [0.70, 0.30],  # This one exceeds threshold
            [0.61, 0.39],
        ])
        target = np.array([0.60, 0.40])
        assert strategy.should_rebalance(1, current, target) == True

    def test_2d_weights_none_exceed(self):
        strategy = ThresholdRebalancing(threshold=0.05)
        current = np.array([
            [0.58, 0.42],
            [0.62, 0.38],
            [0.61, 0.39],
        ])
        target = np.array([0.60, 0.40])
        assert strategy.should_rebalance(1, current, target) == False

    def test_zero_weight_sum_1d(self):
        strategy = ThresholdRebalancing(threshold=0.05)
        current = np.array([0.0, 0.0])
        target = np.array([0.60, 0.40])
        assert strategy.should_rebalance(1, current, target) == True

    def test_zero_weight_sum_2d(self):
        strategy = ThresholdRebalancing(threshold=0.05)
        current = np.array([
            [0.0, 0.0],
            [0.60, 0.40],
        ])
        target = np.array([0.60, 0.40])
        # Zero row normalizes safely (weight_sums replaced with 1.0)
        result = strategy.should_rebalance(1, current, target)
        assert isinstance(result, (bool, np.bool_))

    def test_name(self):
        strategy = ThresholdRebalancing(threshold=0.05)
        assert "5%" in strategy.name


class TestBandRebalancing:
    def test_within_band(self):
        strategy = BandRebalancing(band_width=0.05)
        current = np.array([0.62, 0.38])
        target = np.array([0.60, 0.40])
        assert strategy.should_rebalance(1, current, target) == False

    def test_outside_band(self):
        strategy = BandRebalancing(band_width=0.05)
        current = np.array([0.70, 0.30])
        target = np.array([0.60, 0.40])
        assert strategy.should_rebalance(1, current, target) == True


class TestHybridRebalancing:
    def test_periodic_triggers(self):
        strategy = HybridRebalancing(
            frequency=RebalanceFrequency.QUARTERLY,
            threshold=0.10
        )
        current = np.array([0.60, 0.40])  # No deviation
        target = np.array([0.60, 0.40])
        assert strategy.should_rebalance(63, current, target) == True

    def test_threshold_triggers(self):
        strategy = HybridRebalancing(
            frequency=RebalanceFrequency.QUARTERLY,
            threshold=0.10
        )
        current = np.array([0.80, 0.20])  # Large deviation
        target = np.array([0.60, 0.40])
        assert strategy.should_rebalance(10, current, target) == True

    def test_neither_triggers(self):
        strategy = HybridRebalancing(
            frequency=RebalanceFrequency.QUARTERLY,
            threshold=0.10
        )
        current = np.array([0.62, 0.38])  # Small deviation
        target = np.array([0.60, 0.40])
        assert strategy.should_rebalance(10, current, target) == False

    def test_name(self):
        strategy = HybridRebalancing(
            frequency=RebalanceFrequency.QUARTERLY,
            threshold=0.10
        )
        assert "Hybrid" in strategy.name


class TestHelperFunctions:
    def test_get_available_strategies(self):
        strategies = get_available_strategies()
        assert len(strategies) >= 5
        names = [s.name for s in strategies]
        assert "Buy & Hold" in names

    def test_get_strategy_by_name_found(self):
        strategy = get_strategy_by_name("Buy & Hold")
        assert strategy is not None
        assert isinstance(strategy, NoRebalancing)

    def test_get_strategy_by_name_not_found(self):
        strategy = get_strategy_by_name("Nonexistent Strategy")
        assert strategy is None
