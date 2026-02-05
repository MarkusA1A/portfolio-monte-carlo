"""
Steuer- und Transaktionskostenrechner für Monte Carlo Simulation.

Unterstützt:
- Österreichische Kapitalertragssteuer (KESt): 27,5%
- Transaktionskosten: Prozentual oder Flat Fee
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class TaxConfig:
    """Konfiguration für Kapitalertragssteuer."""
    tax_rate: float = 0.275  # Österreichische KESt: 27,5%

    def __post_init__(self):
        if not 0 <= self.tax_rate <= 1:
            raise ValueError("Steuersatz muss zwischen 0 und 1 liegen")


@dataclass
class TransactionCostConfig:
    """Konfiguration für Transaktionskosten."""
    use_percentage: bool = True  # True = prozentual, False = Flat Fee
    percentage_fee: float = 0.001  # 0,1% Standard
    flat_fee_per_trade: float = 5.0  # €5 pro Trade

    def __post_init__(self):
        if self.percentage_fee < 0:
            raise ValueError("Prozentuale Gebühr kann nicht negativ sein")
        if self.flat_fee_per_trade < 0:
            raise ValueError("Flat Fee kann nicht negativ sein")


@dataclass
class TaxCostResults:
    """Ergebnisse der Steuer- und Kostenberechnung."""
    total_taxes_paid: np.ndarray
    total_transaction_costs: np.ndarray
    rebalancing_events: np.ndarray
    final_value_before_tax: np.ndarray
    final_value_after_tax: np.ndarray
    realized_gains: np.ndarray
    unrealized_gains: np.ndarray
    tax_rate: float = 0.275

    @property
    def mean_taxes_paid(self) -> float:
        return float(np.mean(self.total_taxes_paid))

    @property
    def mean_transaction_costs(self) -> float:
        return float(np.mean(self.total_transaction_costs))

    @property
    def mean_rebalancing_events(self) -> float:
        return float(np.mean(self.rebalancing_events))

    @property
    def mean_final_before_tax(self) -> float:
        return float(np.mean(self.final_value_before_tax))

    @property
    def mean_final_after_tax(self) -> float:
        return float(np.mean(self.final_value_after_tax))

    @property
    def mean_realized_gains(self) -> float:
        return float(np.mean(self.realized_gains))

    @property
    def mean_unrealized_gains(self) -> float:
        return float(np.mean(self.unrealized_gains))

    @property
    def total_cost_impact(self) -> float:
        before = self.mean_final_before_tax
        after = self.mean_final_after_tax
        if before == 0:
            return 0.0
        return (before - after) / before

    @property
    def effective_tax_rate(self) -> float:
        gains = self.mean_realized_gains
        if gains <= 0:
            return 0.0
        return self.mean_taxes_paid / gains

    @property
    def median_final_after_tax(self) -> float:
        return float(np.median(self.final_value_after_tax))

    def get_percentile_after_tax(self, p: float) -> float:
        return float(np.percentile(self.final_value_after_tax, p))


class TaxCostCalculator:
    """Berechnet Steuern und Transaktionskosten während der Simulation."""

    def __init__(
        self,
        tax_config: TaxConfig,
        cost_config: TransactionCostConfig,
        num_simulations: int,
        num_assets: int,
        initial_value: float,
        dtype: np.dtype = np.float32
    ):
        self.tax_config = tax_config
        self.cost_config = cost_config
        self.num_simulations = num_simulations
        self.num_assets = num_assets
        self.initial_value = initial_value
        self.dtype = dtype

        # Tracking-Arrays
        self.total_taxes_paid = np.zeros(num_simulations, dtype=dtype)
        self.total_transaction_costs = np.zeros(num_simulations, dtype=dtype)
        self.rebalancing_events = np.zeros(num_simulations, dtype=np.int32)
        self.realized_gains = np.zeros(num_simulations, dtype=dtype)
        self.cost_basis = np.zeros((num_simulations, num_assets), dtype=dtype)

    def initialize_cost_basis(self, weights: np.ndarray):
        """Initialisiert Cost Basis mit dem Anfangsinvestment."""
        self.cost_basis = np.tile(
            (self.initial_value * weights).astype(self.dtype),
            (self.num_simulations, 1)
        )

    def calculate_transaction_costs(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_values: np.ndarray
    ) -> np.ndarray:
        """Berechnet Transaktionskosten für ein Rebalancing-Event."""
        weight_changes = np.abs(old_weights - new_weights)
        turnover = np.sum(weight_changes, axis=1) / 2
        trade_volume = turnover * portfolio_values

        if self.cost_config.use_percentage:
            costs = trade_volume * self.cost_config.percentage_fee
        else:
            num_trades = np.sum(weight_changes > 0.001, axis=1)
            costs = num_trades * self.cost_config.flat_fee_per_trade

        return costs.astype(self.dtype)

    def calculate_realized_gains_and_tax(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Berechnet realisierte Gewinne und Steuern bei Rebalancing."""
        weight_diff = old_weights - new_weights
        current_values = old_weights * portfolio_values[:, np.newaxis]
        gains_per_asset = current_values - self.cost_basis

        with np.errstate(divide='ignore', invalid='ignore'):
            sell_fraction = np.where(
                old_weights > 0.001,
                np.maximum(weight_diff, 0) / old_weights,
                0
            )

        realized_per_asset = sell_fraction * gains_per_asset
        total_realized = np.sum(realized_per_asset, axis=1)

        taxable_gains = np.maximum(total_realized, 0)
        taxes = taxable_gains * self.tax_config.tax_rate

        # Cost Basis aktualisieren
        self.cost_basis = self.cost_basis * (1 - sell_fraction)
        buy_fraction = np.maximum(new_weights - old_weights, 0)
        buy_value = buy_fraction * portfolio_values[:, np.newaxis]
        self.cost_basis = self.cost_basis + buy_value

        return total_realized.astype(self.dtype), taxes.astype(self.dtype)

    def process_rebalancing(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_values: np.ndarray
    ) -> np.ndarray:
        """Verarbeitet ein Rebalancing-Event vollständig."""
        tx_costs = self.calculate_transaction_costs(
            old_weights, new_weights, portfolio_values
        )
        realized, taxes = self.calculate_realized_gains_and_tax(
            old_weights, new_weights, portfolio_values
        )

        self.total_transaction_costs += tx_costs
        self.total_taxes_paid += taxes
        self.realized_gains += realized
        self.rebalancing_events += 1

        return tx_costs + taxes

    def get_results(
        self,
        final_values_before: np.ndarray,
        final_values_after: np.ndarray
    ) -> TaxCostResults:
        """Erstellt das Ergebnis-Objekt."""
        final_cost_basis = np.sum(self.cost_basis, axis=1)
        unrealized = final_values_after - final_cost_basis

        return TaxCostResults(
            total_taxes_paid=self.total_taxes_paid.copy(),
            total_transaction_costs=self.total_transaction_costs.copy(),
            rebalancing_events=self.rebalancing_events.copy(),
            final_value_before_tax=final_values_before.copy(),
            final_value_after_tax=final_values_after.copy(),
            realized_gains=self.realized_gains.copy(),
            unrealized_gains=unrealized,
            tax_rate=self.tax_config.tax_rate
        )
