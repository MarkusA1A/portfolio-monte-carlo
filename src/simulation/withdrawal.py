"""
Entnahme-Simulation (Withdrawal Simulation)

Simuliert den Vermögensverlauf bei regelmäßigen Entnahmen.
Wichtig für die Ruhestandsplanung: "Wie lange reicht mein Geld?"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WithdrawalTaxResults:
    """Steuer-Ergebnisse der Entnahme-Simulation."""
    total_taxes_paid: np.ndarray          # Pro Simulation
    total_gross_withdrawn: np.ndarray     # Brutto-Entnahmen
    total_net_withdrawn: np.ndarray       # Netto nach Steuer
    realized_gains: np.ndarray            # Realisierte Gewinne
    tax_rate: float = 0.275               # KESt-Satz

    @property
    def mean_taxes_paid(self) -> float:
        return float(np.mean(self.total_taxes_paid))

    @property
    def mean_gross_withdrawn(self) -> float:
        return float(np.mean(self.total_gross_withdrawn))

    @property
    def mean_net_withdrawn(self) -> float:
        return float(np.mean(self.total_net_withdrawn))

    @property
    def mean_realized_gains(self) -> float:
        return float(np.mean(self.realized_gains))

    @property
    def effective_tax_rate(self) -> float:
        """Effektiver Steuersatz (gezahlte Steuern / realisierte Gewinne)."""
        gains = self.mean_realized_gains
        if gains <= 0:
            return 0.0
        return self.mean_taxes_paid / gains


@dataclass
class WithdrawalResults:
    """Ergebnisse der Entnahme-Simulation."""
    # Simulationspfade
    portfolio_paths: np.ndarray  # Shape: (n_simulations, n_periods)
    withdrawal_paths: np.ndarray  # Tatsächliche Entnahmen pro Periode

    # Kernmetriken
    initial_value: float
    monthly_withdrawal: float
    total_periods: int  # Monate
    n_simulations: int

    # Erfolgsmetriken
    success_rate: float  # Anteil Simulationen, wo Geld nicht ausgeht
    median_final_value: float
    mean_final_value: float

    # Vermögens-Percentile am Ende
    percentile_5: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_95: float

    # Zeit bis Erschöpfung (für gescheiterte Simulationen)
    depletion_times: np.ndarray  # Monate bis Vermögen = 0 (np.inf wenn nicht erschöpft)
    median_depletion_time: Optional[float]  # Median nur für gescheiterte
    earliest_depletion: Optional[int]  # Früheste Erschöpfung

    # Entnahme-Statistiken
    total_withdrawn_median: float  # Median der Gesamtentnahmen
    total_withdrawn_mean: float

    # Steuer-Ergebnisse (optional)
    tax_results: Optional[WithdrawalTaxResults] = field(default=None)

    @property
    def failure_rate(self) -> float:
        """Anteil der Simulationen, wo das Geld ausgeht."""
        return 1 - self.success_rate

    @property
    def years_simulated(self) -> float:
        """Simulierte Jahre."""
        return self.total_periods / 12

    @property
    def annual_withdrawal(self) -> float:
        """Jährliche Entnahme."""
        return self.monthly_withdrawal * 12

    @property
    def withdrawal_rate(self) -> float:
        """Initiale Entnahmerate (SWR - Safe Withdrawal Rate)."""
        return self.annual_withdrawal / self.initial_value


class WithdrawalSimulator:
    """
    Monte Carlo Simulation für Entnahme-Szenarien.

    Simuliert, wie sich ein Portfolio entwickelt, wenn regelmäßig
    Geld entnommen wird (z.B. im Ruhestand).
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        random_seed: Optional[int] = None
    ):
        """
        Args:
            n_simulations: Anzahl Monte Carlo Simulationen
            random_seed: Seed für Reproduzierbarkeit
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def simulate(
        self,
        initial_value: float,
        monthly_withdrawal: float,
        expected_annual_return: float,
        annual_volatility: float,
        years: int,
        inflation_rate: float = 0.02,
        adjust_for_inflation: bool = True,
        tax_rate: Optional[float] = None,
        apply_tax_to_gains: bool = False
    ) -> WithdrawalResults:
        """
        Führt die Entnahme-Simulation durch.

        Args:
            initial_value: Anfangsvermögen
            monthly_withdrawal: Monatliche Entnahme (in heutigen €)
            expected_annual_return: Erwartete jährliche Rendite
            annual_volatility: Jährliche Volatilität (Standardabweichung)
            years: Simulationszeitraum in Jahren
            inflation_rate: Jährliche Inflation (für Kaufkraftanpassung)
            adjust_for_inflation: Entnahme an Inflation anpassen?
            tax_rate: Steuersatz (z.B. 0.275 für österreichische KESt)
            apply_tax_to_gains: Steuer auf realisierte Gewinne bei Entnahmen?

        Returns:
            WithdrawalResults mit allen Ergebnissen
        """
        n_months = years * 12

        # Monatliche Parameter
        monthly_return = expected_annual_return / 12
        monthly_volatility = annual_volatility / np.sqrt(12)
        monthly_inflation = inflation_rate / 12

        # Arrays initialisieren
        portfolio_paths = np.zeros((self.n_simulations, n_months + 1))
        portfolio_paths[:, 0] = initial_value

        withdrawal_paths = np.zeros((self.n_simulations, n_months))
        depletion_times = np.full(self.n_simulations, np.inf)

        # Steuer-Tracking (wenn aktiviert)
        if apply_tax_to_gains and tax_rate:
            taxes_paid = np.zeros((self.n_simulations, n_months))
            realized_gains = np.zeros((self.n_simulations, n_months))
        else:
            taxes_paid = None
            realized_gains = None

        # Simulation durchführen
        for sim in range(self.n_simulations):
            current_value = initial_value
            current_withdrawal = monthly_withdrawal
            cost_basis = initial_value  # Einstandswert für Steuerberechnung
            depleted = False

            for month in range(n_months):
                if depleted:
                    portfolio_paths[sim, month + 1] = 0
                    withdrawal_paths[sim, month] = 0
                    if taxes_paid is not None:
                        taxes_paid[sim, month] = 0
                        realized_gains[sim, month] = 0
                    continue

                # Entnahme (inflationsangepasst)
                if adjust_for_inflation:
                    current_withdrawal = monthly_withdrawal * ((1 + monthly_inflation) ** month)

                # Tatsächliche (Brutto-)Entnahme (maximal verfügbares Vermögen)
                gross_withdrawal = min(current_withdrawal, current_value)

                # STEUERBERECHNUNG (wenn aktiviert)
                tax = 0.0
                if apply_tax_to_gains and tax_rate and current_value > cost_basis and gross_withdrawal > 0:
                    # Gewinnanteil im Portfolio
                    gain_ratio = (current_value - cost_basis) / current_value
                    # Realisierter Gewinn durch diese Entnahme
                    realized_gain = gross_withdrawal * gain_ratio
                    # Steuer auf realisierten Gewinn
                    tax = realized_gain * tax_rate

                    # Tracking
                    taxes_paid[sim, month] = tax
                    realized_gains[sim, month] = realized_gain

                    # Cost Basis proportional reduzieren
                    if current_value > 0:
                        withdrawal_ratio = gross_withdrawal / current_value
                        cost_basis *= (1 - withdrawal_ratio)

                # Brutto-Entnahme tracken
                withdrawal_paths[sim, month] = gross_withdrawal

                # Portfolio nach Entnahme und Steuer
                current_value -= gross_withdrawal
                current_value -= tax  # Steuer zusätzlich vom Portfolio abziehen

                if current_value <= 0:
                    depleted = True
                    depletion_times[sim] = month + 1
                    portfolio_paths[sim, month + 1] = 0
                    continue

                # Rendite für diesen Monat
                monthly_r = self.rng.normal(monthly_return, monthly_volatility)
                current_value *= (1 + monthly_r)

                # Wert kann nicht negativ werden
                current_value = max(0, current_value)
                portfolio_paths[sim, month + 1] = current_value

                if current_value <= 0:
                    depleted = True
                    depletion_times[sim] = month + 1

        # Finale Werte analysieren
        final_values = portfolio_paths[:, -1]
        successful = final_values > 0

        # Erfolgsrate
        success_rate = np.mean(successful)

        # Percentile der Endwerte
        percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])

        # Entnahme-Statistiken
        total_withdrawn = np.sum(withdrawal_paths, axis=1)

        # Erschöpfungszeiten für gescheiterte Simulationen
        failed_times = depletion_times[~successful]
        median_depletion = np.median(failed_times) if len(failed_times) > 0 else None
        earliest_depletion = int(np.min(failed_times)) if len(failed_times) > 0 else None

        # Steuer-Ergebnisse erstellen (wenn aktiviert)
        tax_results = None
        if apply_tax_to_gains and tax_rate and taxes_paid is not None:
            total_taxes = np.sum(taxes_paid, axis=1)
            total_gains = np.sum(realized_gains, axis=1)
            net_withdrawn = total_withdrawn - total_taxes

            tax_results = WithdrawalTaxResults(
                total_taxes_paid=total_taxes,
                total_gross_withdrawn=total_withdrawn,
                total_net_withdrawn=net_withdrawn,
                realized_gains=total_gains,
                tax_rate=tax_rate
            )

        return WithdrawalResults(
            portfolio_paths=portfolio_paths,
            withdrawal_paths=withdrawal_paths,
            initial_value=initial_value,
            monthly_withdrawal=monthly_withdrawal,
            total_periods=n_months,
            n_simulations=self.n_simulations,
            success_rate=success_rate,
            median_final_value=np.median(final_values),
            mean_final_value=np.mean(final_values),
            percentile_5=percentiles[0],
            percentile_25=percentiles[1],
            percentile_50=percentiles[2],
            percentile_75=percentiles[3],
            percentile_95=percentiles[4],
            depletion_times=depletion_times,
            median_depletion_time=median_depletion,
            earliest_depletion=earliest_depletion,
            total_withdrawn_median=np.median(total_withdrawn),
            total_withdrawn_mean=np.mean(total_withdrawn),
            tax_results=tax_results
        )

    def find_safe_withdrawal_rate(
        self,
        initial_value: float,
        expected_annual_return: float,
        annual_volatility: float,
        years: int,
        target_success_rate: float = 0.95,
        inflation_rate: float = 0.02,
        adjust_for_inflation: bool = True,
        tax_rate: Optional[float] = None,
        apply_tax_to_gains: bool = False
    ) -> dict:
        """
        Findet die sichere Entnahmerate (SWR) für eine Zielerfolgsrate.

        Verwendet binäre Suche um die maximale monatliche Entnahme zu finden,
        bei der die gewünschte Erfolgsrate erreicht wird.

        Args:
            initial_value: Anfangsvermögen
            expected_annual_return: Erwartete jährliche Rendite
            annual_volatility: Jährliche Volatilität
            years: Simulationszeitraum
            target_success_rate: Gewünschte Erfolgsrate (z.B. 0.95 = 95%)
            inflation_rate: Jährliche Inflation
            adjust_for_inflation: Entnahme an Inflation anpassen?
            tax_rate: Steuersatz (z.B. 0.275 für österreichische KESt)
            apply_tax_to_gains: Steuer auf realisierte Gewinne bei Entnahmen?

        Returns:
            Dictionary mit SWR-Ergebnissen
        """
        # Binäre Suche für optimale Entnahme
        low = 0
        high = initial_value / (years * 12) * 2  # Obere Schranke

        best_withdrawal = 0
        best_results = None

        for _ in range(20):  # Max 20 Iterationen
            mid = (low + high) / 2

            results = self.simulate(
                initial_value=initial_value,
                monthly_withdrawal=mid,
                expected_annual_return=expected_annual_return,
                annual_volatility=annual_volatility,
                years=years,
                inflation_rate=inflation_rate,
                adjust_for_inflation=adjust_for_inflation,
                tax_rate=tax_rate,
                apply_tax_to_gains=apply_tax_to_gains
            )

            if results.success_rate >= target_success_rate:
                best_withdrawal = mid
                best_results = results
                low = mid
            else:
                high = mid

            if high - low < 1:  # Genauigkeit von 1€
                break

        # Finale Berechnung mit gefundener Rate
        if best_results is None:
            best_results = self.simulate(
                initial_value=initial_value,
                monthly_withdrawal=best_withdrawal,
                expected_annual_return=expected_annual_return,
                annual_volatility=annual_volatility,
                years=years,
                inflation_rate=inflation_rate,
                adjust_for_inflation=adjust_for_inflation,
                tax_rate=tax_rate,
                apply_tax_to_gains=apply_tax_to_gains
            )

        annual_rate = (best_withdrawal * 12) / initial_value

        return {
            'monthly_withdrawal': best_withdrawal,
            'annual_withdrawal': best_withdrawal * 12,
            'withdrawal_rate': annual_rate,
            'withdrawal_rate_pct': annual_rate * 100,
            'success_rate': best_results.success_rate,
            'target_success_rate': target_success_rate,
            'results': best_results
        }


def calculate_required_capital(
    monthly_withdrawal: float,
    expected_annual_return: float,
    annual_volatility: float,
    years: int,
    target_success_rate: float = 0.95,
    inflation_rate: float = 0.02,
    n_simulations: int = 5000,
    tax_rate: Optional[float] = None,
    apply_tax_to_gains: bool = False
) -> dict:
    """
    Berechnet das benötigte Anfangskapital für eine gewünschte monatliche Entnahme.

    Args:
        monthly_withdrawal: Gewünschte monatliche Entnahme
        expected_annual_return: Erwartete jährliche Rendite
        annual_volatility: Jährliche Volatilität
        years: Entnahmezeitraum
        target_success_rate: Gewünschte Erfolgsrate
        inflation_rate: Jährliche Inflation
        n_simulations: Anzahl Simulationen
        tax_rate: Steuersatz (z.B. 0.275 für österreichische KESt)
        apply_tax_to_gains: Steuer auf realisierte Gewinne bei Entnahmen?

    Returns:
        Dictionary mit Kapitalanforderungen
    """
    simulator = WithdrawalSimulator(n_simulations=n_simulations)

    # Binäre Suche für benötigtes Kapital
    # Startschätzung: 25x jährliche Entnahme (4% Regel invertiert)
    annual_withdrawal = monthly_withdrawal * 12
    low = annual_withdrawal * 10
    high = annual_withdrawal * 50

    best_capital = high
    best_results = None

    for _ in range(20):
        mid = (low + high) / 2

        results = simulator.simulate(
            initial_value=mid,
            monthly_withdrawal=monthly_withdrawal,
            expected_annual_return=expected_annual_return,
            annual_volatility=annual_volatility,
            years=years,
            inflation_rate=inflation_rate,
            adjust_for_inflation=True,
            tax_rate=tax_rate,
            apply_tax_to_gains=apply_tax_to_gains
        )

        if results.success_rate >= target_success_rate:
            best_capital = mid
            best_results = results
            high = mid
        else:
            low = mid

        if high - low < 1000:  # Genauigkeit von 1000€
            break

    implied_swr = annual_withdrawal / best_capital

    return {
        'required_capital': best_capital,
        'monthly_withdrawal': monthly_withdrawal,
        'annual_withdrawal': annual_withdrawal,
        'implied_withdrawal_rate': implied_swr,
        'implied_withdrawal_rate_pct': implied_swr * 100,
        'success_rate': best_results.success_rate if best_results else 0,
        'target_success_rate': target_success_rate,
        'years': years,
        'results': best_results
    }
