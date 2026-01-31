"""
Efficient Frontier und Portfolio-Optimierung

Dieses Modul implementiert die Markowitz Mean-Variance-Optimierung
zur Berechnung der Efficient Frontier und des optimalen Portfolios.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimizedPortfolio:
    """Ergebnis einer Portfolio-Optimierung."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    ticker_symbols: list[str]

    def get_weights_dict(self) -> dict[str, float]:
        """Gibt Gewichtungen als Dictionary zurück."""
        return {ticker: weight for ticker, weight in zip(self.ticker_symbols, self.weights)}


@dataclass
class EfficientFrontierResult:
    """Ergebnis der Efficient Frontier Berechnung."""
    frontier_returns: np.ndarray
    frontier_volatilities: np.ndarray
    frontier_weights: list[np.ndarray]
    max_sharpe_portfolio: OptimizedPortfolio
    min_volatility_portfolio: OptimizedPortfolio
    all_portfolios_returns: np.ndarray  # Für Scatter-Plot
    all_portfolios_volatilities: np.ndarray
    all_portfolios_sharpe: np.ndarray


class PortfolioOptimizer:
    """
    Portfolio-Optimierer basierend auf der Modernen Portfolio-Theorie (MPT).

    Berechnet die Efficient Frontier und findet optimale Portfolios
    für verschiedene Risiko-/Renditeziele.
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        ticker_symbols: list[str],
        risk_free_rate: float = 0.04
    ):
        """
        Args:
            expected_returns: Erwartete Renditen der Assets (annualisiert)
            covariance_matrix: Kovarianzmatrix der Renditen (annualisiert)
            ticker_symbols: Liste der Ticker-Symbole
            risk_free_rate: Risikofreier Zinssatz (Default: 4%)
        """
        self.expected_returns = np.array(expected_returns)
        self.cov_matrix = np.array(covariance_matrix)
        self.ticker_symbols = ticker_symbols
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)

    def _portfolio_return(self, weights: np.ndarray) -> float:
        """Berechnet die erwartete Portfolio-Rendite."""
        return np.dot(weights, self.expected_returns)

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Berechnet die Portfolio-Volatilität (Standardabweichung)."""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def _portfolio_sharpe(self, weights: np.ndarray) -> float:
        """Berechnet die Sharpe Ratio des Portfolios."""
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        if vol == 0:
            return 0
        return (ret - self.risk_free_rate) / vol

    def _negative_sharpe(self, weights: np.ndarray) -> float:
        """Negatives Sharpe für Minimierung."""
        return -self._portfolio_sharpe(weights)

    def optimize_max_sharpe(
        self,
        weight_bounds: tuple[float, float] = (0, 1),
        allow_short: bool = False
    ) -> OptimizedPortfolio:
        """
        Findet das Portfolio mit der maximalen Sharpe Ratio.

        Args:
            weight_bounds: Min/Max Gewichtung pro Asset
            allow_short: Erlaubt Short-Positionen wenn True

        Returns:
            OptimizedPortfolio mit maximaler Sharpe Ratio
        """
        bounds = ((-1, 1) if allow_short else weight_bounds,) * self.n_assets
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Gewichte summieren zu 1
        ]

        # Mehrere Startpunkte versuchen für robustere Optimierung
        best_result = None
        best_sharpe = -np.inf

        for _ in range(10):
            init_weights = np.random.dirichlet(np.ones(self.n_assets))

            result = minimize(
                self._negative_sharpe,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            if result.success and -result.fun > best_sharpe:
                best_sharpe = -result.fun
                best_result = result

        if best_result is None:
            # Fallback: Gleichgewichtetes Portfolio
            weights = np.ones(self.n_assets) / self.n_assets
        else:
            weights = best_result.x

        # Kleine Gewichte auf 0 setzen
        weights[np.abs(weights) < 1e-4] = 0
        weights = weights / np.sum(weights)  # Renormalisieren

        return OptimizedPortfolio(
            weights=weights,
            expected_return=self._portfolio_return(weights),
            volatility=self._portfolio_volatility(weights),
            sharpe_ratio=self._portfolio_sharpe(weights),
            ticker_symbols=self.ticker_symbols
        )

    def optimize_min_volatility(
        self,
        weight_bounds: tuple[float, float] = (0, 1),
        allow_short: bool = False
    ) -> OptimizedPortfolio:
        """
        Findet das Portfolio mit minimaler Volatilität.

        Args:
            weight_bounds: Min/Max Gewichtung pro Asset
            allow_short: Erlaubt Short-Positionen wenn True

        Returns:
            OptimizedPortfolio mit minimaler Volatilität
        """
        bounds = ((-1, 1) if allow_short else weight_bounds,) * self.n_assets
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        init_weights = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            self._portfolio_volatility,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        weights = result.x if result.success else init_weights
        weights[np.abs(weights) < 1e-4] = 0
        weights = weights / np.sum(weights)

        return OptimizedPortfolio(
            weights=weights,
            expected_return=self._portfolio_return(weights),
            volatility=self._portfolio_volatility(weights),
            sharpe_ratio=self._portfolio_sharpe(weights),
            ticker_symbols=self.ticker_symbols
        )

    def optimize_target_return(
        self,
        target_return: float,
        weight_bounds: tuple[float, float] = (0, 1),
        allow_short: bool = False
    ) -> Optional[OptimizedPortfolio]:
        """
        Findet das Portfolio mit minimaler Volatilität für eine Zielrendite.

        Args:
            target_return: Gewünschte Rendite
            weight_bounds: Min/Max Gewichtung pro Asset
            allow_short: Erlaubt Short-Positionen wenn True

        Returns:
            OptimizedPortfolio oder None wenn Zielrendite nicht erreichbar
        """
        bounds = ((-1, 1) if allow_short else weight_bounds,) * self.n_assets
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: self._portfolio_return(w) - target_return}
        ]

        init_weights = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            self._portfolio_volatility,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            return None

        weights = result.x
        weights[np.abs(weights) < 1e-4] = 0
        weights = weights / np.sum(weights)

        return OptimizedPortfolio(
            weights=weights,
            expected_return=self._portfolio_return(weights),
            volatility=self._portfolio_volatility(weights),
            sharpe_ratio=self._portfolio_sharpe(weights),
            ticker_symbols=self.ticker_symbols
        )

    def calculate_efficient_frontier(
        self,
        n_points: int = 50,
        weight_bounds: tuple[float, float] = (0, 1),
        allow_short: bool = False
    ) -> EfficientFrontierResult:
        """
        Berechnet die komplette Efficient Frontier.

        Args:
            n_points: Anzahl Punkte auf der Frontier
            weight_bounds: Min/Max Gewichtung pro Asset
            allow_short: Erlaubt Short-Positionen wenn True

        Returns:
            EfficientFrontierResult mit allen Frontier-Daten
        """
        # Minimum und Maximum Rendite finden
        min_vol_portfolio = self.optimize_min_volatility(weight_bounds, allow_short)
        max_sharpe_portfolio = self.optimize_max_sharpe(weight_bounds, allow_short)

        min_return = min_vol_portfolio.expected_return
        max_return = max(self.expected_returns)

        # Zielrenditen für die Frontier
        target_returns = np.linspace(min_return, max_return, n_points)

        frontier_returns = []
        frontier_volatilities = []
        frontier_weights = []

        for target in target_returns:
            portfolio = self.optimize_target_return(target, weight_bounds, allow_short)
            if portfolio is not None:
                frontier_returns.append(portfolio.expected_return)
                frontier_volatilities.append(portfolio.volatility)
                frontier_weights.append(portfolio.weights)

        # Zufällige Portfolios für Scatter-Plot generieren
        n_random = 1000
        random_returns = []
        random_volatilities = []
        random_sharpe = []

        for _ in range(n_random):
            weights = np.random.dirichlet(np.ones(self.n_assets))
            ret = self._portfolio_return(weights)
            vol = self._portfolio_volatility(weights)
            sharpe = self._portfolio_sharpe(weights)
            random_returns.append(ret)
            random_volatilities.append(vol)
            random_sharpe.append(sharpe)

        return EfficientFrontierResult(
            frontier_returns=np.array(frontier_returns),
            frontier_volatilities=np.array(frontier_volatilities),
            frontier_weights=frontier_weights,
            max_sharpe_portfolio=max_sharpe_portfolio,
            min_volatility_portfolio=min_vol_portfolio,
            all_portfolios_returns=np.array(random_returns),
            all_portfolios_volatilities=np.array(random_volatilities),
            all_portfolios_sharpe=np.array(random_sharpe)
        )


def optimize_portfolio_from_data(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    trading_days: int = 252
) -> EfficientFrontierResult:
    """
    Hilfsfunktion zur Optimierung basierend auf historischen Renditen.

    Args:
        returns_df: DataFrame mit täglichen Renditen (Spalten = Assets)
        risk_free_rate: Risikofreier Zinssatz
        trading_days: Handelstage pro Jahr

    Returns:
        EfficientFrontierResult
    """
    # Annualisierte Statistiken berechnen
    expected_returns = returns_df.mean() * trading_days
    cov_matrix = returns_df.cov() * trading_days

    optimizer = PortfolioOptimizer(
        expected_returns=expected_returns.values,
        covariance_matrix=cov_matrix.values,
        ticker_symbols=list(returns_df.columns),
        risk_free_rate=risk_free_rate
    )

    return optimizer.calculate_efficient_frontier()
