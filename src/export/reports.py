"""
Export functionality for simulation results.
"""
import io
from datetime import datetime
from typing import TYPE_CHECKING, Optional
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from src.simulation.monte_carlo import SimulationResults
    from src.portfolio.portfolio import Portfolio


def create_excel_report(
    portfolio: "Portfolio",
    results: "SimulationResults",
    initial_value: float,
    var_value: float,
    cvar_value: float,
    confidence_level: float,
    metrics: dict
) -> bytes:
    """
    Create an Excel report with multiple sheets.

    Returns:
        Excel file as bytes
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Portfolio Summary
        portfolio_df = portfolio.to_dataframe()
        portfolio_df.to_excel(writer, sheet_name='Portfolio', index=False)

        # Sheet 2: Simulation Results Summary
        summary_data = {
            'Metrik': [
                'Anfangskapital',
                'Erwarteter Endwert',
                'Median Endwert',
                'Standardabweichung',
                'Minimum',
                'Maximum',
                f'VaR ({confidence_level:.0%})',
                f'CVaR ({confidence_level:.0%})',
                'Gewinnwahrscheinlichkeit',
                'Verlustwahrscheinlichkeit',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Max Drawdown',
                'Volatilität (ann.)'
            ],
            'Wert': [
                initial_value,
                results.mean_final_value,
                results.median_final_value,
                results.std_final_value,
                results.min_value,
                results.max_value,
                var_value,
                cvar_value,
                np.mean(results.final_values > initial_value),
                np.mean(results.final_values < initial_value),
                metrics.get('sharpe', 0),
                metrics.get('sortino', 0),
                metrics.get('max_drawdown', 0),
                metrics.get('volatility', 0)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Zusammenfassung', index=False)

        # Sheet 3: Percentile Analysis
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        perc_data = {
            'Perzentil': [f'{p}%' for p in percentiles],
            'Endwert': [results.percentile(p) for p in percentiles],
            'Rendite': [(results.percentile(p) / initial_value - 1) for p in percentiles]
        }
        perc_df = pd.DataFrame(perc_data)
        perc_df.to_excel(writer, sheet_name='Perzentile', index=False)

        # Sheet 4: Correlation Matrix
        corr_matrix = portfolio.get_correlation_matrix()
        corr_df = pd.DataFrame(
            corr_matrix,
            index=portfolio.tickers,
            columns=portfolio.tickers
        )
        corr_df.to_excel(writer, sheet_name='Korrelationen')

        # Sheet 5: Distribution of Final Values
        final_values_df = pd.DataFrame({
            'Simulation': range(1, len(results.final_values) + 1),
            'Endwert': results.final_values,
            'Rendite': results.returns
        })
        # Only include first 1000 for file size
        final_values_df.head(1000).to_excel(
            writer, sheet_name='Simulationsdaten', index=False
        )

    output.seek(0)
    return output.getvalue()


def create_csv_report(
    portfolio: "Portfolio",
    results: "SimulationResults",
    initial_value: float
) -> str:
    """
    Create a simple CSV summary report.

    Returns:
        CSV content as string
    """
    lines = [
        "Monte Carlo Portfolio Simulation Report",
        f"Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=== Portfolio ===",
        "Ticker,Gewichtung,Preis,Ann. Rendite,Ann. Volatilität"
    ]

    for asset in portfolio.assets:
        weight = portfolio.weights[portfolio.tickers.index(asset.ticker)]
        lines.append(
            f"{asset.ticker},{weight:.2%},{asset.current_price:.2f},"
            f"{asset.annualized_return():.2%},{asset.annualized_volatility():.2%}"
        )

    lines.extend([
        "",
        "=== Simulationsergebnisse ===",
        f"Anfangskapital,{initial_value:.2f}",
        f"Erwarteter Endwert,{results.mean_final_value:.2f}",
        f"Median Endwert,{results.median_final_value:.2f}",
        f"Std. Abweichung,{results.std_final_value:.2f}",
        f"Minimum,{results.min_value:.2f}",
        f"Maximum,{results.max_value:.2f}",
        f"Mittlere Rendite,{results.mean_return:.2%}",
        "",
        "=== Perzentile ===",
        "Perzentil,Endwert,Rendite"
    ])

    for p in [5, 10, 25, 50, 75, 90, 95]:
        val = results.percentile(p)
        ret = val / initial_value - 1
        lines.append(f"{p}%,{val:.2f},{ret:.2%}")

    return "\n".join(lines)


def format_currency(value: float, currency: str = "€") -> str:
    """Format a number as currency."""
    return f"{currency}{value:,.2f}"


def format_percentage(value: float) -> str:
    """Format a number as percentage."""
    return f"{value:.2%}"
