"""
Plotly visualizations for Monte Carlo simulation results.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional

from src.simulation.monte_carlo import SimulationResults


def plot_simulation_paths(
    results: SimulationResults,
    num_paths: int = 100,
    show_percentiles: bool = True,
    initial_value: float = 100000
) -> go.Figure:
    """
    Plot sample simulation paths with percentile bands.

    Args:
        results: Monte Carlo simulation results
        num_paths: Number of sample paths to display
        show_percentiles: Whether to show percentile bands
        initial_value: Initial portfolio value for labeling

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    time_axis = np.arange(results.portfolio_values.shape[1])
    days = time_axis

    # Plot sample paths
    sample_indices = np.random.choice(
        results.num_simulations,
        size=min(num_paths, results.num_simulations),
        replace=False
    )

    for i, idx in enumerate(sample_indices):
        fig.add_trace(go.Scatter(
            x=days,
            y=results.portfolio_values[idx],
            mode='lines',
            line=dict(width=0.5, color='rgba(100, 149, 237, 0.3)'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add percentile bands
    if show_percentiles:
        percentiles = [5, 25, 50, 75, 95]
        colors = ['rgba(255, 0, 0, 0.3)', 'rgba(255, 165, 0, 0.3)',
                  'rgba(0, 128, 0, 1)', 'rgba(255, 165, 0, 0.3)', 'rgba(255, 0, 0, 0.3)']

        for p, color in zip(percentiles, colors):
            percentile_values = np.percentile(results.portfolio_values, p, axis=0)
            fig.add_trace(go.Scatter(
                x=days,
                y=percentile_values,
                mode='lines',
                name=f'{p}. Perzentil',
                line=dict(width=2 if p == 50 else 1, color=color, dash='solid' if p == 50 else 'dash'),
            ))

    fig.update_layout(
        title='Monte Carlo Simulation - Portfolio Entwicklung',
        xaxis_title='Handelstage',
        yaxis_title='Portfolio Wert (€)',
        yaxis_tickformat=',.0f',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


def plot_distribution(
    final_values: np.ndarray,
    initial_value: float,
    var_level: Optional[float] = None,
    cvar_level: Optional[float] = None
) -> go.Figure:
    """
    Plot distribution of final portfolio values.

    Args:
        final_values: Array of final simulation values
        initial_value: Initial portfolio value
        var_level: Optional VaR value to display
        cvar_level: Optional CVaR value to display

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=final_values,
        nbinsx=50,
        name='Verteilung',
        marker_color='rgba(100, 149, 237, 0.7)',
        opacity=0.7
    ))

    # Add initial value line
    fig.add_vline(
        x=initial_value,
        line_dash="solid",
        line_color="black",
        annotation_text="Startwert"
    )

    # Add mean line
    mean_val = np.mean(final_values)
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Mittelwert: {mean_val:,.0f}€"
    )

    # Add VaR line if provided
    if var_level is not None:
        var_value = initial_value + var_level
        fig.add_vline(
            x=var_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR 95%: {var_value:,.0f}€"
        )

    # Add CVaR line if provided
    if cvar_level is not None:
        cvar_value = initial_value + cvar_level
        fig.add_vline(
            x=cvar_value,
            line_dash="dot",
            line_color="darkred",
            annotation_text=f"CVaR 95%: {cvar_value:,.0f}€"
        )

    fig.update_layout(
        title='Verteilung der Endwerte',
        xaxis_title='Portfolio Wert (€)',
        yaxis_title='Häufigkeit',
        xaxis_tickformat=',.0f',
        showlegend=False
    )

    return fig


def plot_var_cone(
    results: SimulationResults,
    confidence_levels: list[float] = [0.90, 0.95, 0.99]
) -> go.Figure:
    """
    Plot VaR cone showing risk over time.

    Args:
        results: Monte Carlo simulation results
        confidence_levels: List of confidence levels to display

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    days = np.arange(results.portfolio_values.shape[1])
    median = np.median(results.portfolio_values, axis=0)

    # Add median line
    fig.add_trace(go.Scatter(
        x=days,
        y=median,
        mode='lines',
        name='Median',
        line=dict(color='green', width=2)
    ))

    colors = ['rgba(255, 200, 200, 0.5)', 'rgba(255, 150, 150, 0.5)', 'rgba(255, 100, 100, 0.5)']

    for conf, color in zip(confidence_levels, colors):
        lower_percentile = (1 - conf) / 2 * 100
        upper_percentile = (1 + conf) / 2 * 100

        lower = np.percentile(results.portfolio_values, lower_percentile, axis=0)
        upper = np.percentile(results.portfolio_values, upper_percentile, axis=0)

        fig.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill='toself',
            fillcolor=color,
            line=dict(color='rgba(0,0,0,0)'),
            name=f'{conf:.0%} Konfidenz'
        ))

    fig.update_layout(
        title='Value-at-Risk Kegel',
        xaxis_title='Handelstage',
        yaxis_title='Portfolio Wert (€)',
        yaxis_tickformat=',.0f',
        hovermode='x unified'
    )

    return fig


def plot_correlation_heatmap(
    correlation_matrix: np.ndarray,
    tickers: list[str]
) -> go.Figure:
    """
    Plot correlation heatmap for portfolio assets.

    Args:
        correlation_matrix: Correlation matrix
        tickers: List of asset tickers

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=tickers,
        y=tickers,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False
    ))

    fig.update_layout(
        title='Asset Korrelationsmatrix',
        xaxis_title='',
        yaxis_title='',
    )

    return fig


def plot_return_distribution_comparison(
    simulation_returns: np.ndarray,
    historical_returns: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Compare simulated vs historical return distributions.

    Args:
        simulation_returns: Returns from simulation
        historical_returns: Optional historical returns for comparison

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=simulation_returns,
        nbinsx=50,
        name='Simulation',
        opacity=0.7,
        marker_color='blue'
    ))

    if historical_returns is not None:
        fig.add_trace(go.Histogram(
            x=historical_returns,
            nbinsx=50,
            name='Historisch',
            opacity=0.7,
            marker_color='orange'
        ))

    fig.update_layout(
        title='Renditeverteilung: Simulation vs. Historie',
        xaxis_title='Rendite',
        yaxis_title='Häufigkeit',
        barmode='overlay',
        xaxis_tickformat='.1%'
    )

    return fig


def plot_risk_metrics_summary(
    metrics: dict[str, float],
    benchmark_metrics: Optional[dict[str, float]] = None
) -> go.Figure:
    """
    Create a summary bar chart of risk metrics.

    Args:
        metrics: Dict of metric names to values
        benchmark_metrics: Optional benchmark metrics for comparison

    Returns:
        Plotly figure
    """
    metric_names = list(metrics.keys())
    values = list(metrics.values())

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Portfolio',
        x=metric_names,
        y=values,
        marker_color='steelblue'
    ))

    if benchmark_metrics:
        bench_values = [benchmark_metrics.get(m, 0) for m in metric_names]
        fig.add_trace(go.Bar(
            name='Benchmark',
            x=metric_names,
            y=bench_values,
            marker_color='gray'
        ))

    fig.update_layout(
        title='Risiko-Metriken Übersicht',
        yaxis_title='Wert',
        barmode='group'
    )

    return fig


def plot_portfolio_weights(
    tickers: list[str],
    weights: np.ndarray
) -> go.Figure:
    """
    Plot portfolio allocation as pie chart.

    Args:
        tickers: Asset tickers
        weights: Portfolio weights

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[go.Pie(
        labels=tickers,
        values=weights,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside'
    )])

    fig.update_layout(
        title='Portfolio Allokation',
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02)
    )

    return fig


def plot_drawdown(
    portfolio_values: np.ndarray
) -> go.Figure:
    """
    Plot drawdown over time.

    Args:
        portfolio_values: Array of portfolio values (median path)

    Returns:
        Plotly figure
    """
    # Calculate running maximum
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max

    days = np.arange(len(drawdown))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=days,
        y=drawdown * 100,
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red'),
        name='Drawdown'
    ))

    fig.update_layout(
        title='Drawdown über Zeit',
        xaxis_title='Handelstage',
        yaxis_title='Drawdown (%)',
        yaxis_ticksuffix='%'
    )

    return fig


def plot_scenario_comparison(
    scenario_results: dict[str, "SimulationResults"],
    initial_value: float
) -> go.Figure:
    """
    Compare results across different scenarios.

    Args:
        scenario_results: Dict mapping scenario name to results
        initial_value: Initial portfolio value

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for name, results in scenario_results.items():
        final_mean = results.mean_final_value
        final_std = results.std_final_value
        return_pct = (final_mean - initial_value) / initial_value * 100

        fig.add_trace(go.Bar(
            name=name,
            x=[name],
            y=[return_pct],
            error_y=dict(
                type='data',
                array=[final_std / initial_value * 100],
                visible=True
            )
        ))

    fig.update_layout(
        title='Szenario-Vergleich: Erwartete Rendite',
        yaxis_title='Rendite (%)',
        yaxis_ticksuffix='%',
        showlegend=False
    )

    return fig
