"""
Plotly visualizations for Monte Carlo simulation results.
Premium theme with cohesive design system.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional

from src.simulation.monte_carlo import SimulationResults

# === CHART THEME – Light fintech theme ===
# Cohesive color palette for all charts
COLORS = {
    'bg': '#fafafa',
    'paper': '#ffffff',
    'grid': 'rgba(0,0,0,0.06)',
    'text': '#1a1a1a',
    'text_muted': '#888888',
    'accent': '#0d6e4f',
    'accent_soft': 'rgba(13,110,79,0.08)',
    'success': '#16a34a',
    'success_soft': 'rgba(22,163,74,0.08)',
    'warning': '#d97706',
    'warning_soft': 'rgba(217,119,6,0.08)',
    'danger': '#dc2626',
    'danger_soft': 'rgba(220,38,38,0.08)',
    'path': 'rgba(13,110,79,0.15)',
    'path_fail': 'rgba(220,38,38,0.12)',
    'p5': '#dc2626',
    'p25': '#d97706',
    'p50': '#0d6e4f',
    'p75': '#d97706',
    'p95': '#dc2626',
    'cone_90': 'rgba(13,110,79,0.05)',
    'cone_95': 'rgba(13,110,79,0.10)',
    'cone_99': 'rgba(13,110,79,0.18)',
    'frontier_scatter': 'Viridis',
    'heatmap': [
        [0.0, '#dc2626'], [0.25, '#f59e8b'], [0.5, '#f5f5f5'],
        [0.75, '#6bc4a0'], [1.0, '#0d6e4f']
    ],
    'pie': ['#0d6e4f', '#16a34a', '#d97706', '#dc2626', '#7c3aed',
            '#059669', '#0891b2', '#ca8a04', '#e11d48', '#9333ea',
            '#14b8a6', '#65a30d'],
    'bar_primary': '#0d6e4f',
    'bar_secondary': '#aaaaaa',
    'gold': '#ca8a04',
}

FONT_FAMILY = 'Outfit, -apple-system, BlinkMacSystemFont, sans-serif'
FONT_MONO = 'IBM Plex Mono, SF Mono, monospace'


def _base_layout(**overrides) -> dict:
    """Shared layout configuration for all charts."""
    layout = dict(
        font=dict(family=FONT_FAMILY, color=COLORS['text'], size=13),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=30, t=50, b=50),
        title_font=dict(family=FONT_FAMILY, size=16, color=COLORS['text']),
        title_x=0,
        title_xanchor='left',
        xaxis=dict(
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid'],
            title_font=dict(size=12, color=COLORS['text_muted']),
            tickfont=dict(size=11, color=COLORS['text_muted']),
        ),
        yaxis=dict(
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid'],
            title_font=dict(size=12, color=COLORS['text_muted']),
            tickfont=dict(size=11, color=COLORS['text_muted']),
        ),
        legend=dict(
            font=dict(size=11, color=COLORS['text_muted']),
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor='#ffffff',
            font_size=12,
            font_family=FONT_FAMILY,
            font_color=COLORS['text'],
            bordercolor=COLORS['accent'],
        ),
    )
    layout.update(overrides)
    return layout


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

    if results.num_simulations == 0 or results.portfolio_values.size == 0:
        fig.update_layout(**_base_layout(title='Keine Simulationsdaten verfügbar'))
        return fig

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
            line=dict(width=0.5, color=COLORS['path']),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add percentile bands
    if show_percentiles:
        percentiles = [5, 25, 50, 75, 95]
        colors = [COLORS['p5'], COLORS['p25'], COLORS['p50'], COLORS['p75'], COLORS['p95']]

        for p, color in zip(percentiles, colors):
            percentile_values = np.percentile(results.portfolio_values, p, axis=0)
            fig.add_trace(go.Scatter(
                x=days,
                y=percentile_values,
                mode='lines',
                name=f'{p}. Perzentil',
                line=dict(width=2.5 if p == 50 else 1.2, color=color, dash='solid' if p == 50 else 'dash'),
            ))

    fig.update_layout(**_base_layout(
        title='Monte Carlo Simulation – Portfolio Entwicklung',
        xaxis_title='Handelstage',
        yaxis_title='Portfolio Wert (€)',
        yaxis_tickformat=',.0f',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    ))

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

    if len(final_values) == 0:
        fig.update_layout(**_base_layout(title='Keine Daten für Verteilung verfügbar'))
        return fig

    # Histogram
    fig.add_trace(go.Histogram(
        x=final_values,
        nbinsx=50,
        name='Verteilung',
        marker_color=COLORS['accent'],
        opacity=0.55
    ))

    # Add initial value line
    fig.add_vline(
        x=initial_value,
        line_dash="solid",
        line_color=COLORS['text_muted'],
        line_width=1.5,
        annotation_text="Startwert",
        annotation_font_color=COLORS['text_muted'],
    )

    # Add mean line
    mean_val = np.mean(final_values)
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=COLORS['success'],
        line_width=1.5,
        annotation_text=f"Mittelwert: {mean_val:,.0f}€",
        annotation_font_color=COLORS['success'],
    )

    # Add VaR line if provided
    if var_level is not None:
        var_value = initial_value + var_level
        fig.add_vline(
            x=var_value,
            line_dash="dash",
            line_color=COLORS['danger'],
            line_width=1.5,
            annotation_text=f"VaR 95%: {var_value:,.0f}€",
            annotation_font_color=COLORS['danger'],
        )

    # Add CVaR line if provided
    if cvar_level is not None:
        cvar_value = initial_value + cvar_level
        fig.add_vline(
            x=cvar_value,
            line_dash="dot",
            line_color='#b91c1c',
            line_width=1.5,
            annotation_text=f"CVaR 95%: {cvar_value:,.0f}€",
            annotation_font_color='#b91c1c',
        )

    fig.update_layout(**_base_layout(
        title='Verteilung der Endwerte',
        xaxis_title='Portfolio Wert (€)',
        yaxis_title='Häufigkeit',
        xaxis_tickformat=',.0f',
        showlegend=False,
    ))

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

    cone_colors = [COLORS['cone_90'], COLORS['cone_95'], COLORS['cone_99']]

    # Add confidence bands (reverse order so widest is behind)
    for conf, color in reversed(list(zip(confidence_levels, cone_colors))):
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

    # Add median line on top
    fig.add_trace(go.Scatter(
        x=days,
        y=median,
        mode='lines',
        name='Median',
        line=dict(color=COLORS['success'], width=2.5)
    ))

    fig.update_layout(**_base_layout(
        title='Value-at-Risk Kegel',
        xaxis_title='Handelstage',
        yaxis_title='Portfolio Wert (€)',
        yaxis_tickformat=',.0f',
        hovermode='x unified',
    ))

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
    if correlation_matrix.shape != (len(tickers), len(tickers)):
        fig = go.Figure()
        fig.update_layout(**_base_layout(title='Korrelationsmatrix: Dimensionen stimmen nicht überein'))
        return fig

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=tickers,
        y=tickers,
        colorscale=COLORS['heatmap'],
        zmid=0,
        text=np.round(correlation_matrix, 2),
        texttemplate='%{text}',
        textfont=dict(size=12, family=FONT_MONO, color=COLORS['text']),
        hoverongaps=False,
        colorbar=dict(
            tickfont=dict(color=COLORS['text_muted'], size=10),
            title=dict(text='Korrelation', font=dict(color=COLORS['text_muted'], size=11)),
        ),
    ))

    fig.update_layout(**_base_layout(
        title='Asset Korrelationsmatrix',
        xaxis_title='',
        yaxis_title='',
    ))

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
        opacity=0.6,
        marker_color=COLORS['accent']
    ))

    if historical_returns is not None:
        fig.add_trace(go.Histogram(
            x=historical_returns,
            nbinsx=50,
            name='Historisch',
            opacity=0.6,
            marker_color=COLORS['warning']
        ))

    fig.update_layout(**_base_layout(
        title='Renditeverteilung: Simulation vs. Historie',
        xaxis_title='Rendite',
        yaxis_title='Häufigkeit',
        barmode='overlay',
        xaxis_tickformat='.1%',
    ))

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
        marker_color=COLORS['accent'],
        marker_line_width=0,
    ))

    if benchmark_metrics:
        bench_values = [benchmark_metrics.get(m, 0) for m in metric_names]
        fig.add_trace(go.Bar(
            name='Benchmark',
            x=metric_names,
            y=bench_values,
            marker_color=COLORS['bar_secondary'],
            marker_line_width=0,
        ))

    fig.update_layout(**_base_layout(
        title='Risiko-Metriken Übersicht',
        yaxis_title='Wert',
        barmode='group',
    ))

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
        hole=0.5,
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(family=FONT_FAMILY, size=12, color=COLORS['text']),
        marker=dict(
            colors=COLORS['pie'][:len(tickers)],
            line=dict(color='#ffffff', width=2),
        ),
        pull=[0.02] * len(tickers),
    )])

    fig.update_layout(**_base_layout(
        title='Portfolio Allokation',
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        margin=dict(l=20, r=20, t=50, b=20),
    ))

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
        fillcolor=COLORS['danger_soft'],
        line=dict(color=COLORS['danger'], width=1.5),
        name='Drawdown'
    ))

    fig.update_layout(**_base_layout(
        title='Drawdown über Zeit',
        xaxis_title='Handelstage',
        yaxis_title='Drawdown (%)',
        yaxis_ticksuffix='%',
    ))

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

        bar_color = COLORS['success'] if return_pct > 0 else COLORS['danger']

        fig.add_trace(go.Bar(
            name=name,
            x=[name],
            y=[return_pct],
            marker_color=bar_color,
            marker_line_width=0,
            error_y=dict(
                type='data',
                array=[final_std / initial_value * 100],
                visible=True,
                color=COLORS['text_muted'],
                thickness=1.5,
            )
        ))

    fig.update_layout(**_base_layout(
        title='Szenario-Vergleich: Erwartete Rendite',
        yaxis_title='Rendite (%)',
        yaxis_ticksuffix='%',
        showlegend=False,
    ))

    return fig


def plot_efficient_frontier(
    frontier_result,
    current_weights: Optional[np.ndarray] = None,
    current_return: Optional[float] = None,
    current_volatility: Optional[float] = None,
    ticker_symbols: Optional[list[str]] = None
) -> go.Figure:
    """
    Plot the Efficient Frontier with optimal portfolios.

    Args:
        frontier_result: EfficientFrontierResult object
        current_weights: Current portfolio weights (optional)
        current_return: Current portfolio expected return (optional)
        current_volatility: Current portfolio volatility (optional)
        ticker_symbols: List of ticker symbols for labels

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Scatter plot of random portfolios, colored by Sharpe Ratio
    fig.add_trace(go.Scatter(
        x=frontier_result.all_portfolios_volatilities * 100,
        y=frontier_result.all_portfolios_returns * 100,
        mode='markers',
        marker=dict(
            size=5,
            color=frontier_result.all_portfolios_sharpe,
            colorscale=COLORS['frontier_scatter'],
            colorbar=dict(
                title=dict(text='Sharpe Ratio', font=dict(color=COLORS['text_muted'], size=11)),
                tickfont=dict(color=COLORS['text_muted'], size=10),
            ),
            opacity=0.45,
        ),
        name='Zufällige Portfolios',
        hovertemplate='Volatilität: %{x:.1f}%<br>Rendite: %{y:.1f}%<extra></extra>'
    ))

    # Efficient Frontier line
    fig.add_trace(go.Scatter(
        x=frontier_result.frontier_volatilities * 100,
        y=frontier_result.frontier_returns * 100,
        mode='lines',
        line=dict(color=COLORS['accent'], width=3),
        name='Efficient Frontier',
        hovertemplate='Volatilität: %{x:.1f}%<br>Rendite: %{y:.1f}%<extra></extra>'
    ))

    # Maximum Sharpe Ratio Portfolio
    max_sharpe = frontier_result.max_sharpe_portfolio
    fig.add_trace(go.Scatter(
        x=[max_sharpe.volatility * 100],
        y=[max_sharpe.expected_return * 100],
        mode='markers',
        marker=dict(size=18, color=COLORS['gold'], symbol='star', line=dict(width=2, color='#ffffff')),
        name=f'Max Sharpe ({max_sharpe.sharpe_ratio:.2f})',
        hovertemplate=f'<b>Maximale Sharpe Ratio</b><br>Volatilität: {max_sharpe.volatility*100:.1f}%<br>Rendite: {max_sharpe.expected_return*100:.1f}%<br>Sharpe: {max_sharpe.sharpe_ratio:.2f}<extra></extra>'
    ))

    # Minimum Volatility Portfolio
    min_vol = frontier_result.min_volatility_portfolio
    fig.add_trace(go.Scatter(
        x=[min_vol.volatility * 100],
        y=[min_vol.expected_return * 100],
        mode='markers',
        marker=dict(size=14, color=COLORS['accent'], symbol='diamond', line=dict(width=2, color='#ffffff')),
        name=f'Min Volatilität ({min_vol.volatility*100:.1f}%)',
        hovertemplate=f'<b>Minimale Volatilität</b><br>Volatilität: {min_vol.volatility*100:.1f}%<br>Rendite: {min_vol.expected_return*100:.1f}%<br>Sharpe: {min_vol.sharpe_ratio:.2f}<extra></extra>'
    ))

    # Current Portfolio (if provided)
    if current_volatility is not None and current_return is not None:
        fig.add_trace(go.Scatter(
            x=[current_volatility * 100],
            y=[current_return * 100],
            mode='markers',
            marker=dict(size=14, color=COLORS['danger'], symbol='circle', line=dict(width=2, color='#ffffff')),
            name='Aktuelles Portfolio',
            hovertemplate=f'<b>Aktuelles Portfolio</b><br>Volatilität: {current_volatility*100:.1f}%<br>Rendite: {current_return*100:.1f}%<extra></extra>'
        ))

    fig.update_layout(**_base_layout(
        title='Efficient Frontier – Rendite vs. Risiko',
        xaxis_title='Volatilität (Risiko) %',
        yaxis_title='Erwartete Rendite %',
        xaxis_ticksuffix='%',
        yaxis_ticksuffix='%',
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    ))

    return fig


def plot_withdrawal_simulation(
    results,
    num_paths: int = 100,
    show_percentiles: bool = True
) -> go.Figure:
    """
    Plot withdrawal simulation paths.

    Args:
        results: WithdrawalResults object
        num_paths: Number of sample paths to display
        show_percentiles: Whether to show percentile bands

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    months = np.arange(results.portfolio_paths.shape[1])
    years = months / 12

    # Plot sample paths
    sample_indices = np.random.choice(
        results.n_simulations,
        size=min(num_paths, results.n_simulations),
        replace=False
    )

    for idx in sample_indices:
        color = COLORS['path'] if results.portfolio_paths[idx, -1] > 0 else COLORS['path_fail']
        fig.add_trace(go.Scatter(
            x=years,
            y=results.portfolio_paths[idx],
            mode='lines',
            line=dict(width=0.5, color=color),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add percentile bands
    if show_percentiles:
        percentiles = [5, 25, 50, 75, 95]
        colors = [COLORS['p5'], COLORS['p25'], COLORS['p50'], COLORS['p75'], COLORS['p95']]

        for p, color in zip(percentiles, colors):
            percentile_values = np.percentile(results.portfolio_paths, p, axis=0)
            fig.add_trace(go.Scatter(
                x=years,
                y=percentile_values,
                mode='lines',
                name=f'{p}. Perzentil',
                line=dict(width=2.5 if p == 50 else 1.2, color=color, dash='solid' if p == 50 else 'dash'),
            ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['text_muted'],
                  line_width=1, annotation_text="Vermögen erschöpft",
                  annotation_font_color=COLORS['text_muted'])

    fig.update_layout(**_base_layout(
        title='Entnahme-Simulation – Vermögensentwicklung',
        xaxis_title='Jahre',
        yaxis_title='Portfolio Wert (€)',
        yaxis_tickformat=',.0f',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    ))

    return fig


def plot_depletion_histogram(
    results,
) -> go.Figure:
    """
    Plot histogram of portfolio depletion times.

    Args:
        results: WithdrawalResults object

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Get depletion times in years (only for failed simulations)
    depletion_times_years = results.depletion_times[results.depletion_times < np.inf] / 12

    if len(depletion_times_years) > 0:
        fig.add_trace(go.Histogram(
            x=depletion_times_years,
            nbinsx=30,
            name='Erschöpfungszeit',
            marker_color=COLORS['danger'],
            opacity=0.55,
        ))

        # Add median line if available
        if results.median_depletion_time is not None:
            median_years = results.median_depletion_time / 12
            fig.add_vline(
                x=median_years,
                line_dash="dash",
                line_color=COLORS['warning'],
                line_width=1.5,
                annotation_text=f"Median: {median_years:.1f} Jahre",
                annotation_font_color=COLORS['warning'],
            )

    fig.update_layout(**_base_layout(
        title=f'Verteilung der Erschöpfungszeiten (Misserfolgsrate: {results.failure_rate*100:.1f}%)',
        xaxis_title='Jahre bis zur Erschöpfung',
        yaxis_title='Häufigkeit',
        showlegend=False,
    ))

    return fig


def plot_success_rate_gauge(
    success_rate: float
) -> go.Figure:
    """
    Create a gauge chart showing success rate.

    Args:
        success_rate: Success rate between 0 and 1

    Returns:
        Plotly figure
    """
    bar_color = COLORS['success'] if success_rate >= 0.9 else COLORS['warning'] if success_rate >= 0.7 else COLORS['danger']

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=success_rate * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Erfolgsquote", 'font': dict(family=FONT_FAMILY, size=16, color=COLORS['text'])},
        number={'suffix': '%', 'font': dict(family=FONT_MONO, size=36, color=COLORS['text'])},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickfont': dict(color=COLORS['text_muted'], size=10),
            },
            'bar': {'color': bar_color, 'thickness': 0.8},
            'bgcolor': '#f5f5f5',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 70], 'color': COLORS['danger_soft']},
                {'range': [70, 90], 'color': COLORS['warning_soft']},
                {'range': [90, 100], 'color': COLORS['success_soft']}
            ],
            'threshold': {
                'line': {'color': COLORS['text_muted'], 'width': 2},
                'thickness': 0.75,
                'value': 95
            }
        }
    ))

    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, color=COLORS['text']),
        margin=dict(l=30, r=30, t=50, b=20),
    )

    return fig


def plot_optimal_weights(
    weights: np.ndarray,
    ticker_symbols: list[str],
    title: str = "Optimale Portfolio-Gewichtung"
) -> go.Figure:
    """
    Plot optimal portfolio weights as bar chart.

    Args:
        weights: Array of weights
        ticker_symbols: List of ticker symbols
        title: Chart title

    Returns:
        Plotly figure
    """
    # Filter out zero weights
    non_zero_mask = weights > 0.001
    filtered_weights = weights[non_zero_mask]
    filtered_tickers = [t for t, m in zip(ticker_symbols, non_zero_mask) if m]

    colors = COLORS['pie'][:len(filtered_tickers)]

    fig = go.Figure(data=[
        go.Bar(
            x=filtered_tickers,
            y=filtered_weights * 100,
            marker_color=colors,
            marker_line_width=0,
            text=[f'{w*100:.1f}%' for w in filtered_weights],
            textposition='auto',
            textfont=dict(family=FONT_MONO, size=12, color=COLORS['text']),
        )
    ])

    fig.update_layout(**_base_layout(
        title=title,
        xaxis_title='Asset',
        yaxis_title='Gewichtung (%)',
        yaxis_ticksuffix='%',
        showlegend=False,
    ))

    return fig
