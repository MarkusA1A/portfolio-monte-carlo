"""
Monte Carlo Portfolio Simulation - Streamlit Web Application
Extended version with Benchmark, Savings Plan, Scenarios, and Export
"""
import streamlit as st
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation.monte_carlo import MonteCarloSimulator
from src.simulation.savings_plan import SavingsPlanSimulator
from src.simulation.scenarios import SCENARIOS, ScenarioType, Scenario
from src.data.market_data import MarketDataProvider
from src.portfolio.rebalancing import (
    NoRebalancing,
    PeriodicRebalancing,
    ThresholdRebalancing,
    RebalanceFrequency
)
from src.risk.var import calculate_var, calculate_cvar
from src.risk.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_volatility,
    calculate_beta,
    calculate_alpha
)
from src.visualization.charts import (
    plot_simulation_paths,
    plot_distribution,
    plot_var_cone,
    plot_correlation_heatmap,
    plot_portfolio_weights,
    plot_drawdown
)
from src.export.reports import create_excel_report, create_csv_report

# Page configuration - responsive layout
st.set_page_config(
    page_title="Portfolio Monte Carlo Simulation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive design
st.markdown("""
<style>
    /* Responsive metrics */
    [data-testid="stMetricValue"] {
        font-size: clamp(1rem, 2.5vw, 1.5rem);
    }
    [data-testid="stMetricLabel"] {
        font-size: clamp(0.7rem, 1.5vw, 0.9rem);
    }
    /* Better mobile sidebar */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            min-width: 100%;
        }
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: wrap;
        }
    }
    /* Responsive charts */
    .js-plotly-plot {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Monte Carlo Portfolio Simulation")

# Initialize session state
for key in ['portfolio', 'results', 'loaded_config', 'benchmark_data',
            'savings_results', 'scenario_results']:
    if key not in st.session_state:
        st.session_state[key] = None


def create_portfolio_config(
    tickers: list[str],
    weights: dict[str, float],
    initial_value: float,
    num_simulations: int,
    time_horizon_years: int,
    data_period: str,
    rebalancing_option: str,
    confidence_level: float,
    risk_free_rate: float,
    name: str = ""
) -> dict:
    """Create a portfolio configuration dictionary."""
    return {
        "name": name or f"Portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "created_at": datetime.now().isoformat(),
        "portfolio": {
            "tickers": tickers,
            "weights": {t: weights[t] for t in tickers},
            "initial_value": initial_value
        },
        "simulation": {
            "num_simulations": num_simulations,
            "time_horizon_years": time_horizon_years,
            "data_period": data_period
        },
        "rebalancing": rebalancing_option,
        "risk": {
            "confidence_level": confidence_level,
            "risk_free_rate": risk_free_rate
        }
    }


def load_portfolio_config(config: dict) -> dict:
    """Parse and validate a portfolio configuration."""
    try:
        return {
            "tickers": config["portfolio"]["tickers"],
            "weights": config["portfolio"]["weights"],
            "initial_value": config["portfolio"]["initial_value"],
            "num_simulations": config["simulation"]["num_simulations"],
            "time_horizon_years": config["simulation"]["time_horizon_years"],
            "data_period": config["simulation"]["data_period"],
            "rebalancing_option": config["rebalancing"],
            "confidence_level": config["risk"]["confidence_level"],
            "risk_free_rate": config["risk"]["risk_free_rate"],
            "name": config.get("name", "Unnamed Portfolio")
        }
    except KeyError as e:
        raise ValueError(f"Ungültiges Portfolio-Format: Feld {e} fehlt")


def get_rebalancing_strategy(option: str):
    strategies = {
        "Kein Rebalancing (Buy & Hold)": NoRebalancing(),
        "Monatlich": PeriodicRebalancing(RebalanceFrequency.MONTHLY),
        "Quartalsweise": PeriodicRebalancing(RebalanceFrequency.QUARTERLY),
        "Jährlich": PeriodicRebalancing(RebalanceFrequency.ANNUALLY),
        "Threshold (5%)": ThresholdRebalancing(0.05),
        "Threshold (10%)": ThresholdRebalancing(0.10),
    }
    return strategies.get(option, NoRebalancing())


# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Konfiguration")

    # Portfolio Load/Save Section
    with st.expander("💾 Portfolio Laden/Speichern", expanded=False):
        uploaded_file = st.file_uploader(
            "Portfolio laden",
            type=["json"],
            help="Laden Sie eine gespeicherte Portfolio-Konfiguration"
        )

        if uploaded_file is not None:
            try:
                config_data = json.load(uploaded_file)
                st.session_state.loaded_config = load_portfolio_config(config_data)
                st.success(f"'{st.session_state.loaded_config['name']}' geladen!")
            except Exception as e:
                st.error(f"Fehler: {e}")
                st.session_state.loaded_config = None

    loaded = st.session_state.loaded_config

    # Portfolio Settings
    st.subheader("Portfolio")

    default_tickers = ", ".join(loaded["tickers"]) if loaded else "AAPL, MSFT, GOOGL, AMZN"
    tickers_input = st.text_input(
        "Ticker-Symbole (kommasepariert)",
        value=default_tickers,
        help="z.B. AAPL, MSFT, GOOGL"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    default_initial = loaded["initial_value"] if loaded else 100000
    initial_value = st.number_input(
        "Anfangskapital (€)",
        min_value=1000,
        max_value=100000000,
        value=default_initial,
        step=10000
    )

    # Benchmark selection
    benchmark_ticker = st.selectbox(
        "Benchmark",
        options=["SPY", "QQQ", "^GDAXI", "^STOXX50E", "VTI"],
        index=0,
        help="Index zum Vergleich"
    )

    # Weight configuration
    st.subheader("Gewichtungen")

    weights = {}
    remaining = 100.0

    for i, ticker in enumerate(tickers):
        if loaded and ticker in loaded["weights"]:
            default_weight = loaded["weights"][ticker] * 100
        else:
            default_weight = 100.0 / len(tickers) if tickers else 25.0

        if i == len(tickers) - 1:
            weight = remaining
            st.text(f"{ticker}: {weight:.1f}%")
        else:
            max_weight = min(remaining, 100.0)
            default = min(default_weight, remaining)
            weight = st.slider(
                f"{ticker}",
                min_value=0.0,
                max_value=max_weight,
                value=default,
                step=1.0,
                key=f"weight_{ticker}"
            )
            remaining -= weight
        weights[ticker] = weight / 100.0

    # Simulation Settings
    st.subheader("Simulation")

    sim_options = [1000, 5000, 10000, 25000, 50000]
    default_sim = loaded["num_simulations"] if loaded and loaded["num_simulations"] in sim_options else 10000
    num_simulations = st.select_slider(
        "Anzahl Simulationen",
        options=sim_options,
        value=default_sim
    )

    default_horizon = loaded["time_horizon_years"] if loaded else 1
    time_horizon_years = st.slider(
        "Zeithorizont (Jahre)",
        min_value=1,
        max_value=10,
        value=default_horizon
    )
    time_horizon_days = time_horizon_years * 252

    period_options = ["1y", "2y", "5y", "10y"]
    default_period_idx = period_options.index(loaded["data_period"]) if loaded and loaded["data_period"] in period_options else 2
    data_period = st.selectbox(
        "Historische Daten",
        options=period_options,
        index=default_period_idx
    )

    # Rebalancing Settings
    st.subheader("Rebalancing")

    rebalancing_options = [
        "Kein Rebalancing (Buy & Hold)",
        "Monatlich",
        "Quartalsweise",
        "Jährlich",
        "Threshold (5%)",
        "Threshold (10%)"
    ]
    default_rebal_idx = rebalancing_options.index(loaded["rebalancing_option"]) if loaded and loaded["rebalancing_option"] in rebalancing_options else 0
    rebalancing_option = st.selectbox(
        "Strategie",
        options=rebalancing_options,
        index=default_rebal_idx
    )

    # Risk Settings
    st.subheader("Risiko")

    default_conf = loaded["confidence_level"] if loaded else 0.95
    confidence_level = st.slider(
        "Konfidenzlevel VaR/CVaR",
        min_value=0.90,
        max_value=0.99,
        value=default_conf,
        step=0.01
    )

    default_rf = loaded["risk_free_rate"] * 100 if loaded else 4.0
    risk_free_rate = st.number_input(
        "Risikofreier Zinssatz (%)",
        min_value=0.0,
        max_value=10.0,
        value=default_rf,
        step=0.1
    ) / 100

    # Sparplan Settings
    st.subheader("💰 Sparplan")
    enable_savings = st.checkbox("Sparplan aktivieren", value=False)
    monthly_contribution = st.number_input(
        "Monatliche Einzahlung (€)",
        min_value=0,
        max_value=100000,
        value=500,
        step=100,
        disabled=not enable_savings
    )

    # Run button
    st.markdown("---")
    run_simulation = st.button("🚀 Simulation starten", type="primary", use_container_width=True)

    # Save Portfolio Section
    with st.expander("💾 Portfolio speichern", expanded=False):
        portfolio_name = st.text_input(
            "Portfolio-Name",
            value=loaded["name"] if loaded else "",
            placeholder="Mein Portfolio"
        )

        current_config = create_portfolio_config(
            tickers=tickers,
            weights=weights,
            initial_value=initial_value,
            num_simulations=num_simulations,
            time_horizon_years=time_horizon_years,
            data_period=data_period,
            rebalancing_option=rebalancing_option,
            confidence_level=confidence_level,
            risk_free_rate=risk_free_rate,
            name=portfolio_name
        )

        config_json = json.dumps(current_config, indent=2, ensure_ascii=False)
        filename = f"{portfolio_name or 'portfolio'}_{datetime.now().strftime('%Y%m%d')}.json"

        st.download_button(
            label="📥 Portfolio herunterladen",
            data=config_json,
            file_name=filename,
            mime="application/json",
            use_container_width=True
        )

# Main content
if run_simulation:
    if len(tickers) < 1:
        st.error("Bitte mindestens 1 Ticker eingeben.")
    elif abs(sum(weights.values()) - 1.0) > 0.01:
        st.error(f"Gewichtungen müssen 100% ergeben. Aktuell: {sum(weights.values())*100:.1f}%")
    else:
        progress = st.progress(0, text="Lade Marktdaten...")

        try:
            provider = MarketDataProvider(period=data_period)
            weight_list = [weights[t] for t in tickers]
            portfolio = provider.create_portfolio(
                tickers=tickers,
                weights=weight_list,
                initial_value=initial_value,
                period=data_period
            )
            st.session_state.portfolio = portfolio

            # Load benchmark data
            progress.progress(20, text="Lade Benchmark-Daten...")
            benchmark_stats = provider.calculate_statistics([benchmark_ticker], data_period)
            st.session_state.benchmark_data = benchmark_stats.get(benchmark_ticker)

        except Exception as e:
            st.error(f"Fehler beim Laden der Daten: {e}")
            st.stop()

        progress.progress(40, text=f"Führe {num_simulations:,} Simulationen durch...")
        simulator = MonteCarloSimulator(
            num_simulations=num_simulations,
            time_horizon=time_horizon_days
        )
        strategy = get_rebalancing_strategy(rebalancing_option)
        results = simulator.run_simulation(portfolio, strategy)
        st.session_state.results = results

        # Savings plan simulation
        if enable_savings and monthly_contribution > 0:
            progress.progress(60, text="Simuliere Sparplan...")
            savings_sim = SavingsPlanSimulator(
                num_simulations=num_simulations,
                time_horizon_years=time_horizon_years
            )
            savings_results = savings_sim.run_simulation(
                portfolio,
                monthly_contribution,
                initial_value
            )
            st.session_state.savings_results = savings_results

        # Scenario analysis
        progress.progress(80, text="Führe Szenario-Analyse durch...")
        scenario_results = {}
        for scenario_type, scenario in SCENARIOS.items():
            modified_portfolio = portfolio.copy()
            modified_portfolio.adjust_statistics(
                scenario.return_adjustment,
                scenario.volatility_multiplier
            )
            scenario_sim = MonteCarloSimulator(
                num_simulations=min(num_simulations, 5000),
                time_horizon=time_horizon_days
            )
            scenario_results[scenario.name] = scenario_sim.run_simulation(modified_portfolio)
        st.session_state.scenario_results = scenario_results

        progress.progress(100, text="Fertig!")
        st.success("Simulation abgeschlossen!")

# Display results with tabs
if st.session_state.results is not None and st.session_state.portfolio is not None:
    results = st.session_state.results
    portfolio = st.session_state.portfolio

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Übersicht",
        "📈 Benchmark",
        "💰 Sparplan",
        "🎭 Szenarien",
        "📥 Export"
    ])

    # Calculate common metrics
    var_value = calculate_var(results.returns, confidence_level, initial_value)
    cvar_value = calculate_cvar(results.returns, confidence_level, initial_value)
    median_path = np.median(results.portfolio_values, axis=0)
    daily_returns = np.diff(median_path) / median_path[:-1]
    sharpe = calculate_sharpe_ratio(daily_returns, risk_free_rate)
    sortino = calculate_sortino_ratio(daily_returns, risk_free_rate)
    max_dd, _, _ = calculate_max_drawdown(median_path)
    vol = calculate_volatility(daily_returns)

    # TAB 1: Overview
    with tab1:
        st.header("Portfolio Übersicht")

        # Responsive columns for portfolio info
        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(
                portfolio.to_dataframe().style.format({
                    'Weight': '{:.1%}',
                    'Price': '€{:,.2f}',
                    'Ann. Return': '{:.1%}',
                    'Ann. Volatility': '{:.1%}'
                }),
                use_container_width=True
            )

        with col2:
            fig_weights = plot_portfolio_weights(portfolio.tickers, portfolio.weights)
            st.plotly_chart(fig_weights, use_container_width=True)

        st.markdown("---")

        # Key Metrics - responsive grid
        st.subheader("Simulationsergebnisse")

        # First row
        cols = st.columns(4)
        with cols[0]:
            st.metric("Erwarteter Endwert", f"€{results.mean_final_value:,.0f}", f"{results.mean_return*100:+.1f}%")
        with cols[1]:
            st.metric("Median Endwert", f"€{results.median_final_value:,.0f}", f"{(results.median_final_value/initial_value - 1)*100:+.1f}%")
        with cols[2]:
            st.metric(f"VaR ({confidence_level:.0%})", f"€{var_value:,.0f}")
        with cols[3]:
            st.metric(f"CVaR ({confidence_level:.0%})", f"€{cvar_value:,.0f}")

        # Second row
        cols = st.columns(4)
        with cols[0]:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        with cols[1]:
            st.metric("Sortino Ratio", f"{sortino:.2f}")
        with cols[2]:
            st.metric("Max Drawdown", f"{max_dd*100:.1f}%")
        with cols[3]:
            st.metric("Ann. Volatilität", f"{vol*100:.1f}%")

        # Charts
        st.markdown("---")
        st.subheader("Simulationspfade")
        fig_paths = plot_simulation_paths(results, num_paths=100, initial_value=initial_value)
        st.plotly_chart(fig_paths, use_container_width=True)

        # Two charts side by side (responsive)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Verteilung der Endwerte")
            fig_dist = plot_distribution(results.final_values, initial_value, var_level=var_value, cvar_level=cvar_value)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            st.subheader("VaR Kegel")
            fig_var = plot_var_cone(results)
            st.plotly_chart(fig_var, use_container_width=True)

        # Correlation Matrix
        st.subheader("Korrelationsmatrix")
        corr_matrix = portfolio.get_correlation_matrix()
        fig_corr = plot_correlation_heatmap(corr_matrix, portfolio.tickers)
        st.plotly_chart(fig_corr, use_container_width=True)

    # TAB 2: Benchmark Comparison
    with tab2:
        st.header("📈 Benchmark-Vergleich")

        benchmark_data = st.session_state.benchmark_data

        if benchmark_data:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Portfolio")
                st.metric("Ann. Rendite", f"{portfolio.annualized_expected_return()*100:.1f}%")
                st.metric("Ann. Volatilität", f"{portfolio.annualized_expected_volatility()*100:.1f}%")
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")

            with col2:
                st.subheader(f"Benchmark ({benchmark_ticker})")
                st.metric("Ann. Rendite", f"{benchmark_data.annualized_return*100:.1f}%")
                st.metric("Ann. Volatilität", f"{benchmark_data.annualized_volatility*100:.1f}%")
                bench_sharpe = (benchmark_data.annualized_return - risk_free_rate) / benchmark_data.annualized_volatility
                st.metric("Sharpe Ratio", f"{bench_sharpe:.2f}")

            # Calculate Beta and Alpha
            if benchmark_data.historical_returns is not None:
                portfolio_hist = portfolio._get_historical_returns_matrix()
                if portfolio_hist is not None:
                    portfolio_returns = np.sum(portfolio_hist * portfolio.weights, axis=1)
                    bench_returns = benchmark_data.historical_returns.values[-len(portfolio_returns):]

                    if len(bench_returns) == len(portfolio_returns):
                        beta = calculate_beta(portfolio_returns, bench_returns)
                        alpha = calculate_alpha(portfolio_returns, bench_returns, risk_free_rate)

                        st.markdown("---")
                        st.subheader("Relative Metriken")
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Beta", f"{beta:.2f}", help="Sensitivität zum Markt")
                        with cols[1]:
                            st.metric("Alpha (ann.)", f"{alpha*100:.2f}%", help="Überrendite nach CAPM")
                        with cols[2]:
                            tracking_error = np.std(portfolio_returns - bench_returns) * np.sqrt(252)
                            st.metric("Tracking Error", f"{tracking_error*100:.1f}%")
        else:
            st.warning("Benchmark-Daten konnten nicht geladen werden.")

    # TAB 3: Savings Plan
    with tab3:
        st.header("💰 Sparplan-Simulation")

        savings_results = st.session_state.savings_results

        if savings_results:
            # Summary metrics
            cols = st.columns(4)
            with cols[0]:
                st.metric("Gesamteinzahlung", f"€{savings_results.total_invested:,.0f}")
            with cols[1]:
                st.metric("Erwarteter Endwert", f"€{savings_results.mean_final_value:,.0f}")
            with cols[2]:
                st.metric("Erwarteter Gewinn", f"€{savings_results.mean_profit:,.0f}", f"{savings_results.mean_return*100:+.1f}%")
            with cols[3]:
                st.metric("Median Endwert", f"€{savings_results.median_final_value:,.0f}")

            # Visualization
            st.subheader("Wertentwicklung über Zeit")

            import plotly.graph_objects as go

            fig = go.Figure()

            # Sample paths
            for i in range(min(50, savings_results.num_simulations)):
                fig.add_trace(go.Scatter(
                    x=list(range(savings_results.time_horizon + 1)),
                    y=savings_results.portfolio_values[i],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(100, 149, 237, 0.3)'),
                    showlegend=False
                ))

            # Contributions line
            fig.add_trace(go.Scatter(
                x=list(range(len(savings_results.total_contributions))),
                y=savings_results.total_contributions,
                mode='lines',
                name='Einzahlungen',
                line=dict(width=2, color='red', dash='dash')
            ))

            # Median line
            median_values = np.median(savings_results.portfolio_values, axis=0)
            fig.add_trace(go.Scatter(
                x=list(range(len(median_values))),
                y=median_values,
                mode='lines',
                name='Median',
                line=dict(width=2, color='green')
            ))

            fig.update_layout(
                title=f'Sparplan: €{monthly_contribution}/Monat über {time_horizon_years} Jahre',
                xaxis_title='Monate',
                yaxis_title='Portfolio Wert (€)',
                yaxis_tickformat=',.0f'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Percentile table
            st.subheader("Perzentil-Analyse")
            percentiles = [5, 25, 50, 75, 95]
            perc_data = {
                'Perzentil': [f'{p}%' for p in percentiles],
                'Endwert': [f"€{savings_results.percentile(p):,.0f}" for p in percentiles],
                'Gewinn': [f"€{savings_results.percentile(p) - savings_results.total_invested:,.0f}" for p in percentiles]
            }
            st.dataframe(pd.DataFrame(perc_data), use_container_width=True)

        else:
            st.info("Aktivieren Sie den Sparplan in der Seitenleiste, um die Simulation zu starten.")

    # TAB 4: Scenario Analysis
    with tab4:
        st.header("🎭 Szenario-Analyse")

        scenario_results = st.session_state.scenario_results

        if scenario_results:
            # Scenario comparison table
            scenario_data = []
            for name, res in scenario_results.items():
                scenario_data.append({
                    'Szenario': name,
                    'Erwarteter Endwert': res.mean_final_value,
                    'Rendite': res.mean_return,
                    'Min': res.min_value,
                    'Max': res.max_value,
                    'Std.Abw.': res.std_final_value
                })

            df = pd.DataFrame(scenario_data)

            st.dataframe(
                df.style.format({
                    'Erwarteter Endwert': '€{:,.0f}',
                    'Rendite': '{:.1%}',
                    'Min': '€{:,.0f}',
                    'Max': '€{:,.0f}',
                    'Std.Abw.': '€{:,.0f}'
                }).background_gradient(subset=['Rendite'], cmap='RdYlGn'),
                use_container_width=True
            )

            # Bar chart comparison
            import plotly.express as px

            fig = px.bar(
                df,
                x='Szenario',
                y='Rendite',
                color='Rendite',
                color_continuous_scale='RdYlGn',
                title='Erwartete Rendite nach Szenario'
            )
            fig.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)

            # Scenario descriptions
            st.subheader("Szenario-Beschreibungen")
            for scenario_type, scenario in SCENARIOS.items():
                with st.expander(f"{scenario.name}"):
                    st.write(scenario.description)
                    st.write(f"- Rendite-Anpassung: {scenario.return_adjustment*100:+.0f}% p.a.")
                    st.write(f"- Volatilitäts-Multiplikator: {scenario.volatility_multiplier:.1f}x")

        else:
            st.info("Starten Sie eine Simulation, um Szenarien zu analysieren.")

    # TAB 5: Export
    with tab5:
        st.header("📥 Export")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Excel Report")
            st.write("Vollständiger Report mit allen Daten und Analysen.")

            metrics = {
                'sharpe': sharpe,
                'sortino': sortino,
                'max_drawdown': max_dd,
                'volatility': vol
            }

            excel_data = create_excel_report(
                portfolio=portfolio,
                results=results,
                initial_value=initial_value,
                var_value=var_value,
                cvar_value=cvar_value,
                confidence_level=confidence_level,
                metrics=metrics
            )

            st.download_button(
                label="📊 Excel herunterladen",
                data=excel_data,
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with col2:
            st.subheader("CSV Report")
            st.write("Einfacher Text-Report für schnellen Überblick.")

            csv_data = create_csv_report(
                portfolio=portfolio,
                results=results,
                initial_value=initial_value
            )

            st.download_button(
                label="📄 CSV herunterladen",
                data=csv_data,
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("---")

        # Summary statistics
        st.subheader("📋 Zusammenfassung")

        summary_data = {
            'Metrik': [
                'Anfangskapital',
                'Erwarteter Endwert',
                'Median Endwert',
                'Standardabweichung',
                'Minimum (Worst Case)',
                'Maximum (Best Case)',
                f'VaR {confidence_level:.0%}',
                f'CVaR {confidence_level:.0%}',
                'Gewinnwahrscheinlichkeit',
                'Verlustwahrscheinlichkeit'
            ],
            'Wert': [
                f'€{initial_value:,.0f}',
                f'€{results.mean_final_value:,.0f}',
                f'€{results.median_final_value:,.0f}',
                f'€{results.std_final_value:,.0f}',
                f'€{results.min_value:,.0f}',
                f'€{results.max_value:,.0f}',
                f'€{var_value:,.0f}',
                f'€{cvar_value:,.0f}',
                f'{np.mean(results.final_values > initial_value)*100:.1f}%',
                f'{np.mean(results.final_values < initial_value)*100:.1f}%'
            ]
        }

        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

else:
    # Welcome message
    st.info("👈 Konfigurieren Sie Ihr Portfolio in der Seitenleiste und klicken Sie auf 'Simulation starten'.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### So funktioniert's:

        1. **Portfolio definieren**: Ticker-Symbole eingeben
        2. **Gewichtungen festlegen**: Kapital verteilen
        3. **Simulation konfigurieren**: Zeithorizont wählen
        4. **Analyse starten**: "Simulation starten" klicken
        """)

    with col2:
        st.markdown("""
        ### Features:

        - 📊 **Monte Carlo Simulation**: Tausende Szenarien
        - 📈 **Benchmark-Vergleich**: vs. S&P 500/DAX
        - 💰 **Sparplan**: Monatliche Einzahlungen
        - 🎭 **Szenarien**: Bull/Bear/Crash-Analyse
        - 📥 **Export**: Excel & CSV Reports
        """)

# Footer
st.markdown("---")
st.caption("Monte Carlo Portfolio Simulation | Yahoo Finance Daten | Made with Streamlit")
