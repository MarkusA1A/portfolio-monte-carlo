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
from src.simulation.withdrawal import WithdrawalSimulator, calculate_required_capital
from src.data.market_data import MarketDataProvider
from src.portfolio.rebalancing import (
    NoRebalancing,
    PeriodicRebalancing,
    ThresholdRebalancing,
    RebalanceFrequency
)
from src.portfolio.optimization import PortfolioOptimizer, optimize_portfolio_from_data
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
    plot_drawdown,
    plot_efficient_frontier,
    plot_withdrawal_simulation,
    plot_depletion_histogram,
    plot_success_rate_gauge,
    plot_optimal_weights
)
from src.export.reports import create_excel_report, create_csv_report


def format_currency(value: float, decimals: int = 0) -> str:
    """Format number with German-style thousands separator (.) and currency symbol."""
    if decimals == 0:
        formatted = f"{value:,.0f}".replace(",", ".")
    else:
        formatted = f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€{formatted}"


def format_number(value: float, decimals: int = 0) -> str:
    """Format number with German-style thousands separator (.)."""
    if decimals == 0:
        return f"{value:,.0f}".replace(",", ".")
    else:
        return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# Page configuration - responsive layout
st.set_page_config(
    page_title="Portfolio Monte Carlo Simulation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive design (Mobile: iPhone, iPad)
st.markdown("""
<style>
    /* Responsive metrics */
    [data-testid="stMetricValue"] {
        font-size: clamp(1rem, 2.5vw, 1.5rem);
    }
    [data-testid="stMetricLabel"] {
        font-size: clamp(0.7rem, 1.5vw, 0.9rem);
    }

    /* Responsive charts */
    .js-plotly-plot {
        width: 100% !important;
    }

    /* === MOBILE STYLES (iPhone, small screens) === */
    @media (max-width: 768px) {
        /* Sidebar takes full width on mobile */
        [data-testid="stSidebar"] {
            min-width: 100%;
        }

        /* Tabs wrap on mobile */
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: wrap;
            gap: 0.25rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem;
            padding: 0.5rem 0.75rem;
        }

        /* Columns stack vertically on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }

        /* Smaller titles */
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.25rem !important; }
        h3 { font-size: 1.1rem !important; }

        /* Better touch targets for sliders */
        .stSlider [data-baseweb="slider"] {
            padding: 1rem 0;
        }

        /* Metrics in smaller cards */
        [data-testid="stMetricValue"] {
            font-size: 1.1rem;
        }

        /* Number inputs more touch-friendly */
        .stNumberInput input {
            font-size: 16px !important; /* Prevents iOS zoom on focus */
            padding: 0.75rem;
        }

        /* Better button touch targets */
        .stButton button {
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            width: 100%;
        }
    }

    /* === TABLET STYLES (iPad) === */
    @media (min-width: 769px) and (max-width: 1024px) {
        [data-testid="stSidebar"] {
            min-width: 280px;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 0.9rem;
        }
    }

    /* === TOUCH DEVICE IMPROVEMENTS === */
    @media (hover: none) and (pointer: coarse) {
        /* Larger touch targets */
        .stSlider [data-baseweb="slider"] {
            min-height: 44px;
        }

        /* Easier to tap checkboxes */
        .stCheckbox label {
            padding: 0.5rem 0;
        }

        /* Select boxes more touch-friendly */
        .stSelectbox [data-baseweb="select"] {
            min-height: 44px;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Monte Carlo Portfolio Simulation")

# Initialize session state
for key in ['portfolio', 'results', 'loaded_config', 'benchmark_data',
            'savings_results', 'scenario_results', 'efficient_frontier', 'withdrawal_results']:
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
    st.caption(f"💰 {format_currency(initial_value)}")

    # Benchmark selection
    benchmark_ticker = st.selectbox(
        "Benchmark",
        options=["SPY", "QQQ", "^GDAXI", "^STOXX50E", "VTI"],
        index=0,
        help="Index zum Vergleich"
    )

    # Weight configuration
    st.subheader("Gewichtungen")

    # Initialize session state for weights if needed
    if 'ticker_weights' not in st.session_state:
        st.session_state.ticker_weights = {}

    # Clean up old tickers from session state
    current_tickers = set(tickers)
    st.session_state.ticker_weights = {
        k: v for k, v in st.session_state.ticker_weights.items()
        if k in current_tickers
    }

    # Collect weights with automatic capping
    slider_values = {}
    for i, ticker in enumerate(tickers):
        # Calculate how much is already used by previous tickers
        used_by_previous = sum(slider_values.values())
        max_available = max(0.0, 100.0 - used_by_previous)

        # Get default value
        if ticker in st.session_state.ticker_weights:
            default_weight = st.session_state.ticker_weights[ticker]
        elif loaded and ticker in loaded["weights"]:
            default_weight = loaded["weights"][ticker] * 100
        else:
            # Distribute remaining equally among remaining tickers
            remaining_tickers = len(tickers) - i
            default_weight = max_available / remaining_tickers if remaining_tickers > 0 else 0

        # Cap default to max available
        default_weight = min(default_weight, max_available)

        # Create slider for ALL tickers
        weight = st.slider(
            f"{ticker}",
            min_value=0.0,
            max_value=100.0,
            value=default_weight,
            step=1.0,
            key=f"weight_{ticker}_{i}"  # Unique key
        )

        # Auto-cap if exceeds available
        if weight > max_available:
            weight = max_available
            st.caption(f"↳ Auf {weight:.1f}% begrenzt (max. verfügbar)")

        slider_values[ticker] = weight
        st.session_state.ticker_weights[ticker] = weight

    # Show total and status
    total = sum(slider_values.values())
    remaining = 100.0 - total

    if remaining > 0.1:
        st.info(f"📊 Summe: {total:.1f}% | Noch {remaining:.1f}% verfügbar")
    elif remaining < -0.1:
        st.warning(f"⚠️ Summe: {total:.1f}% - zu viel vergeben!")
    else:
        st.success(f"✅ Summe: 100%")

    # Convert to decimal weights (normalize to ensure sum = 1)
    if total > 0:
        weights = {t: w / total for t, w in slider_values.items()}
    else:
        weights = {t: 1.0 / len(tickers) for t in tickers} if tickers else {}

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
        progress.progress(70, text="Führe Szenario-Analyse durch...")
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

        # Efficient Frontier calculation
        progress.progress(90, text="Berechne Efficient Frontier...")
        try:
            # Get historical returns from portfolio
            returns_matrix = portfolio._get_historical_returns_matrix()
            if returns_matrix is not None:
                import pandas as pd
                returns_df = pd.DataFrame(returns_matrix, columns=portfolio.tickers)
                frontier_result = optimize_portfolio_from_data(returns_df, risk_free_rate)
                st.session_state.efficient_frontier = frontier_result
        except Exception as e:
            st.session_state.efficient_frontier = None

        progress.progress(100, text="Fertig!")
        st.success("Simulation abgeschlossen!")

# Display results with tabs
if st.session_state.results is not None and st.session_state.portfolio is not None:
    results = st.session_state.results
    portfolio = st.session_state.portfolio

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Übersicht",
        "📈 Benchmark",
        "💰 Sparplan",
        "🏦 Entnahme",
        "🎯 Efficient Frontier",
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
            st.metric("Erwarteter Endwert", format_currency(results.mean_final_value), f"{results.mean_return*100:+.1f}%")
        with cols[1]:
            st.metric("Median Endwert", format_currency(results.median_final_value), f"{(results.median_final_value/initial_value - 1)*100:+.1f}%")
        with cols[2]:
            st.metric(f"VaR ({confidence_level:.0%})", format_currency(var_value))
        with cols[3]:
            st.metric(f"CVaR ({confidence_level:.0%})", format_currency(cvar_value))

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
                bench_return = float(benchmark_data.annualized_return)
                bench_vol = float(benchmark_data.annualized_volatility)
                st.metric("Ann. Rendite", f"{bench_return*100:.1f}%")
                st.metric("Ann. Volatilität", f"{bench_vol*100:.1f}%")
                bench_sharpe = (bench_return - risk_free_rate) / bench_vol if bench_vol > 0 else 0
                st.metric("Sharpe Ratio", f"{bench_sharpe:.2f}")

            # Calculate Beta and Alpha
            if benchmark_data.historical_returns is not None:
                portfolio_hist = portfolio._get_historical_returns_matrix()
                if portfolio_hist is not None:
                    portfolio_returns = np.sum(portfolio_hist * portfolio.weights, axis=1)
                    bench_returns_raw = benchmark_data.historical_returns.values[-len(portfolio_returns):]
                    # Ensure bench_returns is 1D
                    bench_returns = bench_returns_raw.flatten() if bench_returns_raw.ndim > 1 else bench_returns_raw

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
                st.metric("Gesamteinzahlung", format_currency(savings_results.total_invested))
            with cols[1]:
                st.metric("Erwarteter Endwert", format_currency(savings_results.mean_final_value))
            with cols[2]:
                st.metric("Erwarteter Gewinn", format_currency(savings_results.mean_profit), f"{savings_results.mean_return*100:+.1f}%")
            with cols[3]:
                st.metric("Median Endwert", format_currency(savings_results.median_final_value))

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
                'Endwert': [format_currency(savings_results.percentile(p)) for p in percentiles],
                'Gewinn': [format_currency(savings_results.percentile(p) - savings_results.total_invested) for p in percentiles]
            }
            st.dataframe(pd.DataFrame(perc_data), use_container_width=True)

        else:
            st.info("Aktivieren Sie den Sparplan in der Seitenleiste, um die Simulation zu starten.")

    # TAB 4: Withdrawal Simulation (Entnahme)
    with tab4:
        st.header("🏦 Entnahme-Simulation (Ruhestandsplanung)")

        st.markdown("""
        **Wie lange reicht mein Geld?** Simulieren Sie, wie sich Ihr Vermögen entwickelt,
        wenn Sie regelmäßig Geld entnehmen (z.B. im Ruhestand).
        """)

        # Input parameters
        col1, col2, col3 = st.columns(3)

        with col1:
            withdrawal_initial = st.number_input(
                "Anfangsvermögen (€)",
                min_value=10000,
                max_value=100000000,
                value=500000,
                step=10000,
                key="withdrawal_initial"
            )

        with col2:
            monthly_withdrawal = st.number_input(
                "Monatliche Entnahme (€)",
                min_value=100,
                max_value=100000,
                value=2000,
                step=100,
                key="monthly_withdrawal"
            )

        with col3:
            withdrawal_years = st.slider(
                "Simulationszeitraum (Jahre)",
                min_value=5,
                max_value=50,
                value=30,
                key="withdrawal_years"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            withdrawal_return = st.slider(
                "Erwartete Rendite (% p.a.)",
                min_value=0.0,
                max_value=15.0,
                value=6.0,
                step=0.5,
                key="withdrawal_return"
            ) / 100

        with col2:
            withdrawal_volatility = st.slider(
                "Volatilität (% p.a.)",
                min_value=5.0,
                max_value=40.0,
                value=15.0,
                step=1.0,
                key="withdrawal_volatility"
            ) / 100

        with col3:
            withdrawal_inflation = st.slider(
                "Inflation (% p.a.)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
                key="withdrawal_inflation"
            ) / 100

        adjust_for_inflation = st.checkbox(
            "Entnahme an Inflation anpassen",
            value=True,
            help="Erhöht die Entnahme jährlich um die Inflationsrate, um die Kaufkraft zu erhalten"
        )

        # Calculate withdrawal rate
        annual_withdrawal = monthly_withdrawal * 12
        withdrawal_rate = annual_withdrawal / withdrawal_initial * 100

        st.info(f"**Entnahmerate**: {withdrawal_rate:.1f}% p.a. (Die '4%-Regel' gilt als konservativ)")

        if st.button("🏦 Entnahme simulieren", key="run_withdrawal"):
            with st.spinner("Simuliere Entnahme-Szenarien..."):
                simulator = WithdrawalSimulator(n_simulations=5000)

                withdrawal_results = simulator.simulate(
                    initial_value=withdrawal_initial,
                    monthly_withdrawal=monthly_withdrawal,
                    expected_annual_return=withdrawal_return,
                    annual_volatility=withdrawal_volatility,
                    years=withdrawal_years,
                    inflation_rate=withdrawal_inflation,
                    adjust_for_inflation=adjust_for_inflation
                )

                st.session_state.withdrawal_results = withdrawal_results

        # Display results
        if st.session_state.withdrawal_results is not None:
            wr = st.session_state.withdrawal_results

            st.markdown("---")
            st.subheader("Ergebnisse")

            # Success rate gauge
            col1, col2 = st.columns([1, 2])

            with col1:
                fig_gauge = plot_success_rate_gauge(wr.success_rate)
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown(f"""
                **Interpretation:**
                - ✅ **{wr.success_rate*100:.1f}%** der Simulationen: Geld reicht
                - ❌ **{wr.failure_rate*100:.1f}%** der Simulationen: Geld geht aus
                """)

            with col2:
                # Key metrics
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Median Endwert", format_currency(wr.median_final_value))
                with cols[1]:
                    st.metric("Gesamtentnahme (Median)", format_currency(wr.total_withdrawn_median))
                with cols[2]:
                    if wr.earliest_depletion:
                        st.metric("Früheste Erschöpfung", f"{wr.earliest_depletion/12:.1f} Jahre")
                    else:
                        st.metric("Früheste Erschöpfung", "Nie")

            # Charts
            st.subheader("Vermögensentwicklung")
            fig_paths = plot_withdrawal_simulation(wr)
            st.plotly_chart(fig_paths, use_container_width=True)

            # Percentile analysis
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Endwert-Perzentile")
                perc_data = {
                    'Perzentil': ['5%', '25%', '50% (Median)', '75%', '95%'],
                    'Endwert': [
                        format_currency(wr.percentile_5),
                        format_currency(wr.percentile_25),
                        format_currency(wr.percentile_50),
                        format_currency(wr.percentile_75),
                        format_currency(wr.percentile_95)
                    ]
                }
                st.dataframe(pd.DataFrame(perc_data), use_container_width=True)

            with col2:
                if wr.failure_rate > 0:
                    st.subheader("Erschöpfungszeit-Verteilung")
                    fig_depletion = plot_depletion_histogram(wr)
                    st.plotly_chart(fig_depletion, use_container_width=True)
                else:
                    st.success("In allen Simulationen reicht das Vermögen bis zum Ende des Zeitraums!")

            # SWR Calculator
            st.markdown("---")
            st.subheader("💡 Sichere Entnahmerate berechnen")

            target_success = st.slider(
                "Gewünschte Erfolgsquote",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                format="%.0f%%",
                key="target_success"
            )

            if st.button("Sichere Entnahmerate berechnen", key="calc_swr"):
                with st.spinner("Berechne optimale Entnahmerate..."):
                    simulator = WithdrawalSimulator(n_simulations=3000)
                    swr_result = simulator.find_safe_withdrawal_rate(
                        initial_value=withdrawal_initial,
                        expected_annual_return=withdrawal_return,
                        annual_volatility=withdrawal_volatility,
                        years=withdrawal_years,
                        target_success_rate=target_success,
                        inflation_rate=withdrawal_inflation,
                        adjust_for_inflation=adjust_for_inflation
                    )

                    st.success(f"""
                    **Ergebnis für {target_success*100:.0f}% Erfolgswahrscheinlichkeit:**

                    - 💰 **Sichere monatliche Entnahme**: {format_currency(swr_result['monthly_withdrawal'])}
                    - 📊 **Sichere Entnahmerate (SWR)**: {swr_result['withdrawal_rate_pct']:.2f}% p.a.
                    - 📅 **Jährliche Entnahme**: {format_currency(swr_result['annual_withdrawal'])}
                    """)

    # TAB 5: Efficient Frontier
    with tab5:
        st.header("🎯 Efficient Frontier - Portfolio-Optimierung")

        st.markdown("""
        Die **Efficient Frontier** (Effizienzgrenze) zeigt alle optimalen Portfolio-Kombinationen.
        Kein anderes Portfolio bietet bei gleichem Risiko mehr Rendite oder bei gleicher Rendite weniger Risiko.
        """)

        frontier_result = st.session_state.efficient_frontier

        if frontier_result is not None:
            # Plot the efficient frontier
            current_return = portfolio.annualized_expected_return()
            current_vol = portfolio.annualized_expected_volatility()

            fig_frontier = plot_efficient_frontier(
                frontier_result,
                current_weights=portfolio.weights,
                current_return=current_return,
                current_volatility=current_vol,
                ticker_symbols=portfolio.tickers
            )
            st.plotly_chart(fig_frontier, use_container_width=True)

            # Optimal portfolios comparison
            st.markdown("---")
            st.subheader("Optimale Portfolios")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### 🔴 Aktuelles Portfolio")
                st.metric("Erwartete Rendite", f"{current_return*100:.1f}%")
                st.metric("Volatilität", f"{current_vol*100:.1f}%")
                current_sharpe = (current_return - risk_free_rate) / current_vol if current_vol > 0 else 0
                st.metric("Sharpe Ratio", f"{current_sharpe:.2f}")

                st.markdown("**Gewichtung:**")
                for ticker, weight in zip(portfolio.tickers, portfolio.weights):
                    if weight > 0.001:
                        st.write(f"- {ticker}: {weight*100:.1f}%")

            with col2:
                st.markdown("### ⭐ Max Sharpe Portfolio")
                max_sharpe = frontier_result.max_sharpe_portfolio
                st.metric("Erwartete Rendite", f"{max_sharpe.expected_return*100:.1f}%")
                st.metric("Volatilität", f"{max_sharpe.volatility*100:.1f}%")
                st.metric("Sharpe Ratio", f"{max_sharpe.sharpe_ratio:.2f}")

                st.markdown("**Optimale Gewichtung:**")
                for ticker, weight in max_sharpe.get_weights_dict().items():
                    if weight > 0.001:
                        st.write(f"- {ticker}: {weight*100:.1f}%")

            with col3:
                st.markdown("### 💎 Min Volatilität Portfolio")
                min_vol = frontier_result.min_volatility_portfolio
                st.metric("Erwartete Rendite", f"{min_vol.expected_return*100:.1f}%")
                st.metric("Volatilität", f"{min_vol.volatility*100:.1f}%")
                st.metric("Sharpe Ratio", f"{min_vol.sharpe_ratio:.2f}")

                st.markdown("**Optimale Gewichtung:**")
                for ticker, weight in min_vol.get_weights_dict().items():
                    if weight > 0.001:
                        st.write(f"- {ticker}: {weight*100:.1f}%")

            # Optimal weights visualization
            st.markdown("---")
            st.subheader("Gewichtungsvergleich")

            col1, col2 = st.columns(2)

            with col1:
                fig_max_sharpe = plot_optimal_weights(
                    max_sharpe.weights,
                    portfolio.tickers,
                    "Max Sharpe Ratio Portfolio"
                )
                st.plotly_chart(fig_max_sharpe, use_container_width=True)

            with col2:
                fig_min_vol = plot_optimal_weights(
                    min_vol.weights,
                    portfolio.tickers,
                    "Minimum Volatilität Portfolio"
                )
                st.plotly_chart(fig_min_vol, use_container_width=True)

            # Interpretation
            st.markdown("---")
            with st.expander("📚 Interpretation der Efficient Frontier"):
                st.markdown("""
                ### Was zeigt der Chart?

                - **Bunte Punkte**: Zufällig generierte Portfolios (Farbe = Sharpe Ratio)
                - **Rote Linie**: Die Efficient Frontier - alle optimalen Kombinationen
                - **Goldener Stern**: Portfolio mit der höchsten Sharpe Ratio (bestes Rendite/Risiko-Verhältnis)
                - **Blauer Diamant**: Portfolio mit der niedrigsten Volatilität
                - **Roter Kreis**: Ihr aktuelles Portfolio

                ### Empfehlungen

                1. **Liegt Ihr Portfolio unter der roten Linie?** Dann gibt es bessere Kombinationen!
                2. **Max Sharpe Portfolio**: Ideal, wenn Sie das beste Verhältnis von Rendite zu Risiko wollen
                3. **Min Volatilität Portfolio**: Ideal, wenn Sie ruhig schlafen möchten

                ### Einschränkungen

                - Die Optimierung basiert auf **historischen Daten** - die Zukunft kann anders sein
                - Keine Berücksichtigung von Transaktionskosten beim Umschichten
                - Extreme Gewichtungen (z.B. 90% in einer Aktie) können unrealistisch sein
                """)

        else:
            st.warning("Die Efficient Frontier konnte nicht berechnet werden. Starten Sie eine Simulation mit mindestens 2 Assets.")

    # TAB 6: Scenario Analysis
    with tab6:
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

    # TAB 7: Export
    with tab7:
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
                format_currency(initial_value),
                format_currency(results.mean_final_value),
                format_currency(results.median_final_value),
                format_currency(results.std_final_value),
                format_currency(results.min_value),
                format_currency(results.max_value),
                format_currency(var_value),
                format_currency(cvar_value),
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
        - 🏦 **Entnahme-Simulation**: Ruhestandsplanung
        - 🎯 **Efficient Frontier**: Portfolio-Optimierung
        - 🎭 **Szenarien**: Bull/Bear/Crash-Analyse
        - 📥 **Export**: Excel & CSV Reports
        """)

    # Ausführliche Einführung für Laien
    st.markdown("---")
    st.header("📚 Einführung: Was ist Monte Carlo Simulation?")

    with st.expander("🎲 Was ist eine Monte Carlo Simulation?", expanded=True):
        st.markdown("""
        ### Die Idee dahinter

        Stellen Sie sich vor, Sie werfen eine Münze 10.000 Mal. Sie können nicht vorhersagen,
        ob bei einem einzelnen Wurf Kopf oder Zahl kommt – aber nach 10.000 Würfen wissen Sie
        ziemlich genau, dass etwa 50% Kopf waren.

        **Die Monte Carlo Simulation funktioniert genauso:**

        Anstatt zu versuchen, die Zukunft der Börse vorherzusagen (unmöglich!), simulieren wir
        tausende mögliche Zukunftsverläufe basierend auf historischen Daten. Jede Simulation
        ist wie ein "Was wäre wenn"-Szenario.

        ### Warum ist das nützlich?

        - **Risiko verstehen**: Sie sehen nicht nur den "erwarteten" Gewinn, sondern auch,
          wie schlecht es im Worst Case laufen könnte
        - **Wahrscheinlichkeiten**: "Mit 95% Wahrscheinlichkeit verliere ich nicht mehr als X Euro"
        - **Entscheidungshilfe**: Ist mein Portfolio zu riskant? Soll ich anders gewichten?

        ### Wie funktioniert es technisch?

        1. Wir laden **historische Kursdaten** Ihrer gewählten Aktien
        2. Daraus berechnen wir **Durchschnittsrenditen** und **Volatilität** (Schwankungsbreite)
        3. Wir würfeln zufällige Tagesrenditen, die statistisch zu den historischen Daten passen
        4. Das wiederholen wir **10.000 Mal** für den gesamten Zeithorizont
        5. Am Ende haben wir 10.000 mögliche Endwerte – und können Statistiken darüber berechnen
        """)

    with st.expander("📊 Eingaben erklärt: Was bedeuten die Felder?"):
        st.markdown("""
        ### Portfolio-Einstellungen

        | Eingabe | Erklärung | Beispiel |
        |---------|-----------|----------|
        | **Ticker-Symbole** | Börsenkürzel der Aktien/ETFs. Finden Sie auf Yahoo Finance. | AAPL = Apple, MSFT = Microsoft, VOO = S&P 500 ETF |
        | **Anfangskapital** | Wie viel Geld Sie investieren möchten | 100.000 € |
        | **Gewichtungen** | Wie Sie Ihr Geld auf die Aktien verteilen | 40% Apple, 30% Microsoft, 30% Google |
        | **Benchmark** | Ein Vergleichsindex, um Ihre Performance einzuordnen | SPY = S&P 500, ^GDAXI = DAX |

        ### Simulation

        | Eingabe | Erklärung | Empfehlung |
        |---------|-----------|------------|
        | **Anzahl Simulationen** | Wie oft wir die Zukunft "durchspielen" | 10.000 ist ein guter Standard |
        | **Zeithorizont** | Wie viele Jahre in die Zukunft simulieren | Ihr tatsächlicher Anlagehorizont |
        | **Historische Daten** | Zeitraum für die Berechnung der Statistiken | 5 Jahre ist ein guter Kompromiss |

        ### Rebalancing

        **Was ist Rebalancing?** Wenn eine Aktie stark steigt, verschiebt sich Ihre Gewichtung.
        Rebalancing bedeutet: Regelmäßig zurück zur Zielgewichtung.

        | Strategie | Bedeutung |
        |-----------|-----------|
        | **Buy & Hold** | Einmal kaufen, nie umschichten – einfach, aber Gewichte driften |
        | **Monatlich/Quartalsweise** | Regelmäßig zur Zielgewichtung zurückkehren |
        | **Threshold 5%/10%** | Nur umschichten, wenn Abweichung > 5% oder 10% |

        ### Risiko-Einstellungen

        | Eingabe | Erklärung |
        |---------|-----------|
        | **Konfidenzlevel** | Für VaR/CVaR: "Mit 95% Sicherheit verliere ich nicht mehr als..." |
        | **Risikofreier Zinssatz** | Rendite einer "sicheren" Anlage (z.B. Staatsanleihen). Wird für Sharpe Ratio verwendet. |
        """)

    with st.expander("📈 Ergebnisse erklärt: Was bedeuten die Kennzahlen?"):
        st.markdown("""
        ### Basis-Kennzahlen

        | Kennzahl | Bedeutung | Gut oder schlecht? |
        |----------|-----------|-------------------|
        | **Erwarteter Endwert** | Durchschnitt aller 10.000 Simulationen | Höher = besser |
        | **Median Endwert** | Der "mittlere" Wert (50% sind besser, 50% schlechter) | Oft realistischer als Durchschnitt |
        | **Minimum/Maximum** | Best Case und Worst Case aus allen Simulationen | Zeigt die Bandbreite |

        ### Risiko-Kennzahlen

        | Kennzahl | Bedeutung | Beispiel |
        |----------|-----------|----------|
        | **VaR (Value at Risk)** | "Mit 95% Wahrscheinlichkeit verliere ich nicht mehr als X" | VaR 95% = -15.000€ bedeutet: In 95 von 100 Fällen ist der Verlust kleiner |
        | **CVaR (Conditional VaR)** | "Wenn es schlecht läuft, wie schlecht im Schnitt?" | Immer schlechter als VaR – zeigt das "Tail Risk" |
        | **Max Drawdown** | Größter Verlust vom Höchststand | 30% = Portfolio fiel mal um 30% vom Höchststand |
        | **Volatilität** | Wie stark schwankt der Wert? | 20% = typische Aktien-Volatilität |

        ### Performance-Kennzahlen

        | Kennzahl | Bedeutung | Interpretation |
        |----------|-----------|----------------|
        | **Sharpe Ratio** | Rendite pro Risikoeinheit | > 1 = gut, > 2 = sehr gut |
        | **Sortino Ratio** | Wie Sharpe, aber nur Abwärtsrisiko zählt | Besser für asymmetrische Renditen |
        | **Alpha** | Überrendite gegenüber dem Markt | > 0% = Sie schlagen den Markt |
        | **Beta** | Sensitivität zum Markt | 1.0 = wie Markt, 1.5 = 50% mehr Schwankung |

        ### Perzentile verstehen

        - **5. Perzentil**: "In 5% der Fälle war das Ergebnis schlechter als dieser Wert"
        - **50. Perzentil**: Der Median – die Hälfte ist besser, die Hälfte schlechter
        - **95. Perzentil**: "In 5% der Fälle war das Ergebnis besser als dieser Wert"
        """)

    with st.expander("🎭 Szenarien erklärt"):
        st.markdown("""
        ### Was sind die Szenarien?

        Zusätzlich zur normalen Simulation testen wir Ihr Portfolio unter verschiedenen
        Marktbedingungen:

        | Szenario | Beschreibung | Rendite-Anpassung | Volatilität |
        |----------|--------------|-------------------|-------------|
        | **Bullenmarkt** | Starkes Wirtschaftswachstum, Optimismus | +8% p.a. | -20% |
        | **Normal** | Durchschnittliche Bedingungen | ±0% | ±0% |
        | **Bärenmarkt** | Wirtschaftliche Abschwächung | -10% p.a. | +30% |
        | **Crash** | Schwere Krise (wie 2008, 2020) | -30% p.a. | +150% |
        | **Hohe Volatilität** | Unsichere Märkte | -2% p.a. | +100% |
        | **Stagflation** | Niedrige Renditen, hohe Inflation | -5% p.a. | +50% |

        **Warum ist das wichtig?**

        Sie sehen, wie Ihr Portfolio in Krisenzeiten reagieren würde.
        Ein Portfolio, das im Crash-Szenario 60% verliert, ist vielleicht zu riskant.
        """)

    with st.expander("💰 Sparplan erklärt"):
        st.markdown("""
        ### Was ist ein Sparplan?

        Statt einmalig zu investieren, zahlen Sie **jeden Monat einen festen Betrag** ein.
        Das nennt man auch "Cost Averaging" oder "Durchschnittskosteneffekt".

        ### Vorteile

        - **Geringeres Timing-Risiko**: Sie kaufen mal teuer, mal günstig – im Schnitt OK
        - **Disziplin**: Automatisches Sparen ohne Emotionen
        - **Einstieg mit wenig Kapital**: Sie brauchen nicht sofort 100.000€

        ### Was zeigt die Simulation?

        - **Gesamteinzahlung**: Ihre Summe aller monatlichen Beiträge + Anfangskapital
        - **Erwarteter Endwert**: Was Sie am Ende wahrscheinlich haben
        - **Gewinn**: Endwert minus Einzahlungen = Ihr Renditegewinn
        - **Visualisierung**: Rote Linie = Ihre Einzahlungen, Grüne Linie = erwarteter Wert
        """)

    with st.expander("🏦 Entnahme-Simulation erklärt"):
        st.markdown("""
        ### Wofür ist die Entnahme-Simulation?

        Die wichtigste Frage für die Ruhestandsplanung: **"Wie lange reicht mein Geld,
        wenn ich jeden Monat X Euro entnehme?"**

        ### Die 4%-Regel

        Eine bekannte Faustregel besagt: Sie können etwa **4% Ihres Vermögens pro Jahr
        entnehmen**, ohne dass es innerhalb von 30 Jahren aufgebraucht wird (mit hoher
        Wahrscheinlichkeit). Bei 500.000€ wären das 20.000€/Jahr oder ~1.670€/Monat.

        ### Was simuliert die App?

        - Tausende mögliche Verläufe der Börse
        - Monatliche Entnahmen (optional an Inflation angepasst)
        - In wie vielen Szenarien das Geld reicht (Erfolgsquote)
        - Wann das Geld im schlimmsten Fall aufgebraucht ist

        ### Wichtige Kennzahlen

        | Kennzahl | Bedeutung |
        |----------|-----------|
        | **Erfolgsquote** | In wie viel % der Simulationen reicht das Geld |
        | **Entnahmerate** | Jährliche Entnahme / Anfangsvermögen |
        | **Sichere Entnahmerate (SWR)** | Rate, bei der z.B. 95% Erfolgswahrscheinlichkeit erreicht wird |

        ### Inflation anpassen?

        Wenn aktiviert, steigt Ihre monatliche Entnahme jedes Jahr um die Inflationsrate.
        Das erhält Ihre **Kaufkraft**, aber das Geld geht schneller aus.
        """)

    with st.expander("🎯 Efficient Frontier erklärt"):
        st.markdown("""
        ### Was ist die Efficient Frontier?

        Die Efficient Frontier (Effizienzgrenze) ist ein zentrales Konzept der
        **Modernen Portfolio-Theorie** von Harry Markowitz (Nobelpreis 1990).

        Die Idee: Bei mehreren Aktien können Sie das Risiko **durch kluge Gewichtung reduzieren**,
        ohne die erwartete Rendite zu senken – das nennt man **Diversifikation**.

        ### Was zeigt der Chart?

        - **X-Achse**: Risiko (Volatilität) des Portfolios
        - **Y-Achse**: Erwartete Rendite des Portfolios
        - **Jeder Punkt**: Eine mögliche Kombination von Gewichtungen

        ### Die rote Linie (Efficient Frontier)

        Portfolios auf dieser Linie sind **optimal**:
        - Kein anderes Portfolio bietet bei gleichem Risiko mehr Rendite
        - Kein anderes Portfolio bietet bei gleicher Rendite weniger Risiko

        ### Besondere Portfolios

        | Portfolio | Bedeutung | Für wen? |
        |-----------|-----------|----------|
        | **Max Sharpe** ⭐ | Bestes Rendite/Risiko-Verhältnis | Die meisten Anleger |
        | **Min Volatilität** 💎 | Geringstes Risiko | Konservative Anleger |

        ### Einschränkungen

        - Basiert auf **historischen Daten** – Zukunft kann anders sein
        - Optimale Gewichtungen können sich schnell ändern
        - Transaktionskosten beim Umschichten nicht berücksichtigt
        - Extreme Gewichtungen (z.B. 90% in einer Aktie) sind praktisch problematisch
        """)

    with st.expander("⚠️ Wichtige Hinweise & Limitationen"):
        st.markdown("""
        ### Was diese Simulation NICHT kann

        1. **Die Zukunft vorhersagen**: Wir simulieren basierend auf der Vergangenheit.
           Die Zukunft kann völlig anders sein!

        2. **Schwarze Schwäne berücksichtigen**: Ereignisse wie COVID-19 oder die
           Finanzkrise 2008 sind in historischen Daten selten – und in der Zukunft
           könnten ganz neue Krisen auftreten.

        3. **Steuern und Gebühren**: Die Simulation ignoriert Transaktionskosten,
           Depotgebühren und Steuern auf Gewinne.

        4. **Währungsrisiken**: Bei US-Aktien haben Sie als Euro-Anleger auch ein
           Dollar/Euro-Risiko.

        ### Wie sollten Sie die Ergebnisse nutzen?

        - **Zur Orientierung**, nicht als exakte Prognose
        - **Zum Vergleich** verschiedener Portfolio-Zusammensetzungen
        - **Zum Risikoverständnis**: Können Sie einen 30% Verlust verkraften?
        - **Als Diskussionsgrundlage** mit einem Finanzberater

        ### Keine Anlageberatung!

        Diese App ist ein **Bildungs- und Planungstool**. Sie ersetzt keine
        professionelle Finanzberatung. Investieren Sie nie Geld, das Sie
        kurzfristig brauchen könnten!
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>Monte Carlo Portfolio Simulation</strong></p>
    <p>Erdacht von <strong>Markus O. Thalhamer</strong>
    (<a href="mailto:mthalhamer@thalhamer.com">mthalhamer@thalhamer.com</a>)
    mit Unterstützung von <strong>Claude</strong></p>
    <p style="font-size: 0.8rem;">Daten von Yahoo Finance | Erstellt mit Streamlit</p>
</div>
""", unsafe_allow_html=True)
