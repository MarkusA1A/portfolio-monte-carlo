"""
Monte Carlo Portfolio Simulation - Streamlit Web Application
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
from src.data.market_data import MarketDataProvider
from src.portfolio.rebalancing import (
    get_available_strategies,
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
    calculate_volatility
)
from src.visualization.charts import (
    plot_simulation_paths,
    plot_distribution,
    plot_var_cone,
    plot_correlation_heatmap,
    plot_portfolio_weights,
    plot_drawdown
)

# Page configuration
st.set_page_config(
    page_title="Portfolio Monte Carlo Simulation",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Monte Carlo Portfolio Simulation")
st.markdown("---")

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'loaded_config' not in st.session_state:
    st.session_state.loaded_config = None


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

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Konfiguration")

    # Portfolio Load/Save Section
    st.subheader("💾 Portfolio Laden/Speichern")

    uploaded_file = st.file_uploader(
        "Portfolio laden",
        type=["json"],
        help="Laden Sie eine gespeicherte Portfolio-Konfiguration"
    )

    if uploaded_file is not None:
        try:
            config_data = json.load(uploaded_file)
            st.session_state.loaded_config = load_portfolio_config(config_data)
            st.success(f"Portfolio '{st.session_state.loaded_config['name']}' geladen!")
        except Exception as e:
            st.error(f"Fehler beim Laden: {e}")
            st.session_state.loaded_config = None

    # Use loaded config or defaults
    loaded = st.session_state.loaded_config

    st.markdown("---")

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

    # Weight configuration
    st.subheader("Gewichtungen")

    weights = {}
    remaining = 100.0

    for i, ticker in enumerate(tickers):
        # Get default weight from loaded config or calculate equal weight
        if loaded and ticker in loaded["weights"]:
            default_weight = loaded["weights"][ticker] * 100
        else:
            default_weight = 100.0 / len(tickers)

        if i == len(tickers) - 1:
            # Last ticker gets remaining weight
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
        index=default_period_idx,
        help="Zeitraum für Berechnung der Statistiken"
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
        "Konfidenzlevel für VaR/CVaR",
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

    # Run button
    st.markdown("---")
    run_simulation = st.button("🚀 Simulation starten", type="primary", use_container_width=True)

    # Save Portfolio Section
    st.markdown("---")
    st.subheader("💾 Portfolio speichern")

    portfolio_name = st.text_input(
        "Portfolio-Name",
        value=loaded["name"] if loaded else "",
        placeholder="Mein Portfolio"
    )

    # Create config for download
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

# Map rebalancing option to strategy
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

# Main content
if run_simulation:
    if len(tickers) < 2:
        st.error("Bitte mindestens 2 Ticker eingeben.")
    elif abs(sum(weights.values()) - 1.0) > 0.01:
        st.error(f"Gewichtungen müssen 100% ergeben. Aktuell: {sum(weights.values())*100:.1f}%")
    else:
        with st.spinner("Lade Marktdaten..."):
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
            except Exception as e:
                st.error(f"Fehler beim Laden der Daten: {e}")
                st.stop()

        with st.spinner(f"Führe {num_simulations:,} Simulationen durch..."):
            simulator = MonteCarloSimulator(
                num_simulations=num_simulations,
                time_horizon=time_horizon_days
            )
            strategy = get_rebalancing_strategy(rebalancing_option)
            results = simulator.run_simulation(portfolio, strategy)
            st.session_state.results = results

        st.success("Simulation abgeschlossen!")

# Display results
if st.session_state.results is not None and st.session_state.portfolio is not None:
    results = st.session_state.results
    portfolio = st.session_state.portfolio

    # Portfolio Overview
    st.header("📊 Portfolio Übersicht")

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

    # Correlation Matrix
    st.subheader("Korrelationsmatrix")
    corr_matrix = portfolio.get_correlation_matrix()
    fig_corr = plot_correlation_heatmap(corr_matrix, portfolio.tickers)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # Simulation Results
    st.header("🎲 Simulationsergebnisse")

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    var_value = calculate_var(results.returns, confidence_level, initial_value)
    cvar_value = calculate_cvar(results.returns, confidence_level, initial_value)

    with col1:
        st.metric(
            "Erwarteter Endwert",
            f"€{results.mean_final_value:,.0f}",
            f"{results.mean_return*100:+.1f}%"
        )

    with col2:
        st.metric(
            "Median Endwert",
            f"€{results.median_final_value:,.0f}",
            f"{(results.median_final_value/initial_value - 1)*100:+.1f}%"
        )

    with col3:
        st.metric(
            f"VaR ({confidence_level:.0%})",
            f"€{var_value:,.0f}",
            delta=None
        )

    with col4:
        st.metric(
            f"CVaR ({confidence_level:.0%})",
            f"€{cvar_value:,.0f}",
            delta=None
        )

    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)

    median_path = np.median(results.portfolio_values, axis=0)
    daily_returns = np.diff(median_path) / median_path[:-1]

    sharpe = calculate_sharpe_ratio(daily_returns, risk_free_rate)
    sortino = calculate_sortino_ratio(daily_returns, risk_free_rate)
    max_dd, _, _ = calculate_max_drawdown(median_path)
    vol = calculate_volatility(daily_returns)

    with col1:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

    with col2:
        st.metric("Sortino Ratio", f"{sortino:.2f}")

    with col3:
        st.metric("Max Drawdown", f"{max_dd*100:.1f}%")

    with col4:
        st.metric("Ann. Volatilität", f"{vol*100:.1f}%")

    # Percentile Table
    st.subheader("Perzentil-Analyse")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = [results.percentile(p) for p in percentiles]
    percentile_returns = [(v/initial_value - 1) * 100 for v in percentile_values]

    perc_df = pd.DataFrame({
        'Perzentil': [f'{p}%' for p in percentiles],
        'Endwert (€)': percentile_values,
        'Rendite (%)': percentile_returns
    })

    st.dataframe(
        perc_df.style.format({
            'Endwert (€)': '€{:,.0f}',
            'Rendite (%)': '{:+.1f}%'
        }),
        use_container_width=True
    )

    st.markdown("---")

    # Charts
    st.header("📈 Visualisierungen")

    # Simulation Paths
    st.subheader("Simulationspfade")
    fig_paths = plot_simulation_paths(results, num_paths=100, initial_value=initial_value)
    st.plotly_chart(fig_paths, use_container_width=True)

    # Distribution and VaR Cone
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Verteilung der Endwerte")
        fig_dist = plot_distribution(
            results.final_values,
            initial_value,
            var_level=var_value,
            cvar_level=cvar_value
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        st.subheader("VaR Kegel")
        fig_var = plot_var_cone(results)
        st.plotly_chart(fig_var, use_container_width=True)

    # Drawdown
    st.subheader("Drawdown (Median-Pfad)")
    fig_dd = plot_drawdown(median_path)
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("---")

    # Summary Statistics
    st.header("📋 Zusammenfassung")

    summary_data = {
        'Metrik': [
            'Anfangskapital',
            'Erwarteter Endwert',
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

    st.markdown("""
    ### So funktioniert's:

    1. **Portfolio definieren**: Geben Sie die Ticker-Symbole ein (z.B. AAPL, MSFT)
    2. **Gewichtungen festlegen**: Verteilen Sie Ihr Kapital auf die Assets
    3. **Simulation konfigurieren**: Wählen Sie Zeithorizont und Anzahl der Simulationen
    4. **Rebalancing wählen**: Entscheiden Sie sich für eine Strategie
    5. **Analyse starten**: Klicken Sie auf "Simulation starten"

    ### Was berechnet wird:

    - **Monte Carlo Simulation**: Tausende mögliche Marktszenarien
    - **VaR & CVaR**: Risikokennzahlen für Ihre Verlustabschätzung
    - **Sharpe & Sortino Ratio**: Risikoadjustierte Renditekennzahlen
    - **Korrelationen**: Wie Ihre Assets zusammenhängen
    """)

# Footer
st.markdown("---")
st.caption("Monte Carlo Portfolio Simulation | Basierend auf historischen Daten von Yahoo Finance")
