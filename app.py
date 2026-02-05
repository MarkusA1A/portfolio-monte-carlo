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
from src.simulation.tax_costs import TaxConfig, TransactionCostConfig, TaxCostCalculator
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
import yfinance as yf
import re


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_code_statistics() -> dict:
    """Count lines of code in the project."""
    base_path = Path(__file__).parent
    stats = {
        'app': 0,
        'src': 0,
        'tests': 0,
        'total': 0,
        'files': 0
    }

    # Count app.py
    app_file = base_path / 'app.py'
    if app_file.exists():
        stats['app'] = len(app_file.read_text().splitlines())
        stats['files'] += 1

    # Count src/ files
    src_path = base_path / 'src'
    if src_path.exists():
        for py_file in src_path.rglob('*.py'):
            if '__pycache__' not in str(py_file):
                stats['src'] += len(py_file.read_text().splitlines())
                stats['files'] += 1

    # Count tests/ files
    tests_path = base_path / 'tests'
    if tests_path.exists():
        for py_file in tests_path.rglob('*.py'):
            if '__pycache__' not in str(py_file):
                stats['tests'] += len(py_file.read_text().splitlines())
                stats['files'] += 1

    stats['total'] = stats['app'] + stats['src'] + stats['tests']
    return stats


def is_isin(text: str) -> bool:
    """Check if text looks like an ISIN (12 chars, starts with 2 letters)."""
    if len(text) != 12:
        return False
    return bool(re.match(r'^[A-Z]{2}[A-Z0-9]{10}$', text))


def isin_to_ticker(isin: str) -> str | None:
    """Convert ISIN to ticker symbol using yfinance search."""
    try:
        search = yf.Search(isin, max_results=5)
        if search.quotes:
            for quote in search.quotes:
                if quote.get('symbol'):
                    return quote['symbol']
    except Exception:
        pass
    return None


def process_ticker_input(input_str: str) -> tuple[str, str | None]:
    """
    Process a single ticker/ISIN input.
    Returns (ticker, original_isin) - original_isin is set if input was an ISIN.
    """
    cleaned = input_str.strip().upper()
    if is_isin(cleaned):
        ticker = isin_to_ticker(cleaned)
        if ticker:
            return (ticker, cleaned)
        return (cleaned, cleaned)  # Fallback: use ISIN as ticker
    return (cleaned, None)


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


def scroll_to_top():
    """Inject JavaScript to scroll to top of the main content area."""
    js = """
    <script>
        // Scroll the main content area to top
        const mainContent = window.parent.document.querySelector('section.main');
        if (mainContent) {
            mainContent.scrollTo({top: 0, behavior: 'smooth'});
        }
        // Also scroll the window itself
        window.parent.scrollTo({top: 0, behavior: 'smooth'});
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)


def parse_german_number(text: str, default: int = 0) -> int:
    """Parse German-formatted number (dots as thousand separators) to integer."""
    if not text:
        return default
    # Remove dots (thousand separators) and spaces
    cleaned = text.replace(".", "").replace(" ", "").replace("€", "")
    # Replace comma with dot for decimals (if any)
    cleaned = cleaned.replace(",", ".")
    try:
        return int(float(cleaned))
    except ValueError:
        return default


# Page configuration - responsive layout
st.set_page_config(
    page_title="Portfolio Monte Carlo Simulation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapsed on mobile for better UX
)

# Custom CSS for responsive design (Mobile: iPhone, iPad)
st.markdown("""
<style>
    /* === BASE RESPONSIVE STYLES === */
    [data-testid="stMetricValue"] {
        font-size: clamp(1rem, 2.5vw, 1.5rem);
    }
    [data-testid="stMetricLabel"] {
        font-size: clamp(0.7rem, 1.5vw, 0.9rem);
    }
    .js-plotly-plot {
        width: 100% !important;
    }

    /* === MOBILE STYLES (iPhone) === */
    @media (max-width: 768px) {
        /* Main container padding */
        .main .block-container {
            padding: 1rem 0.75rem 5rem 0.75rem !important;
            max-width: 100% !important;
        }

        /* Sidebar full width overlay on mobile */
        [data-testid="stSidebar"] {
            min-width: 100vw !important;
            width: 100vw !important;
        }
        [data-testid="stSidebar"] > div {
            padding: 1rem;
        }

        /* Horizontal scrollable tabs */
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            overflow-y: hidden;
            flex-wrap: nowrap !important;
            gap: 0.25rem;
            padding-bottom: 0.5rem;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.75rem;
            padding: 0.5rem 0.6rem;
            white-space: nowrap;
            flex-shrink: 0;
        }

        /* Columns stack vertically */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }

        /* Compact headers */
        h1 { font-size: 1.4rem !important; margin-bottom: 0.5rem !important; }
        h2 { font-size: 1.15rem !important; margin-top: 1rem !important; }
        h3 { font-size: 1rem !important; }

        /* Compact metrics grid */
        [data-testid="stMetricValue"] {
            font-size: 1rem !important;
            line-height: 1.2;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.7rem !important;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.7rem !important;
        }

        /* All inputs: prevent iOS zoom */
        input, select, textarea {
            font-size: 16px !important;
        }

        /* Number inputs touch-friendly */
        .stNumberInput > div {
            flex-direction: column;
        }
        .stNumberInput input {
            padding: 0.75rem !important;
            text-align: center;
        }

        /* Buttons full width and touch-friendly */
        .stButton button, .stFormSubmitButton button {
            padding: 0.875rem 1rem !important;
            font-size: 1rem !important;
            width: 100% !important;
            min-height: 48px;
        }

        /* Expanders more compact */
        .streamlit-expanderHeader {
            font-size: 0.9rem !important;
            padding: 0.5rem 0 !important;
        }

        /* Forms tighter spacing */
        [data-testid="stForm"] {
            padding: 0.5rem !important;
        }

        /* Charts responsive */
        .stPlotlyChart {
            margin: 0 -0.5rem;
        }

        /* DataFrames scrollable */
        .stDataFrame {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }

        /* Alerts compact */
        .stAlert {
            padding: 0.5rem 0.75rem !important;
            font-size: 0.85rem;
        }

        /* Info/Warning/Success boxes */
        [data-testid="stNotificationContentInfo"],
        [data-testid="stNotificationContentWarning"],
        [data-testid="stNotificationContentSuccess"] {
            padding: 0.5rem !important;
            font-size: 0.85rem !important;
        }

        /* Caption text */
        .stCaption {
            font-size: 0.75rem !important;
        }

        /* Selectbox */
        .stSelectbox > div > div {
            min-height: 44px;
        }

        /* Text inputs */
        .stTextInput input {
            padding: 0.75rem !important;
        }

        /* Bottom safe area for iOS */
        .main {
            padding-bottom: env(safe-area-inset-bottom, 20px);
        }
    }

    /* === SMALL MOBILE (iPhone SE, etc.) === */
    @media (max-width: 375px) {
        .main .block-container {
            padding: 0.5rem 0.5rem 5rem 0.5rem !important;
        }
        h1 { font-size: 1.2rem !important; }
        h2 { font-size: 1rem !important; }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.7rem;
            padding: 0.4rem 0.5rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 0.9rem !important;
        }
    }

    /* === TABLET STYLES (iPad) === */
    @media (min-width: 769px) and (max-width: 1024px) {
        [data-testid="stSidebar"] {
            min-width: 300px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.9rem;
        }
    }

    /* === TOUCH DEVICE IMPROVEMENTS === */
    @media (hover: none) and (pointer: coarse) {
        /* Minimum touch target size (44px Apple HIG) */
        .stSlider [data-baseweb="slider"] {
            min-height: 44px;
            padding: 0.75rem 0;
        }
        .stCheckbox label {
            min-height: 44px;
            display: flex;
            align-items: center;
            padding: 0.25rem 0;
        }
        .stSelectbox [data-baseweb="select"] {
            min-height: 44px;
        }
        .stRadio label {
            min-height: 44px;
            display: flex;
            align-items: center;
        }
        /* Prevent accidental double-taps */
        button, a {
            touch-action: manipulation;
        }
    }

    /* === DARK MODE ADJUSTMENTS === */
    @media (prefers-color-scheme: dark) {
        .stAlert {
            border-radius: 8px;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Monte Carlo Portfolio Simulation")
st.caption("© Markus O. Thalhamer")
st.markdown("### 💰 *Show me the money*")

# Initialize session state
for key in ['portfolio', 'results', 'loaded_config', 'benchmark_data',
            'savings_results', 'scenario_results', 'efficient_frontier', 'withdrawal_results',
            'swr_result', 'swr_params']:
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


def estimate_memory_mb(num_simulations: int, time_horizon_days: int, num_assets: int) -> float:
    """
    Estimate memory usage for Monte Carlo simulation in MB.

    Optimized version using float32 (4 bytes) and no price_paths storage.
    Main arrays:
    - daily_returns: (num_simulations, time_horizon, num_assets) * 4 bytes
    - portfolio_values: (num_simulations, time_horizon + 1) * 4 bytes
    """
    bytes_per_float = 4  # Using float32 now
    daily_returns = num_simulations * time_horizon_days * num_assets * bytes_per_float
    portfolio_values = num_simulations * (time_horizon_days + 1) * bytes_per_float

    # Add overhead for scenario analysis (6 scenarios with reduced simulations)
    scenario_overhead = 6 * min(num_simulations, 3000) * (time_horizon_days + 1) * bytes_per_float

    total_bytes = daily_returns + portfolio_values + scenario_overhead

    return total_bytes / (1024 * 1024)  # Convert to MB


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
        "Ticker-Symbole oder ISINs (kommasepariert)",
        value=default_tickers,
        help="z.B. AAPL, MSFT, GOOGL oder ISINs wie US0378331005, DE0007164600"
    )

    # Process inputs (handle both tickers and ISINs)
    raw_inputs = [t.strip() for t in tickers_input.split(",") if t.strip()]
    tickers = []
    isin_mappings = {}  # ISIN -> Ticker mapping for display

    for raw in raw_inputs:
        ticker, original_isin = process_ticker_input(raw)
        if ticker:
            tickers.append(ticker)
            if original_isin:
                isin_mappings[original_isin] = ticker

    # Show ISIN conversions to user
    if isin_mappings:
        conversion_text = ", ".join([f"{isin} → {ticker}" for isin, ticker in isin_mappings.items()])
        st.info(f"🔄 ISIN-Konvertierung: {conversion_text}")

    default_initial = loaded["initial_value"] if loaded else 100000
    initial_value_str = st.text_input(
        "Anfangskapital (€)",
        value=format_number(default_initial),
        help="z.B. 100.000 oder 1.000.000"
    )
    initial_value = parse_german_number(initial_value_str, default_initial)
    if initial_value < 1000:
        initial_value = 1000
        st.warning("Mindestkapital: €1.000")
    elif initial_value > 100000000:
        initial_value = 100000000
        st.warning("Maximalkapital: €100.000.000")

    # Benchmark selection with descriptions
    benchmark_options = {
        "SPY": "S&P 500 (USA) – Die 500 größten US-Unternehmen",
        "QQQ": "NASDAQ 100 (USA) – Die 100 größten Tech-Unternehmen",
        "^GDAXI": "DAX (Deutschland) – Die 40 größten deutschen Unternehmen",
        "^STOXX50E": "Euro Stoxx 50 (Europa) – Die 50 größten Eurozone-Unternehmen",
        "VTI": "Total Stock Market (USA) – Der gesamte US-Aktienmarkt"
    }
    benchmark_display = st.selectbox(
        "Benchmark",
        options=list(benchmark_options.values()),
        index=0,
        help="Wählen Sie einen Vergleichsindex, um Ihre Portfolio-Performance einzuordnen"
    )
    # Map display name back to ticker
    benchmark_ticker = [k for k, v in benchmark_options.items() if v == benchmark_display][0]

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

    # Initialize weights for new tickers
    for ticker in tickers:
        if ticker not in st.session_state.ticker_weights:
            if loaded and ticker in loaded["weights"]:
                st.session_state.ticker_weights[ticker] = loaded["weights"][ticker] * 100
            else:
                st.session_state.ticker_weights[ticker] = 100.0 / len(tickers) if tickers else 25.0

    # Collect weights with number input per ticker (2 columns for mobile-friendly layout)
    weight_values = {}
    cols = st.columns(2) if tickers else []

    for i, ticker in enumerate(tickers):
        current_weight = st.session_state.ticker_weights[ticker]
        col_idx = i % 2

        with cols[col_idx]:
            weight = st.number_input(
                f"{ticker} (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(current_weight),
                step=1.0,
                key=f"weight_{ticker}_{i}"
            )

        weight_values[ticker] = weight
        st.session_state.ticker_weights[ticker] = weight

    # Show total and status
    total = sum(weight_values.values())
    remaining = 100.0 - total

    if abs(remaining) < 0.1:
        st.success(f"✅ Summe: 100%")
    elif remaining > 0:
        st.info(f"📊 Summe: {total:.1f}% | Noch {remaining:.1f}% verfügbar")
    else:
        st.warning(f"⚠️ Summe: {total:.1f}% - bitte reduzieren!")

    # Convert to decimal weights (normalize to ensure sum = 1)
    if total > 0:
        weights = {t: w / total for t, w in weight_values.items()}
    else:
        weights = {t: 1.0 / len(tickers) for t in tickers} if tickers else {}

    # Simulation Settings
    st.subheader("Simulation")

    sim_options = [1000, 5000, 10000, 25000, 50000]
    default_sim = loaded["num_simulations"] if loaded and loaded["num_simulations"] in sim_options else 5000
    num_simulations = st.select_slider(
        "Anzahl Simulationen",
        options=sim_options,
        value=default_sim,
        help="5.000 Simulationen bieten ca. 1,4% statistische Genauigkeit – ausreichend, da die Unsicherheit in historischen Marktdaten deutlich größer ist."
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
        index=default_rebal_idx,
        help="**Rebalancing** = Regelmäßiges Zurücksetzen auf Ihre Zielgewichtung. Beispiel: Wenn Apple stark steigt, wird der Apple-Anteil zu groß. Beim Rebalancing verkaufen Sie etwas Apple und kaufen die anderen Positionen nach. **Buy & Hold**: Einmal kaufen, nie umschichten. **Monatlich/Quartalsweise/Jährlich**: Zu festen Terminen zurück zur Zielgewichtung. **Threshold 5%/10%**: Nur umschichten, wenn eine Position mehr als 5% bzw. 10% von der Zielgewichtung abweicht."
    )

    # Risk Settings
    st.subheader("Risiko")

    default_conf = loaded["confidence_level"] if loaded else 0.95
    confidence_level = st.slider(
        "Konfidenzlevel VaR/CVaR",
        min_value=0.90,
        max_value=0.99,
        value=default_conf,
        step=0.01,
        help="Das Konfidenzlevel bestimmt, wie 'sicher' die Risikoaussage sein soll. Bei **95%** bedeutet VaR: 'Mit 95% Wahrscheinlichkeit verliere ich nicht mehr als X Euro.' Die restlichen 5% sind die Extremfälle. **90%** = optimistischer (kleinerer VaR), **99%** = vorsichtiger (größerer VaR). Standard ist 95% – ein guter Kompromiss zwischen Vorsicht und Praxisnähe."
    )

    default_rf = loaded["risk_free_rate"] * 100 if loaded else 4.0
    risk_free_rate = st.number_input(
        "Risikofreier Zinssatz (%)",
        min_value=0.0,
        max_value=10.0,
        value=default_rf,
        step=0.1,
        help="Die Rendite einer 'sicheren' Anlage wie Staatsanleihen oder Tagesgeld. Wird zur Berechnung der Sharpe Ratio verwendet: Wie viel Mehrrendite erzielen Sie pro Risikoeinheit im Vergleich zur sicheren Anlage? Aktuell (2024/25) liegt der Zinssatz für deutsche Staatsanleihen bei ca. 2-4%."
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

    # Tax & Cost Settings
    st.subheader("💶 Steuern & Kosten")
    enable_tax_costs = st.checkbox(
        "Steuern & Transaktionskosten berücksichtigen",
        value=False,
        help="Berechnet österreichische KESt (27,5%) auf realisierte Gewinne bei Rebalancing und Transaktionskosten"
    )

    if enable_tax_costs:
        tax_rate = st.slider(
            "Kapitalertragssteuer (%)",
            min_value=0.0,
            max_value=50.0,
            value=27.5,
            step=0.5,
            help="Österreichische KESt: 27,5% auf Kapitalerträge. Deutschland: 26,375% (inkl. Soli)"
        ) / 100

        cost_model = st.radio(
            "Kostenmodell",
            ["Prozentual", "Flat Fee"],
            help="**Prozentual**: Prozentsatz des Handelsvolumens. **Flat Fee**: Fixer Betrag pro Transaktion."
        )

        if cost_model == "Prozentual":
            transaction_cost_pct = st.slider(
                "Transaktionskosten (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Typisch: 0,1% für Online-Broker, 0,5% für Filialbanken"
            ) / 100
            flat_fee = 0.0
        else:
            flat_fee = st.number_input(
                "Flat Fee pro Trade (€)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5,
                help="Fixer Betrag pro Kauf/Verkauf"
            )
            transaction_cost_pct = 0.0
    else:
        tax_rate = 0.275
        transaction_cost_pct = 0.001
        flat_fee = 0.0
        cost_model = "Prozentual"

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
        # Scroll to top so user sees progress
        scroll_to_top()

        # Memory estimation and automatic adjustment
        estimated_memory = estimate_memory_mb(num_simulations, time_horizon_days, len(tickers))

        # Hard limit: 800 MB max to prevent crashes
        MAX_MEMORY_MB = 800
        original_simulations = num_simulations

        if estimated_memory > MAX_MEMORY_MB:
            # Auto-reduce simulations to fit within memory limit
            while estimated_memory > MAX_MEMORY_MB and num_simulations > 1000:
                num_simulations = num_simulations // 2
                estimated_memory = estimate_memory_mb(num_simulations, time_horizon_days, len(tickers))

            st.warning(
                f"⚠️ **Automatische Anpassung:** Simulationen von {original_simulations:,} auf {num_simulations:,} reduziert, "
                f"um Abstürze zu vermeiden ({len(tickers)} Assets, {time_horizon_years} Jahre)."
            )
        elif estimated_memory > 500:
            st.info(
                f"ℹ️ Geschätzter Speicherbedarf: {estimated_memory:.0f} MB. "
                f"Bei Problemen reduzieren Sie die Simulationsanzahl."
            )

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
        try:
            simulator = MonteCarloSimulator(
                num_simulations=num_simulations,
                time_horizon=time_horizon_days
            )
            strategy = get_rebalancing_strategy(rebalancing_option)

            # Create tax/cost calculator if enabled and rebalancing is active
            tax_cost_calculator = None
            if enable_tax_costs and strategy is not None:
                tax_config = TaxConfig(tax_rate=tax_rate)
                cost_config = TransactionCostConfig(
                    use_percentage=(cost_model == "Prozentual"),
                    percentage_fee=transaction_cost_pct,
                    flat_fee_per_trade=flat_fee
                )
                tax_cost_calculator = TaxCostCalculator(
                    tax_config=tax_config,
                    cost_config=cost_config,
                    num_simulations=num_simulations,
                    num_assets=len(portfolio.assets),
                    initial_value=initial_value
                )

            # Run simulation
            sim_result = simulator.run_simulation(portfolio, strategy, tax_cost_calculator=tax_cost_calculator)

            # Handle return value (tuple if tax calculator provided)
            if tax_cost_calculator is not None:
                results, tax_cost_results = sim_result
                st.session_state.tax_cost_results = tax_cost_results
            else:
                results = sim_result
                st.session_state.tax_cost_results = None

            st.session_state.results = results
        except MemoryError:
            st.error(
                "❌ Nicht genügend Speicher! Bitte reduzieren Sie:\n"
                "- Anzahl der Simulationen\n"
                "- Zeithorizont\n"
                "- Anzahl der Assets"
            )
            st.stop()
        except Exception as e:
            st.error(f"❌ Fehler bei der Simulation: {e}")
            st.stop()

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

        # Scenario analysis - reduce simulations for large portfolios to prevent memory issues
        progress.progress(70, text="Führe Szenario-Analyse durch...")
        scenario_results = {}
        # Scale down scenario simulations based on portfolio size
        scenario_sims = min(num_simulations, 5000)
        if len(tickers) > 5:
            scenario_sims = min(scenario_sims, 3000)
        if len(tickers) > 8:
            scenario_sims = min(scenario_sims, 2000)

        try:
            for scenario_type, scenario in SCENARIOS.items():
                modified_portfolio = portfolio.copy()
                modified_portfolio.adjust_statistics(
                    scenario.return_adjustment,
                    scenario.volatility_multiplier
                )
                scenario_sim = MonteCarloSimulator(
                    num_simulations=scenario_sims,
                    time_horizon=time_horizon_days
                )
                scenario_results[scenario.name] = scenario_sim.run_simulation(modified_portfolio)
            st.session_state.scenario_results = scenario_results
        except Exception as e:
            st.warning(f"Szenario-Analyse übersprungen: {e}")
            st.session_state.scenario_results = None

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Übersicht",
        "📈 Benchmark",
        "💰 Sparplan",
        "🏦 Entnahme",
        "🎯 Efficient Frontier",
        "🎭 Szenarien",
        "📥 Export",
        "💶 Steuern & Kosten"
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
        st.subheader(f"Simulationsergebnisse ({results.num_simulations:,} Simulationen)".replace(",", "."))

        # First row
        cols = st.columns(4)
        with cols[0]:
            st.metric(
                "Erwarteter Endwert",
                format_currency(results.mean_final_value),
                f"{results.mean_return*100:+.1f}%",
                help="Der Durchschnitt aller simulierten Endwerte. Gibt an, welchen Wert Ihr Portfolio im Mittel nach dem gewählten Zeitraum haben könnte. Achtung: Einzelne Szenarien können deutlich darüber oder darunter liegen."
            )
        with cols[1]:
            st.metric(
                "Median Endwert",
                format_currency(results.median_final_value),
                f"{(results.median_final_value/initial_value - 1)*100:+.1f}%",
                help="Der mittlere Wert aller Simulationen: 50% der Ergebnisse liegen darüber, 50% darunter. Der Median ist robuster als der Durchschnitt und wird weniger von Extremwerten beeinflusst – oft die realistischere Erwartung."
            )
        with cols[2]:
            st.metric(
                f"VaR ({confidence_level:.0%})",
                format_currency(var_value),
                help=f"Value at Risk: Der maximale Verlust, der mit {confidence_level:.0%} Wahrscheinlichkeit NICHT überschritten wird. Beispiel: Ein VaR von €10.000 bei 95% bedeutet: In 95 von 100 Fällen verlieren Sie höchstens €10.000. In 5 von 100 Fällen kann der Verlust größer sein."
            )
        with cols[3]:
            st.metric(
                f"CVaR ({confidence_level:.0%})",
                format_currency(cvar_value),
                help=f"Conditional Value at Risk (auch Expected Shortfall): Der durchschnittliche Verlust in den schlimmsten {(1-confidence_level):.0%} aller Fälle. Zeigt, wie hoch die Verluste ausfallen könnten, wenn es wirklich schlecht läuft – wichtig für die Risikoplanung."
            )

        # Second row
        cols = st.columns(4)
        with cols[0]:
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                help="Misst die Rendite im Verhältnis zum eingegangenen Risiko. Berechnung: (Rendite - risikofreier Zins) / Volatilität. Interpretation: < 1 = mäßig, 1-2 = gut, > 2 = sehr gut. Je höher, desto mehr Rendite pro Risikoeinheit."
            )
        with cols[1]:
            st.metric(
                "Sortino Ratio",
                f"{sortino:.2f}",
                help="Ähnlich wie Sharpe Ratio, aber berücksichtigt nur negative Schwankungen (Verluste). Positive Schwankungen werden nicht 'bestraft'. Daher oft aussagekräftiger für Anleger, die vor allem Verluste vermeiden möchten."
            )
        with cols[2]:
            st.metric(
                "Max Drawdown",
                f"{max_dd*100:.1f}%",
                help="Der größte prozentuale Wertverlust vom Höchststand zum Tiefststand. Beispiel: -30% bedeutet, dass das Portfolio zwischenzeitlich 30% seines Höchstwertes verloren hat. Zeigt das Worst-Case-Szenario während der Anlage."
            )
        with cols[3]:
            st.metric(
                "Ann. Volatilität",
                f"{vol*100:.1f}%",
                help="Die jährliche Schwankungsbreite der Renditen. Eine Volatilität von 20% bedeutet: Die Rendite schwankt typischerweise um ±20% pro Jahr. Höhere Volatilität = höheres Risiko, aber auch höhere Chancen."
            )

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
            st.caption("ℹ️ Zeigt die mögliche Wertentwicklung über die Zeit mit Konfidenzintervallen. Der innere Bereich zeigt wahrscheinliche Szenarien, der äußere Bereich extreme aber mögliche Verläufe. Je breiter der Kegel, desto unsicherer die Prognose.")
            fig_var = plot_var_cone(results)
            st.plotly_chart(fig_var, use_container_width=True)

        # Correlation Matrix
        st.subheader("Korrelationsmatrix")
        with st.expander("ℹ️ Was zeigt die Korrelationsmatrix?", expanded=False):
            st.markdown(f"""
**Interpretation der Werte:**
- **+1.0**: Bewegen sich identisch (kein Diversifikationsvorteil)
- **0.0**: Unabhängig voneinander (gute Diversifikation)
- **-1.0**: Bewegen sich entgegengesetzt (perfekte Absicherung)

**So werden die Werte berechnet:**

1. **Historische Kurse laden**: Für jede Anlage werden die täglichen Schlusskurse der letzten {data_period} von Yahoo Finance abgerufen.

2. **Tägliche Renditen berechnen**: Aus den Kursen wird die prozentuale Veränderung von Tag zu Tag berechnet:
   - *Rendite = (Kurs heute - Kurs gestern) / Kurs gestern*

3. **Pearson-Korrelation berechnen**: Für jedes Anlagenpaar wird gemessen, wie stark die täglichen Renditen zusammenhängen:
   - Wenn Anlage A an guten Tagen steigt und Anlage B auch → hohe positive Korrelation
   - Wenn Anlage A steigt während Anlage B fällt → negative Korrelation

**Tipp:** Kombinieren Sie Anlagen mit niedriger oder negativer Korrelation, um das Gesamtrisiko zu reduzieren.
            """)
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
                st.metric("Ann. Rendite", f"{portfolio.annualized_expected_return()*100:.1f}%",
                    help="Annualisierte (jährliche) Rendite: Die durchschnittliche Rendite pro Jahr, hochgerechnet aus den historischen Daten. Zeigt, wie viel Ihr Portfolio im Schnitt pro Jahr gewonnen oder verloren hat.")
                st.metric("Ann. Volatilität", f"{portfolio.annualized_expected_volatility()*100:.1f}%",
                    help="Annualisierte Volatilität: Die jährliche Schwankungsbreite der Renditen. Je höher der Wert, desto stärker schwankt der Wert Ihres Portfolios – also höheres Risiko, aber auch höhere Chancen.")
                st.metric("Sharpe Ratio", f"{sharpe:.2f}",
                    help="Rendite pro Risikoeinheit im Vergleich zur sicheren Anlage. Formel: (Rendite - risikofreier Zins) / Volatilität. Je höher, desto besser. Vergleichen Sie mit dem Benchmark: Ist Ihre Sharpe Ratio höher, schlagen Sie den Markt risikoadjustiert.")

            with col2:
                benchmark_name = benchmark_options.get(benchmark_ticker, benchmark_ticker)
                st.subheader(f"Benchmark: {benchmark_name.split(' – ')[0]}")
                bench_return = float(benchmark_data.annualized_return)
                bench_vol = float(benchmark_data.annualized_volatility)
                st.metric("Ann. Rendite", f"{bench_return*100:.1f}%",
                    help="Die jährliche Rendite des Vergleichsindex. Liegt Ihr Portfolio darüber, haben Sie den Markt geschlagen.")
                st.metric("Ann. Volatilität", f"{bench_vol*100:.1f}%",
                    help="Die jährliche Schwankung des Index. Ist Ihre Volatilität niedriger bei ähnlicher Rendite, ist Ihr Portfolio effizienter.")
                bench_sharpe = (bench_return - risk_free_rate) / bench_vol if bench_vol > 0 else 0
                st.metric("Sharpe Ratio", f"{bench_sharpe:.2f}",
                    help="Die Sharpe Ratio des Benchmark. Vergleichen Sie: Ist Ihre Sharpe Ratio höher, erzielen Sie mehr Rendite pro Risiko als der Markt.")

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
                            st.metric("Beta", f"{beta:.2f}",
                                help="Misst, wie stark Ihr Portfolio auf Marktbewegungen reagiert. Beta = 1: bewegt sich wie der Markt. Beta > 1: schwankt stärker (z.B. 1.5 = 50% stärkere Bewegung). Beta < 1: schwankt weniger. Beta < 0: bewegt sich entgegengesetzt (selten).")
                        with cols[1]:
                            st.metric("Alpha (ann.)", f"{alpha*100:.2f}%",
                                help="Die Überrendite gegenüber dem, was aufgrund des Risikos (Beta) zu erwarten wäre. Positives Alpha = Sie schlagen den Markt durch gute Auswahl. Negatives Alpha = Sie hinken dem Markt hinterher. Alpha ist der 'Mehrwert' Ihrer Strategie.")
                        with cols[2]:
                            tracking_error = np.std(portfolio_returns - bench_returns) * np.sqrt(252)
                            st.metric("Tracking Error", f"{tracking_error*100:.1f}%",
                                help="Misst, wie stark Ihr Portfolio vom Benchmark abweicht. Niedriger Tracking Error (< 5%): Portfolio verhält sich ähnlich wie der Index. Hoher Tracking Error: Portfolio weicht stark ab – gewollt bei aktiver Strategie, ungewollt bei passiver.")
        else:
            st.warning("Benchmark-Daten konnten nicht geladen werden.")

    # TAB 3: Savings Plan
    with tab3:
        st.header("💰 Sparplan-Simulation")

        savings_results = st.session_state.savings_results

        if savings_results:
            st.caption(f"Basierend auf {savings_results.num_simulations:,} Simulationen".replace(",", "."))
            # Summary metrics
            cols = st.columns(4)
            with cols[0]:
                st.metric("Gesamteinzahlung", format_currency(savings_results.total_invested), help="Die Summe aller monatlichen Einzahlungen über den gesamten Anlagezeitraum.")
            with cols[1]:
                st.metric("Erwarteter Endwert", format_currency(savings_results.mean_final_value), help="Der durchschnittliche Endwert aller Simulationen. Gibt an, welchen Wert Ihr Sparplan im Mittel erreichen könnte.")
            with cols[2]:
                st.metric("Erwarteter Gewinn", format_currency(savings_results.mean_profit), f"{savings_results.mean_return*100:+.1f}%", help="Die Differenz zwischen erwartetem Endwert und Gesamteinzahlung – Ihr durchschnittlicher Gewinn durch Kursgewinne und Zinseszins.")
            with cols[3]:
                st.metric("Median Endwert", format_currency(savings_results.median_final_value), help="Der mittlere Wert: 50% der Simulationen liegen darüber, 50% darunter. Oft realistischer als der Durchschnitt, da Ausreißer weniger Einfluss haben.")

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

        # Age and time horizon inputs
        col1, col2, col3 = st.columns(3)

        with col1:
            start_age = st.number_input(
                "Alter bei Entnahmebeginn",
                min_value=30,
                max_value=90,
                value=65,
                step=1,
                key="withdrawal_start_age",
                help="In welchem Alter beginnen Sie mit der Entnahme?"
            )

        with col2:
            end_age = st.number_input(
                "Geplantes Endalter",
                min_value=50,
                max_value=110,
                value=95,
                step=1,
                key="withdrawal_end_age",
                help="Bis zu welchem Alter soll das Geld reichen?"
            )

        with col3:
            withdrawal_years = end_age - start_age
            if withdrawal_years < 1:
                withdrawal_years = 1
                st.error("Endalter muss größer als Startalter sein!")
            st.metric("Entnahmedauer", f"{withdrawal_years} Jahre", help="Automatisch berechnet")

        st.markdown("---")

        # Financial inputs
        col1, col2, col3 = st.columns(3)

        with col1:
            withdrawal_initial_str = st.text_input(
                "Anfangsvermögen (€)",
                value=format_number(500000),
                key="withdrawal_initial",
                help="z.B. 500.000"
            )
            withdrawal_initial = parse_german_number(withdrawal_initial_str, 500000)
            withdrawal_initial = max(10000, min(100000000, withdrawal_initial))

        with col2:
            monthly_withdrawal_str = st.text_input(
                "Monatliche Entnahme (€)",
                value=format_number(2000),
                key="monthly_withdrawal",
                help="z.B. 2.000"
            )
            monthly_withdrawal = parse_german_number(monthly_withdrawal_str, 2000)
            monthly_withdrawal = max(100, min(100000, monthly_withdrawal))

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
            help="Erhöht die Entnahme jährlich um die Inflationsrate, um die Kaufkraft zu erhalten",
            key="adjust_for_inflation"
        )

        # Steuer-Optionen
        st.subheader("💶 Steuern bei Entnahme")

        apply_withdrawal_tax = st.checkbox(
            "Kapitalertragssteuer berücksichtigen",
            value=False,
            help="Berechnet KESt auf realisierte Gewinne bei Entnahmen. "
                 "Bei jeder Entnahme wird der Gewinnanteil versteuert.",
            key="apply_withdrawal_tax"
        )

        if apply_withdrawal_tax:
            withdrawal_tax_rate = st.slider(
                "Steuersatz (%)",
                min_value=0.0,
                max_value=50.0,
                value=27.5,
                step=0.5,
                help="Österreichische KESt: 27,5%",
                key="withdrawal_tax_rate"
            ) / 100
        else:
            withdrawal_tax_rate = None

        # Calculate withdrawal rate
        annual_withdrawal = monthly_withdrawal * 12
        withdrawal_rate = annual_withdrawal / withdrawal_initial * 100

        st.info(f"**Entnahmerate**: {withdrawal_rate:.1f}% p.a. (Die '4%-Regel' gilt als konservativ)")

        # Simulation Button
        if st.button("🏦 Entnahme simulieren", use_container_width=True, key="run_withdrawal_sim"):
            # Scroll to top so user sees progress
            scroll_to_top()

            # Sofortige Benachrichtigung
            st.toast("🏦 Simulation gestartet!", icon="🚀")

            # Prominenter Fortschrittscontainer
            with st.status("🔄 **Entnahme-Simulation läuft...**", expanded=True) as status:
                st.write("⏳ Initialisiere 5.000 Simulationen...")
                simulator = WithdrawalSimulator(n_simulations=5000)

                st.write(f"📊 Simuliere {withdrawal_years} Jahre Entnahmephase...")
                withdrawal_results = simulator.simulate(
                    initial_value=withdrawal_initial,
                    monthly_withdrawal=monthly_withdrawal,
                    expected_annual_return=withdrawal_return,
                    annual_volatility=withdrawal_volatility,
                    years=withdrawal_years,
                    inflation_rate=withdrawal_inflation,
                    adjust_for_inflation=adjust_for_inflation,
                    tax_rate=withdrawal_tax_rate,
                    apply_tax_to_gains=apply_withdrawal_tax
                )

                st.write("✅ Analyse abgeschlossen!")
                st.session_state.withdrawal_results = withdrawal_results
                status.update(label="✅ **Simulation abgeschlossen!**", state="complete", expanded=False)

            st.toast("✅ Ergebnisse bereit!", icon="✅")

        # Display results
        if st.session_state.withdrawal_results is not None:
            wr = st.session_state.withdrawal_results

            st.markdown("---")
            st.subheader(f"Ergebnisse ({wr.n_simulations:,} Simulationen)".replace(",", "."))

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
                    st.metric("Median Endwert", format_currency(wr.median_final_value), help="Der mittlere Vermögenswert am Ende des Entnahmezeitraums. 50% der Simulationen enden mit mehr, 50% mit weniger. Bei 0€ bedeutet das: In der Hälfte der Fälle ist das Geld aufgebraucht.")
                with cols[1]:
                    st.metric("Gesamtentnahme (Median)", format_currency(wr.total_withdrawn_median), help="Die Summe aller Entnahmen im mittleren Szenario. Zeigt, wie viel Sie insgesamt entnehmen konnten – je höher, desto besser hat die Strategie funktioniert.")
                with cols[2]:
                    if wr.earliest_depletion:
                        depletion_years = wr.earliest_depletion / 12
                        depletion_age = start_age + depletion_years
                        st.metric(
                            "Früheste Erschöpfung",
                            f"mit {depletion_age:.0f} Jahren",
                            f"nach {depletion_years:.1f} Jahren",
                            help="Das Worst-Case-Szenario: In mindestens einer Simulation war das Geld zu diesem Zeitpunkt aufgebraucht. Wichtig für die Planung eines Sicherheitspuffers."
                        )
                    else:
                        st.metric("Früheste Erschöpfung", "Nie", f"Geld reicht bis {end_age}", help="Gute Nachricht: In keiner Simulation war das Geld vor dem geplanten Ende aufgebraucht.")

            # Steuer-Übersicht (wenn aktiviert)
            if wr.tax_results is not None:
                st.markdown("---")
                st.subheader("💶 Steuerübersicht")

                tax_cols = st.columns(4)
                with tax_cols[0]:
                    st.metric(
                        "Brutto entnommen (Ø)",
                        format_currency(wr.tax_results.mean_gross_withdrawn),
                        help="Durchschnittliche Gesamtsumme der Brutto-Entnahmen über alle Simulationen"
                    )
                with tax_cols[1]:
                    st.metric(
                        "Steuern (KESt) (Ø)",
                        format_currency(wr.tax_results.mean_taxes_paid),
                        help="Durchschnittlich gezahlte Kapitalertragssteuer über alle Simulationen"
                    )
                with tax_cols[2]:
                    st.metric(
                        "Netto erhalten (Ø)",
                        format_currency(wr.tax_results.mean_net_withdrawn),
                        help="Durchschnittliche Netto-Entnahme nach Abzug der Steuern"
                    )
                with tax_cols[3]:
                    st.metric(
                        "Effektiver Steuersatz",
                        f"{wr.tax_results.effective_tax_rate:.1%}",
                        help=f"Tatsächlicher Steuersatz: Gezahlte Steuern / Realisierte Gewinne. "
                             f"Nomineller Satz: {wr.tax_results.tax_rate:.1%}"
                    )

                # Zusätzliche Info
                st.info(
                    f"📊 **Realisierte Gewinne (Ø)**: {format_currency(wr.tax_results.mean_realized_gains)} — "
                    f"Davon {wr.tax_results.tax_rate:.1%} KESt = {format_currency(wr.tax_results.mean_taxes_paid)}"
                )

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
                    st.caption(f"Wann geht das Geld aus? (Startalter: {start_age})")
                    fig_depletion = plot_depletion_histogram(wr)
                    st.plotly_chart(fig_depletion, use_container_width=True)
                else:
                    st.success(f"In allen Simulationen reicht das Vermögen bis zum Alter von {end_age} Jahren!")

            # SWR Calculator
            st.markdown("---")
            st.subheader("💡 Sichere Entnahmerate berechnen")

            target_success_pct = st.slider(
                "Gewünschte Erfolgsquote",
                min_value=80,
                max_value=99,
                value=95,
                step=1,
                format="%d%%",
                key="target_success"
            )
            target_success = target_success_pct / 100.0

            # SWR Button
            if st.button("Sichere Entnahmerate berechnen", use_container_width=True, key="run_swr_calc"):
                # Scroll to top so user sees progress
                scroll_to_top()

                # Sofortige Benachrichtigung
                st.toast("💡 SWR-Berechnung gestartet!", icon="🚀")

                # Prominenter Fortschrittscontainer
                with st.status("🔄 **Berechne sichere Entnahmerate...**", expanded=True) as status:
                    st.write("⏳ Initialisiere binäre Suche...")
                    simulator = WithdrawalSimulator(n_simulations=3000)

                    st.write(f"📊 Optimiere für {target_success*100:.0f}% Erfolgsrate...")
                    swr_result = simulator.find_safe_withdrawal_rate(
                        initial_value=withdrawal_initial,
                        expected_annual_return=withdrawal_return,
                        annual_volatility=withdrawal_volatility,
                        years=withdrawal_years,
                        target_success_rate=target_success,
                        inflation_rate=withdrawal_inflation,
                        adjust_for_inflation=adjust_for_inflation,
                        tax_rate=withdrawal_tax_rate,
                        apply_tax_to_gains=apply_withdrawal_tax
                    )

                    st.write("✅ Optimale Rate gefunden!")
                    st.session_state.swr_result = swr_result
                    st.session_state.swr_params = {
                        'target_success': target_success,
                        'start_age': start_age,
                        'end_age': end_age,
                        'withdrawal_years': withdrawal_years
                    }
                    status.update(label="✅ **Berechnung abgeschlossen!**", state="complete", expanded=False)

                st.toast("✅ Ergebnis bereit!", icon="✅")

            # Display SWR result if available
            if 'swr_result' in st.session_state and st.session_state.swr_result is not None:
                swr = st.session_state.swr_result
                params = st.session_state.swr_params
                st.success(f"""
                **Ergebnis für {params['target_success']*100:.0f}% Erfolgswahrscheinlichkeit:**

                - 👤 **Zeitraum**: Von {params['start_age']} bis {params['end_age']} Jahren ({params['withdrawal_years']} Jahre)
                - 💰 **Sichere monatliche Entnahme**: {format_currency(swr['monthly_withdrawal'])}
                - 📊 **Sichere Entnahmerate (SWR)**: {swr['withdrawal_rate_pct']:.2f}% p.a.
                - 📅 **Jährliche Entnahme**: {format_currency(swr['annual_withdrawal'])}
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
                st.metric("Erwartete Rendite", f"{current_return*100:.1f}%", help="Die geschätzte jährliche Rendite basierend auf historischen Daten.")
                st.metric("Volatilität", f"{current_vol*100:.1f}%", help="Die jährliche Schwankungsbreite – je höher, desto riskanter.")
                current_sharpe = (current_return - risk_free_rate) / current_vol if current_vol > 0 else 0
                st.metric("Sharpe Ratio", f"{current_sharpe:.2f}", help="Rendite pro Risikoeinheit. Vergleichen Sie mit den optimalen Portfolios – liegt Ihr Wert deutlich darunter, gibt es Verbesserungspotential.")

                st.markdown("**Gewichtung:**")
                for ticker, weight in zip(portfolio.tickers, portfolio.weights):
                    if weight > 0.001:
                        st.write(f"- {ticker}: {weight*100:.1f}%")

            with col2:
                st.markdown("### ⭐ Max Sharpe Portfolio")
                st.caption("Beste Rendite pro Risikoeinheit")
                max_sharpe = frontier_result.max_sharpe_portfolio
                st.metric("Erwartete Rendite", f"{max_sharpe.expected_return*100:.1f}%")
                st.metric("Volatilität", f"{max_sharpe.volatility*100:.1f}%")
                st.metric("Sharpe Ratio", f"{max_sharpe.sharpe_ratio:.2f}", help="Das theoretisch optimale Portfolio: Maximale Rendite für das eingegangene Risiko.")

                st.markdown("**Optimale Gewichtung:**")
                for ticker, weight in max_sharpe.get_weights_dict().items():
                    if weight > 0.001:
                        st.write(f"- {ticker}: {weight*100:.1f}%")

            with col3:
                st.markdown("### 💎 Min Volatilität Portfolio")
                st.caption("Geringstes Risiko")
                min_vol = frontier_result.min_volatility_portfolio
                st.metric("Erwartete Rendite", f"{min_vol.expected_return*100:.1f}%")
                st.metric("Volatilität", f"{min_vol.volatility*100:.1f}%", help="Die niedrigste erreichbare Volatilität mit den gewählten Anlagen.")
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
            # Get simulation count from first scenario result
            first_result = next(iter(scenario_results.values()))
            st.caption(f"Basierend auf {first_result.num_simulations:,} Simulationen pro Szenario".replace(",", "."))
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
                metrics=metrics,
                tax_cost_results=st.session_state.get('tax_cost_results')
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
                initial_value=initial_value,
                tax_cost_results=st.session_state.get('tax_cost_results')
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

    # TAB 8: Taxes & Costs
    with tab8:
        st.header("💶 Steuern & Transaktionskosten")

        tax_cost_results = st.session_state.get('tax_cost_results')

        if tax_cost_results is not None:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Endwert vor Steuern",
                    format_currency(tax_cost_results.mean_final_before_tax),
                    help="Durchschnittlicher Endwert ohne Abzüge"
                )

            with col2:
                delta_pct = -tax_cost_results.total_cost_impact * 100
                st.metric(
                    "Endwert nach Steuern",
                    format_currency(tax_cost_results.mean_final_after_tax),
                    f"{delta_pct:.1f}%",
                    help="Nach Abzug aller Steuern und Kosten"
                )

            with col3:
                st.metric(
                    "Ø Gezahlte Steuern",
                    format_currency(tax_cost_results.mean_taxes_paid),
                    help="Durchschnittliche KESt"
                )

            with col4:
                st.metric(
                    "Ø Transaktionskosten",
                    format_currency(tax_cost_results.mean_transaction_costs),
                    help="Durchschnittliche Broker-Gebühren"
                )

            st.markdown("---")

            # Detailed breakdown
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Detaillierte Aufschlüsselung")
                breakdown_data = {
                    'Metrik': [
                        'Anfangskapital',
                        'Endwert (brutto)',
                        'Realisierte Gewinne (Ø)',
                        'Steuern (KESt) (Ø)',
                        'Transaktionskosten (Ø)',
                        'Endwert nach Steuern (Ø)',
                        'Unrealisierte Gewinne (Ø)',
                        'Rebalancing-Events (Ø)',
                        'Effektiver Steuersatz'
                    ],
                    'Wert': [
                        format_currency(initial_value),
                        format_currency(tax_cost_results.mean_final_before_tax),
                        format_currency(tax_cost_results.mean_realized_gains),
                        format_currency(tax_cost_results.mean_taxes_paid),
                        format_currency(tax_cost_results.mean_transaction_costs),
                        format_currency(tax_cost_results.mean_final_after_tax),
                        format_currency(tax_cost_results.mean_unrealized_gains),
                        f"{tax_cost_results.mean_rebalancing_events:.1f}",
                        f"{tax_cost_results.effective_tax_rate * 100:.1f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

            with col2:
                st.subheader("📈 Perzentile nach Steuern")
                percentile_data = {
                    'Perzentil': ['5%', '25%', '50% (Median)', '75%', '95%'],
                    'Endwert': [
                        format_currency(tax_cost_results.get_percentile_after_tax(5)),
                        format_currency(tax_cost_results.get_percentile_after_tax(25)),
                        format_currency(tax_cost_results.median_final_after_tax),
                        format_currency(tax_cost_results.get_percentile_after_tax(75)),
                        format_currency(tax_cost_results.get_percentile_after_tax(95))
                    ]
                }
                st.dataframe(pd.DataFrame(percentile_data), use_container_width=True, hide_index=True)

            st.markdown("---")

            # Comparison chart
            st.subheader("📊 Vergleich: Vor vs. Nach Steuern")
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Vor Steuern',
                x=['Durchschnitt', 'Median'],
                y=[tax_cost_results.mean_final_before_tax, np.median(tax_cost_results.final_value_before_tax)],
                marker_color='#2E86AB'
            ))
            fig.add_trace(go.Bar(
                name='Nach Steuern',
                x=['Durchschnitt', 'Median'],
                y=[tax_cost_results.mean_final_after_tax, tax_cost_results.median_final_after_tax],
                marker_color='#A23B72'
            ))
            fig.update_layout(
                barmode='group',
                title='Endwert-Vergleich',
                yaxis_title='Endwert (€)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Info section
            with st.expander("ℹ️ Erklärungen"):
                st.markdown(f"""
                ### Österreichische KESt (Kapitalertragssteuer)
                Die **KESt** beträgt **{tax_cost_results.tax_rate * 100:.1f}%** auf realisierte Kapitalerträge.

                **Wann wird Steuer fällig?**
                - Bei jedem **Rebalancing** werden Gewinne realisiert
                - Steuer nur auf **positive Gewinne** (keine Steuer bei Verlust)
                - **Unrealisierte Gewinne** erst bei Verkauf besteuert

                ### Tipps zur Steueroptimierung
                1. **Weniger Rebalancing** = weniger realisierte Gewinne
                2. **Threshold-Rebalancing** statt zeitbasiert
                3. **Buy & Hold** vermeidet Steuern bis zum Verkauf
                """)
        else:
            if not enable_tax_costs:
                st.info("👈 Aktivieren Sie 'Steuern & Transaktionskosten' in der Seitenleiste.")
            elif rebalancing_option == "Kein Rebalancing (Buy & Hold)":
                st.info("📊 Bei Buy & Hold entstehen keine realisierten Gewinne. Wählen Sie eine Rebalancing-Strategie.")
            else:
                st.warning("Keine Steuer-/Kostendaten verfügbar. Bitte Simulation erneut starten.")

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

    # What's New / Updates Section
    st.markdown("---")
    with st.expander("🆕 **Was ist neu?** - Aktuelle Updates", expanded=False):
        st.markdown("""
        ### Version 1.2.0 (Februar 2025)

        #### 💶 NEU: Steuern bei Entnahme
        - **KESt-Berechnung im Entnahmeplan**: Bei jeder Entnahme wird der Gewinnanteil versteuert
        - Neuer Abschnitt "Steuerübersicht" mit Brutto/Netto-Entnahmen
        - Effektiver Steuersatz über die gesamte Entnahmephase
        - SWR-Berechnung berücksichtigt nun auch Steuern

        ---

        ### Version 1.1.0 (Februar 2025)

        #### 💶 Steuer- und Kostenrechner bei Rebalancing
        - **Österreichische KESt (27,5%)** auf realisierte Gewinne bei Rebalancing
        - **Transaktionskosten**: Prozentual (0,1%) oder Flat Fee (€5 pro Trade)
        - Neuer Tab "Steuern & Kosten" mit detaillierter Aufschlüsselung
        - Vergleich: Endwert vor vs. nach Steuern
        - Cost-Basis-Tracking für korrekte Gewinnberechnung

        ---

        ### Version 1.0.1 (Februar 2025)

        #### 🔧 Verbesserungen
        - Hilfetexte für Rebalancing-Strategien hinzugefügt
        - VaR Konfidenzlevel-Erklärungen verbessert
        - Copyright-Hinweis im Header

        ---

        ### Version 1.0.0 (Januar 2025)

        #### 📊 Kern-Features
        - Monte Carlo Simulation mit bis zu 50.000 Pfaden
        - Rebalancing-Strategien (Buy & Hold, Monatlich, Quartalsweise, Threshold)
        - VaR/CVaR Risikoanalyse mit konfigurierbarem Konfidenzlevel
        - Sharpe Ratio, Sortino Ratio, Max Drawdown

        #### 📈 Benchmark & Analyse
        - Benchmark-Vergleich (S&P 500, DAX, Euro Stoxx 50, etc.)
        - Beta, Alpha, Information Ratio
        - Efficient Frontier Portfolio-Optimierung
        - Korrelationsmatrix

        #### 💰 Planungs-Tools
        - **Sparplan-Simulation**: Monatliche Einzahlungen simulieren
        - **Entnahme-Simulation**: Ruhestandsplanung mit Inflationsanpassung
        - **Safe Withdrawal Rate Calculator**: Optimale Entnahmerate berechnen

        #### 🎭 Szenarien & Stress-Tests
        - Bullenmarkt, Bärenmarkt, Crash, Stagflation
        - Hohe Volatilität Szenarien

        #### 📱 UI/UX
        - Mobile-optimiert (iPhone/iPad)
        - Responsive Design
        - Deutschsprachige Oberfläche
        - Excel/CSV Export

        ---

        ### Geplante Features 🚀
        - 📊 Backtesting
        - 🔄 Eigene Szenarien erstellen
        - 📄 PDF-Export
        """)

    # Code Statistics Section
    with st.expander("📊 **Code-Statistik** - Projektumfang", expanded=False):
        code_stats = get_code_statistics()
        st.markdown(f"""
        ### Projektumfang

        | Bereich | Zeilen | Anteil |
        |---------|--------|--------|
        | **app.py** (Hauptanwendung) | {code_stats['app']:,} | {code_stats['app']/code_stats['total']*100:.0f}% |
        | **src/** (Kernmodule) | {code_stats['src']:,} | {code_stats['src']/code_stats['total']*100:.0f}% |
        | **tests/** (Tests) | {code_stats['tests']:,} | {code_stats['tests']/code_stats['total']*100:.0f}% |
        | **Gesamt** | **{code_stats['total']:,}** | 100% |

        *{code_stats['files']} Python-Dateien*
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
