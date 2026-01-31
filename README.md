# Portfolio Monte Carlo Simulation

Eine interaktive Web-Applikation zur Monte Carlo Simulation von Investment-Portfolios.

## Features

- **Multi-Asset Portfolio**: Beliebig viele Assets mit automatischer Korrelationsberechnung
- **Historische Marktdaten**: Integration von Yahoo Finance
- **Risikometriken**: VaR, CVaR, Sharpe Ratio, Sortino Ratio, Max Drawdown
- **Rebalancing-Strategien**: Buy & Hold, Periodisch, Threshold, Hybrid
- **Interaktive Visualisierungen**: Simulationspfade, VaR-Kegel, Korrelations-Heatmaps

## Installation

```bash
# Repository klonen
git clone https://github.com/DEIN-USERNAME/portfolio-monte-carlo.git
cd portfolio-monte-carlo

# Dependencies installieren
pip install -r requirements.txt

# App starten
streamlit run app.py
```

Die App öffnet sich unter `http://localhost:8501`

## Verwendung

1. **Portfolio definieren**: Ticker-Symbole eingeben (z.B. AAPL, MSFT, GOOGL)
2. **Gewichtungen festlegen**: Kapital auf die Assets verteilen
3. **Simulation konfigurieren**: Zeithorizont und Anzahl Simulationen wählen
4. **Rebalancing wählen**: Strategie auswählen
5. **Analyse starten**: "Simulation starten" klicken

## Projektstruktur

```
portfolio-monte-carlo/
├── app.py                      # Streamlit Web-App
├── requirements.txt            # Dependencies
├── src/
│   ├── simulation/
│   │   └── monte_carlo.py      # Monte Carlo Engine
│   ├── portfolio/
│   │   ├── portfolio.py        # Portfolio & Asset Klassen
│   │   └── rebalancing.py      # Rebalancing-Strategien
│   ├── data/
│   │   └── market_data.py      # Yahoo Finance Integration
│   ├── risk/
│   │   ├── var.py              # VaR, CVaR
│   │   └── metrics.py          # Sharpe, Sortino, etc.
│   └── visualization/
│       └── charts.py           # Plotly Charts
└── tests/
    └── test_simulation.py      # Unit Tests
```

## Technologie-Stack

- **Python 3.11+**
- **Streamlit** - Web Framework
- **NumPy / Pandas** - Datenanalyse
- **yfinance** - Marktdaten
- **Plotly** - Visualisierungen
- **SciPy** - Statistik

## Lizenz

MIT
