# Portfolio Monte Carlo Simulation

Eine interaktive Web-Applikation zur Monte Carlo Simulation von Investment-Portfolios.

**Erdacht von Markus O. Thalhamer** ([mthalhamer@thalhamer.com](mailto:mthalhamer@thalhamer.com)) mit Unterstützung von **Claude**.

## Features

- **Multi-Asset Portfolio**: Beliebig viele Assets mit automatischer Korrelationsberechnung
- **Historische Marktdaten**: Integration von Yahoo Finance
- **Risikometriken**: VaR, CVaR, Sharpe Ratio, Sortino Ratio, Max Drawdown, Beta, Alpha
- **Rebalancing-Strategien**: Buy & Hold, Periodisch, Threshold
- **Benchmark-Vergleich**: Portfolio vs. S&P 500, DAX, etc.
- **Sparplan-Simulation**: Monatliche Einzahlungen simulieren
- **Szenario-Analyse**: Bull/Bear/Crash-Szenarien testen
- **Export**: Excel und CSV Reports
- **Interaktive Visualisierungen**: Simulationspfade, VaR-Kegel, Korrelations-Heatmaps
- **Einführung für Laien**: Ausführliche Erklärungen aller Konzepte

## Demo

Die App läuft auf Streamlit Cloud: [Link folgt nach Deployment]

## Installation

```bash
# Repository klonen
git clone https://github.com/MarkusA1A/portfolio-monte-carlo.git
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
│   │   ├── monte_carlo.py      # Monte Carlo Engine
│   │   ├── savings_plan.py     # Sparplan-Simulation
│   │   └── scenarios.py        # Szenario-Analyse
│   ├── portfolio/
│   │   ├── portfolio.py        # Portfolio & Asset Klassen
│   │   └── rebalancing.py      # Rebalancing-Strategien
│   ├── data/
│   │   └── market_data.py      # Yahoo Finance Integration
│   ├── risk/
│   │   ├── var.py              # VaR, CVaR
│   │   └── metrics.py          # Sharpe, Sortino, Beta, Alpha
│   ├── visualization/
│   │   └── charts.py           # Plotly Charts
│   └── export/
│       └── reports.py          # Excel/CSV Export
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
- **openpyxl** - Excel Export

## Hinweis

Diese App dient ausschließlich zu Bildungs- und Planungszwecken. Sie stellt keine Anlageberatung dar. Investitionsentscheidungen sollten immer unter Berücksichtigung der persönlichen finanziellen Situation und ggf. mit professioneller Beratung getroffen werden.

## Lizenz

MIT

## Autor

**Markus O. Thalhamer**
Email: [mthalhamer@thalhamer.com](mailto:mthalhamer@thalhamer.com)

Entwickelt mit Unterstützung von **Claude** (Anthropic).
