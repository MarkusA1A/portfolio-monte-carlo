"""
Dividend Screener - Findet Aktien mit nachhaltigen und wachsenden Dividenden.

Nutzt yfinance EquityQuery für Server-seitiges Screening und ergänzt
die Ergebnisse mit detaillierten Dividenden-Historien und Qualitäts-Scores.
"""
from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import date
import logging
import time
import numpy as np
import pandas as pd
import yfinance as yf


logger = logging.getLogger(__name__)

# yfinance screen() unterstützt maximal 250 Ergebnisse pro Anfrage
_YF_SCREEN_MAX = 250

# Cache-TTL: Einträge älter als 1 Stunde werden als veraltet betrachtet
_CACHE_TTL_SECONDS = 3600

# Retry-Konfiguration für API-Aufrufe
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 0.5  # Sekunden, wird exponentiell erhöht


# Exchange-Mapping für die UI
EXCHANGE_OPTIONS = {
    "USA (NYSE)": "NYQ",
    "USA (NASDAQ)": "NMS",
    "Deutschland (XETRA)": "GER",
    "London (LSE)": "LSE",
    "Schweiz (SIX)": "EBS",
    "Alle": None,
}


@dataclass
class ScreenerFilter:
    """Filter-Kriterien für das Dividenden-Screening."""
    min_dividend_yield: float = 1.0       # Minimum Rendite %
    max_dividend_yield: float = 10.0      # Maximum Rendite % (sehr hoch = Falle)
    max_payout_ratio: float = 0.70        # Max. Ausschüttungsquote (0-1)
    min_consecutive_years: int = 5        # Min. Jahre Dividendensteigerung
    min_dividend_growth_rate: float = 0.0 # Min. 5J-CAGR %
    min_market_cap: float = 1e10          # Min. Marktkapitalisierung ($10B)
    exchange: Optional[str] = "NYQ"       # Börse (None = Alle)
    max_results: int = 25                 # Max. Ergebnisse


@dataclass
class DividendMetrics:
    """Dividenden-Kennzahlen für eine einzelne Aktie."""
    ticker: str
    name: str
    sector: str
    current_price: float
    market_cap: float
    currency: str
    # Dividenden-Kennzahlen
    dividend_yield: float              # Aktuelle Forward-Dividendenrendite (%)
    trailing_dividend_yield: float     # Trailing 12M Rendite (%)
    dividend_rate: float               # Jährliche Dividende pro Aktie
    payout_ratio: float                # Ausschüttungsquote (0-1)
    consecutive_years_increase: int    # Jahre konsekutiver Steigerung
    dividend_growth_rate_5y: float     # 5-Jahres CAGR (%)
    five_year_avg_yield: float         # 5-Jahres-Durchschnittsrendite (%)
    # Qualitäts-Kennzahlen
    return_on_equity: float            # ROE (0-1)
    debt_to_equity: float              # Verschuldungsgrad
    free_cashflow: float               # Freier Cashflow
    # Berechnet
    quality_score: float = 0.0         # Qualitätsscore 0-100
    dividend_history: Optional[pd.Series] = None  # Gespeicherte Dividenden-Historie


@dataclass
class _CacheEntry:
    """Cache-Eintrag mit Zeitstempel für TTL-basierte Invalidierung."""
    metrics: DividendMetrics
    timestamp: float  # time.monotonic()


def _safe(val, default=0.0):
    """Gibt val zurück falls nicht None, sonst default. Bewahrt valide 0.0 Werte."""
    return val if val is not None else default


class DividendScreener:
    """
    Screent Aktien nach nachhaltigen und wachsenden Dividenden.

    Nutzt yf.screen() für initiales Server-seitiges Filtern,
    dann Anreicherung mit detaillierten Dividenden-Historien.
    """

    def __init__(self, cache_ttl: int = _CACHE_TTL_SECONDS):
        self._cache: dict[str, _CacheEntry] = {}
        self._cache_ttl = cache_ttl

    def _get_cached(self, ticker: str) -> Optional[DividendMetrics]:
        """Gibt gecachte Metrics zurück, falls vorhanden und nicht abgelaufen."""
        entry = self._cache.get(ticker)
        if entry is None:
            return None
        age = time.monotonic() - entry.timestamp
        if age > self._cache_ttl:
            del self._cache[ticker]
            return None
        return entry.metrics

    def _set_cached(self, ticker: str, metrics: DividendMetrics) -> None:
        """Speichert Metrics mit aktuellem Zeitstempel im Cache."""
        self._cache[ticker] = _CacheEntry(
            metrics=metrics,
            timestamp=time.monotonic()
        )

    def screen(
        self,
        filters: ScreenerFilter,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> list[DividendMetrics]:
        """
        Führt das Dividenden-Screening mit den gegebenen Filtern durch.

        Args:
            filters: ScreenerFilter mit Kriterien
            progress_callback: Optionaler Callback(prozent, text) für UI-Fortschritt

        Returns:
            Liste von DividendMetrics, sortiert nach quality_score absteigend
        """
        if progress_callback:
            progress_callback(10, "Baue Suchanfrage...")

        query = self._build_screen_query(filters)

        if progress_callback:
            progress_callback(20, "Suche Dividenden-Aktien...")

        requested_size = filters.max_results * 2
        quotes = self._fetch_screen_results(query, requested_size)

        if not quotes:
            return []

        if len(quotes) >= _YF_SCREEN_MAX:
            logger.warning(
                "yfinance screen() Limit von %d erreicht – "
                "möglicherweise wurden nicht alle passenden Aktien zurückgegeben. "
                "Bitte Filter einschränken.", _YF_SCREEN_MAX
            )

        if progress_callback:
            progress_callback(40, f"{len(quotes)} Kandidaten gefunden, lade Details...")

        # Enrich top candidates with detailed data
        results = []
        failed_tickers = []
        total = len(quotes)
        for i, quote in enumerate(quotes):
            ticker_symbol = quote.get('symbol', '')
            if not ticker_symbol:
                continue

            # Check cache (TTL-aware)
            cached = self._get_cached(ticker_symbol)
            if cached is not None:
                results.append(cached)
                continue

            detailed = self._enrich_with_detailed_data(ticker_symbol)
            metrics = self._build_dividend_metrics(quote, detailed)

            if metrics is not None:
                metrics.quality_score = self._calculate_quality_score(metrics)
                self._set_cached(ticker_symbol, metrics)
                results.append(metrics)
            else:
                failed_tickers.append(ticker_symbol)

            if progress_callback:
                pct = 40 + int(50 * (i + 1) / total)
                progress_callback(pct, f"Analysiere {ticker_symbol} ({i+1}/{total})...")

            # Rate limiting zwischen API-Aufrufen (nicht nach Cache-Hits)
            time.sleep(0.15)

        if failed_tickers:
            logger.info(
                "%d Ticker konnten nicht angereichert werden: %s",
                len(failed_tickers), ", ".join(failed_tickers[:10])
            )

        # Post-filter (payout ratio, growth rate need detailed data)
        results = self._apply_post_filters(results, filters)

        # Sort by quality score descending
        results.sort(key=lambda m: m.quality_score, reverse=True)

        # Limit results
        results = results[:filters.max_results]

        if progress_callback:
            progress_callback(100, f"{len(results)} Dividenden-Aktien gefunden.")

        return results

    def _build_screen_query(self, filters: ScreenerFilter) -> yf.EquityQuery:
        """Baut yfinance EquityQuery aus ScreenerFilter."""
        conditions = [
            yf.EquityQuery('gt', ['forward_dividend_yield', filters.min_dividend_yield]),
        ]

        if filters.max_dividend_yield < 100:
            conditions.append(
                yf.EquityQuery('lt', ['forward_dividend_yield', filters.max_dividend_yield])
            )

        if filters.min_market_cap > 0:
            conditions.append(
                yf.EquityQuery('gt', ['intradaymarketcap', filters.min_market_cap])
            )

        if filters.exchange:
            conditions.append(
                yf.EquityQuery('is-in', ['exchange', filters.exchange])
            )

        if len(conditions) == 1:
            return conditions[0]
        return yf.EquityQuery('and', conditions)

    def _fetch_screen_results(
        self,
        query: yf.EquityQuery,
        max_results: int
    ) -> list[dict]:
        """Führt Screen-Query aus und gibt rohe Quote-Dicts zurück."""
        try:
            result = yf.screen(
                query,
                size=min(max_results, _YF_SCREEN_MAX),
                sortField='forward_dividend_yield',
                sortAsc=False
            )
            return result.get('quotes', [])
        except Exception as e:
            logger.error("yf.screen() fehlgeschlagen: %s", e)
            return []

    def _enrich_with_detailed_data(
        self,
        ticker_symbol: str
    ) -> Optional[dict]:
        """
        Lädt detaillierte Dividenden-Daten für einen einzelnen Ticker.
        Nutzt exponentielles Backoff bei Fehlern (max. 3 Versuche).
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info or {}
                dividends = ticker.dividends
                if dividends is None:
                    dividends = pd.Series(dtype=float)
                return {
                    'info': info,
                    'dividends': dividends
                }
            except Exception as e:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "Versuch %d/%d für %s fehlgeschlagen: %s – Retry in %.1fs",
                        attempt, _MAX_RETRIES, ticker_symbol, e, delay
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Alle %d Versuche für %s fehlgeschlagen: %s",
                        _MAX_RETRIES, ticker_symbol, e
                    )
        return None

    @staticmethod
    def _get_complete_annual_dividends(dividends: pd.Series) -> pd.Series:
        """
        Gruppiert Dividenden nach Jahr und schließt das unvollständige
        aktuelle Jahr aus, um falsche Vergleiche zu vermeiden.
        """
        if dividends.empty:
            return pd.Series(dtype=float)
        annual = dividends.groupby(dividends.index.year).sum()
        current_year = date.today().year
        if current_year in annual.index:
            annual = annual.drop(current_year)
        return annual

    def _calculate_dividend_growth_rate(
        self,
        dividends: pd.Series,
        years: int = 5
    ) -> float:
        """
        Berechnet den Dividenden-CAGR über die gegebenen Jahre.

        Gruppiert Dividenden nach Jahr, summiert jährlich,
        und berechnet die durchschnittliche jährliche Wachstumsrate.
        Schließt das unvollständige aktuelle Jahr aus.
        """
        if dividends.empty:
            return 0.0

        annual = self._get_complete_annual_dividends(dividends)

        if len(annual) < 2:
            return 0.0

        n_years = min(years, len(annual) - 1)
        if n_years < 1:
            return 0.0

        end_div = annual.iloc[-1]
        start_div = annual.iloc[-(n_years + 1)]

        if start_div <= 0 or end_div <= 0:
            return 0.0

        cagr = (end_div / start_div) ** (1 / n_years) - 1
        return cagr * 100  # Als Prozent

    def _calculate_consecutive_increases(
        self,
        dividends: pd.Series
    ) -> int:
        """
        Berechnet konsekutive Jahre mit Dividendensteigerungen,
        vom aktuellsten Jahr rückwärts gezählt.
        Schließt das unvollständige aktuelle Jahr aus.
        """
        if dividends.empty:
            return 0

        annual = self._get_complete_annual_dividends(dividends)

        if len(annual) < 2:
            return 0

        consecutive = 0
        for i in range(len(annual) - 1, 0, -1):
            if annual.iloc[i] > annual.iloc[i - 1]:
                consecutive += 1
            else:
                break
        return consecutive

    def _calculate_quality_score(self, metrics: DividendMetrics) -> float:
        """
        Berechnet den gewichteten Qualitätsscore (0-100).

        Komponenten:
        - Konsekutive Jahre Steigerung (25%)
        - Dividenden-Wachstumsrate 5J (25%)
        - Ausschüttungsquote Nachhaltigkeit (20%)
        - Return on Equity (15%)
        - Debt-to-Equity (15%)
        """
        score = 0.0

        # 1. Konsekutive Jahre (25 Punkte)
        years = metrics.consecutive_years_increase
        if years >= 25:
            s = 25.0
        elif years >= 10:
            s = 15.0 + (years - 10) / 15 * 10
        elif years >= 5:
            s = 8.0 + (years - 5) / 5 * 7
        else:
            s = years / 5 * 8
        score += s

        # 2. Dividenden-Wachstumsrate (25 Punkte)
        growth = metrics.dividend_growth_rate_5y
        if growth > 10:
            s = 25.0
        elif growth > 5:
            s = 15.0 + (growth - 5) / 5 * 10
        elif growth > 0:
            s = growth / 5 * 15
        else:
            s = 0.0
        score += s

        # 3. Ausschüttungsquote (20 Punkte) - Sweet Spot 30-60%
        pr = max(0.0, metrics.payout_ratio)
        if 0.30 <= pr <= 0.60:
            s = 20.0
        elif pr < 0.30:
            s = pr / 0.30 * 18
        elif pr <= 0.80:
            s = 20.0 - (pr - 0.60) / 0.20 * 12
        elif pr <= 1.0:
            s = 8.0 - (pr - 0.80) / 0.20 * 8
        else:
            s = 0.0
        score += s

        # 4. Return on Equity (15 Punkte)
        roe = metrics.return_on_equity
        if 0.12 <= roe <= 0.25:
            s = 15.0
        elif roe > 0.25:
            s = 12.0  # Sehr hoher ROE kann Leverage anzeigen
        elif roe > 0:
            s = roe / 0.12 * 15
        else:
            s = 0.0
        score += s

        # 5. Debt-to-Equity (15 Punkte)
        de = metrics.debt_to_equity
        if de < 0:
            s = 0.0  # Negatives Eigenkapital
        elif de <= 0.5:
            s = 15.0
        elif de <= 1.0:
            s = 15.0 - (de - 0.5) / 0.5 * 3
        elif de <= 2.0:
            s = 12.0 - (de - 1.0) / 1.0 * 5
        else:
            s = max(0, 7.0 - (de - 2.0))
        score += s

        return round(max(0.0, min(100.0, score)), 1)

    def _build_dividend_metrics(
        self,
        quote: dict,
        detailed: Optional[dict]
    ) -> Optional[DividendMetrics]:
        """
        Erstellt DividendMetrics aus Screen-Quote + optionalen Detail-Daten.
        Behandelt fehlende Daten robust.
        """
        ticker = quote.get('symbol', '?')
        try:
            info = detailed.get('info', {}) if detailed else {}
            dividends = detailed.get('dividends', pd.Series(dtype=float)) if detailed else pd.Series(dtype=float)

            # Dividend Yield: Normalisierung falls als Fraktion (< 1) statt Prozent
            div_yield = _safe(quote.get('dividendYield'))
            if 0 < div_yield < 1:
                div_yield *= 100
            trailing_yield = _safe(quote.get('trailingAnnualDividendYield')) * 100

            return DividendMetrics(
                ticker=ticker,
                name=_safe(quote.get('longName') or quote.get('shortName'), ticker),
                sector=_safe(info.get('sector') or quote.get('sector'), 'N/A'),
                current_price=_safe(quote.get('regularMarketPrice')),
                market_cap=_safe(quote.get('marketCap'), 0),
                currency=_safe(quote.get('currency'), 'USD'),
                dividend_yield=div_yield,
                trailing_dividend_yield=trailing_yield,
                dividend_rate=_safe(quote.get('dividendRate')),
                payout_ratio=_safe(info.get('payoutRatio')),
                consecutive_years_increase=self._calculate_consecutive_increases(dividends),
                dividend_growth_rate_5y=self._calculate_dividend_growth_rate(dividends),
                five_year_avg_yield=_safe(info.get('fiveYearAvgDividendYield')),
                return_on_equity=_safe(info.get('returnOnEquity')),
                debt_to_equity=_safe(info.get('debtToEquity')) / 100,
                free_cashflow=_safe(info.get('freeCashflow'), 0),
                dividend_history=dividends if not dividends.empty else None,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Metrics für %s konnten nicht erstellt werden: %s", ticker, e)
            return None

    def _apply_post_filters(
        self,
        results: list[DividendMetrics],
        filters: ScreenerFilter
    ) -> list[DividendMetrics]:
        """Wendet Filter an, die Detail-Daten benötigen."""
        filtered = []
        for m in results:
            if m.payout_ratio > 0 and m.payout_ratio > filters.max_payout_ratio:
                continue
            if m.dividend_growth_rate_5y < filters.min_dividend_growth_rate:
                continue
            if m.consecutive_years_increase < filters.min_consecutive_years:
                continue
            filtered.append(m)
        return filtered

    def clear_cache(self):
        """Leert den Enrichment-Cache."""
        self._cache.clear()
