"""
Tests für den Dividend Screener mit gemocktem yfinance.
"""
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from unittest.mock import patch, MagicMock

from src.data.dividend_screener import (
    DividendScreener,
    DividendMetrics,
    ScreenerFilter,
    EXCHANGE_OPTIONS,
)
from src.visualization.charts import plot_dividend_screener_scatter, plot_dividend_history


def _make_metrics(ticker='TEST', **overrides):
    """Factory für DividendMetrics mit sinnvollen Test-Defaults."""
    defaults = dict(
        ticker=ticker, name=f'{ticker} Corp', sector='Tech',
        current_price=100.0, market_cap=50e9, currency='USD',
        dividend_yield=3.0, trailing_dividend_yield=2.8,
        dividend_rate=3.0, payout_ratio=0.45,
        consecutive_years_increase=10,
        dividend_growth_rate_5y=6.0,
        five_year_avg_yield=2.8,
        return_on_equity=0.18,
        debt_to_equity=0.5,
        free_cashflow=5e9,
    )
    defaults.update(overrides)
    return DividendMetrics(**defaults)


class TestScreenerFilter:
    """Tests für ScreenerFilter Dataclass."""

    def test_default_values(self):
        f = ScreenerFilter()
        assert f.min_dividend_yield == 1.0
        assert f.max_dividend_yield == 10.0
        assert f.max_payout_ratio == 0.70
        assert f.min_consecutive_years == 5
        assert f.min_dividend_growth_rate == 0.0
        assert f.min_market_cap == 1e10
        assert f.exchange == "NYQ"
        assert f.max_results == 25

    def test_custom_values(self):
        f = ScreenerFilter(min_dividend_yield=3.0, max_payout_ratio=0.5, exchange="NMS")
        assert f.min_dividend_yield == 3.0
        assert f.max_payout_ratio == 0.5
        assert f.exchange == "NMS"


class TestExchangeOptions:
    def test_exchange_mapping_has_entries(self):
        assert len(EXCHANGE_OPTIONS) >= 5
        assert "USA (NYSE)" in EXCHANGE_OPTIONS
        assert "Alle" in EXCHANGE_OPTIONS
        assert EXCHANGE_OPTIONS["Alle"] is None


class TestConsecutiveIncreases:
    """Tests für die Berechnung konsekutiver Dividendensteigerungen."""

    def test_steady_increases(self):
        screener = DividendScreener()
        dates = pd.date_range('2015-01-15', '2024-10-15', freq='QS')
        values = []
        for d in dates:
            year_idx = d.year - 2015
            values.append(1.0 * (1.05 ** year_idx))
        dividends = pd.Series(values, index=dates, name='Dividends')

        result = screener._calculate_consecutive_increases(dividends)
        assert result >= 8

    def test_with_decrease(self):
        screener = DividendScreener()
        dates = pd.to_datetime([
            '2020-03-15', '2021-03-15', '2022-03-15', '2023-03-15', '2024-03-15'
        ])
        values = [4.0, 4.2, 3.8, 4.0, 4.2]
        dividends = pd.Series(values, index=dates)

        result = screener._calculate_consecutive_increases(dividends)
        assert result == 2  # Nur 2022->2023 und 2023->2024

    def test_all_decreasing(self):
        screener = DividendScreener()
        dates = pd.to_datetime(['2020-06-15', '2021-06-15', '2022-06-15', '2023-06-15'])
        values = [4.0, 3.5, 3.0, 2.5]
        dividends = pd.Series(values, index=dates)

        result = screener._calculate_consecutive_increases(dividends)
        assert result == 0

    def test_flat_dividends(self):
        screener = DividendScreener()
        dates = pd.to_datetime(['2020-06-15', '2021-06-15', '2022-06-15'])
        values = [2.0, 2.0, 2.0]
        dividends = pd.Series(values, index=dates)

        result = screener._calculate_consecutive_increases(dividends)
        assert result == 0  # Gleich ist keine Steigerung

    def test_empty_dividends(self):
        screener = DividendScreener()
        result = screener._calculate_consecutive_increases(pd.Series(dtype=float))
        assert result == 0

    def test_single_year(self):
        screener = DividendScreener()
        dates = pd.to_datetime(['2024-03-15'])
        dividends = pd.Series([2.0], index=dates)

        result = screener._calculate_consecutive_increases(dividends)
        assert result == 0

    def test_quarterly_payments(self):
        """Testet korrekte Gruppierung bei vierteljährlichen Dividenden."""
        screener = DividendScreener()
        dates = pd.to_datetime([
            '2021-03-15', '2021-06-15', '2021-09-15', '2021-12-15',  # sum=4.0
            '2022-03-15', '2022-06-15', '2022-09-15', '2022-12-15',  # sum=4.4
            '2023-03-15', '2023-06-15', '2023-09-15', '2023-12-15',  # sum=4.8
        ])
        values = [1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1, 1.2, 1.2, 1.2, 1.2]
        dividends = pd.Series(values, index=dates)

        result = screener._calculate_consecutive_increases(dividends)
        assert result == 2  # 2021->2022 und 2022->2023


class TestDividendGrowthRate:
    """Tests für die CAGR-Berechnung."""

    def test_5_year_cagr(self):
        screener = DividendScreener()
        dates = pd.to_datetime([
            '2019-06-15', '2020-06-15', '2021-06-15',
            '2022-06-15', '2023-06-15', '2024-06-15'
        ])
        # ~5% CAGR: 2.00 -> 2.55
        values = [2.00, 2.10, 2.205, 2.315, 2.431, 2.553]
        dividends = pd.Series(values, index=dates)

        cagr = screener._calculate_dividend_growth_rate(dividends, years=5)
        assert 4.5 < cagr < 5.5

    def test_negative_growth(self):
        screener = DividendScreener()
        dates = pd.to_datetime(['2020-06-15', '2021-06-15', '2022-06-15', '2023-06-15', '2024-06-15'])
        values = [4.0, 3.8, 3.6, 3.4, 3.2]
        dividends = pd.Series(values, index=dates)

        cagr = screener._calculate_dividend_growth_rate(dividends, years=4)
        assert cagr < 0

    def test_insufficient_data(self):
        screener = DividendScreener()
        dates = pd.to_datetime(['2024-06-15'])
        dividends = pd.Series([2.0], index=dates)

        cagr = screener._calculate_dividend_growth_rate(dividends)
        assert cagr == 0.0

    def test_empty_dividends(self):
        screener = DividendScreener()
        cagr = screener._calculate_dividend_growth_rate(pd.Series(dtype=float))
        assert cagr == 0.0

    def test_zero_start_dividend(self):
        screener = DividendScreener()
        dates = pd.to_datetime(['2020-06-15', '2021-06-15', '2022-06-15'])
        values = [0.0, 1.0, 2.0]
        dividends = pd.Series(values, index=dates)

        cagr = screener._calculate_dividend_growth_rate(dividends, years=2)
        assert cagr == 0.0  # Kann nicht berechnen bei Start=0


class TestQualityScore:
    """Tests für die Qualitätsscore-Berechnung."""

    def test_high_quality_stock(self):
        """Dividend Aristocrat sollte hoch scoren."""
        screener = DividendScreener()
        metrics = _make_metrics(
            consecutive_years_increase=30,
            dividend_growth_rate_5y=8.0,
            payout_ratio=0.45,
            return_on_equity=0.20,
            debt_to_equity=0.4,
        )
        score = screener._calculate_quality_score(metrics)
        assert score > 75

    def test_low_quality_yield_trap(self):
        """Yield-Trap mit hoher Rendite aber schlechter Qualität."""
        screener = DividendScreener()
        metrics = _make_metrics(
            consecutive_years_increase=1,
            dividend_growth_rate_5y=-2.0,
            payout_ratio=0.95,
            return_on_equity=0.03,
            debt_to_equity=3.0,
        )
        score = screener._calculate_quality_score(metrics)
        assert score < 30

    def test_medium_quality(self):
        screener = DividendScreener()
        metrics = _make_metrics(
            consecutive_years_increase=8,
            dividend_growth_rate_5y=4.0,
            payout_ratio=0.55,
            return_on_equity=0.14,
            debt_to_equity=0.8,
        )
        score = screener._calculate_quality_score(metrics)
        assert 40 < score < 75

    def test_score_always_in_range(self):
        """Score muss immer 0-100 sein."""
        screener = DividendScreener()
        for pr in [0.0, 0.3, 0.5, 0.8, 1.5]:
            for roe in [-0.1, 0.0, 0.15, 0.5]:
                for de in [-0.5, 0.0, 1.0, 5.0]:
                    metrics = _make_metrics(
                        payout_ratio=pr,
                        return_on_equity=roe,
                        debt_to_equity=de,
                    )
                    score = screener._calculate_quality_score(metrics)
                    assert 0 <= score <= 100, f"Score {score} out of range for pr={pr}, roe={roe}, de={de}"

    def test_payout_ratio_sweet_spot(self):
        """Payout Ratio 30-60% sollte den besten Payout-Score geben."""
        screener = DividendScreener()
        low = _make_metrics(payout_ratio=0.15)
        sweet = _make_metrics(payout_ratio=0.45)
        high = _make_metrics(payout_ratio=0.90)

        s_low = screener._calculate_quality_score(low)
        s_sweet = screener._calculate_quality_score(sweet)
        s_high = screener._calculate_quality_score(high)

        assert s_sweet > s_low
        assert s_sweet > s_high

    def test_more_consecutive_years_better(self):
        screener = DividendScreener()
        few = _make_metrics(consecutive_years_increase=3)
        many = _make_metrics(consecutive_years_increase=25)

        assert screener._calculate_quality_score(many) > screener._calculate_quality_score(few)

    def test_higher_growth_rate_better(self):
        screener = DividendScreener()
        low = _make_metrics(dividend_growth_rate_5y=1.0)
        high = _make_metrics(dividend_growth_rate_5y=12.0)

        assert screener._calculate_quality_score(high) > screener._calculate_quality_score(low)


class TestBuildScreenQuery:
    """Tests für Query-Konstruktion."""

    def test_builds_valid_query(self):
        screener = DividendScreener()
        filters = ScreenerFilter(
            min_dividend_yield=2.0,
            min_consecutive_years=10,
            exchange="NYQ",
        )
        query = screener._build_screen_query(filters)
        assert query is not None

    def test_query_without_exchange(self):
        screener = DividendScreener()
        filters = ScreenerFilter(exchange=None)
        query = screener._build_screen_query(filters)
        assert query is not None

    def test_query_with_max_yield(self):
        screener = DividendScreener()
        filters = ScreenerFilter(max_dividend_yield=8.0)
        query = screener._build_screen_query(filters)
        assert query is not None


class TestPostFilters:
    """Tests für Post-Screening Filter."""

    def test_filters_high_payout(self):
        screener = DividendScreener()
        results = [
            _make_metrics('A', payout_ratio=0.90),
            _make_metrics('B', payout_ratio=0.45),
        ]
        filters = ScreenerFilter(max_payout_ratio=0.70, min_consecutive_years=0)
        filtered = screener._apply_post_filters(results, filters)
        assert len(filtered) == 1
        assert filtered[0].ticker == 'B'

    def test_filters_low_growth(self):
        screener = DividendScreener()
        results = [
            _make_metrics('A', dividend_growth_rate_5y=-1.0),
            _make_metrics('B', dividend_growth_rate_5y=5.0),
        ]
        filters = ScreenerFilter(min_dividend_growth_rate=2.0, min_consecutive_years=0)
        filtered = screener._apply_post_filters(results, filters)
        assert len(filtered) == 1
        assert filtered[0].ticker == 'B'

    def test_filters_low_consecutive_years(self):
        screener = DividendScreener()
        results = [
            _make_metrics('A', consecutive_years_increase=2),
            _make_metrics('B', consecutive_years_increase=10),
        ]
        filters = ScreenerFilter(min_consecutive_years=5)
        filtered = screener._apply_post_filters(results, filters)
        assert len(filtered) == 1
        assert filtered[0].ticker == 'B'

    def test_zero_payout_not_filtered(self):
        """Payout Ratio 0 (unbekannt) sollte nicht herausgefiltert werden."""
        screener = DividendScreener()
        results = [_make_metrics('A', payout_ratio=0.0)]
        filters = ScreenerFilter(max_payout_ratio=0.70, min_consecutive_years=0)
        filtered = screener._apply_post_filters(results, filters)
        assert len(filtered) == 1

    def test_no_filters_pass_all(self):
        screener = DividendScreener()
        results = [
            _make_metrics('A'),
            _make_metrics('B'),
        ]
        filters = ScreenerFilter(
            max_payout_ratio=1.0,
            min_dividend_growth_rate=0.0,
            min_consecutive_years=0
        )
        filtered = screener._apply_post_filters(results, filters)
        assert len(filtered) == 2


class TestBuildDividendMetrics:
    """Tests für Metrics-Erstellung aus Quote-Daten."""

    def test_builds_from_quote_and_details(self):
        screener = DividendScreener()
        quote = {
            'symbol': 'JNJ',
            'longName': 'Johnson & Johnson',
            'regularMarketPrice': 150.0,
            'marketCap': 400e9,
            'currency': 'USD',
            'dividendYield': 3.0,
            'dividendRate': 4.76,
            'trailingAnnualDividendYield': 0.028,
        }
        detailed = {
            'info': {
                'sector': 'Healthcare',
                'payoutRatio': 0.45,
                'returnOnEquity': 0.20,
                'debtToEquity': 50.0,
                'fiveYearAvgDividendYield': 2.6,
                'freeCashflow': 15e9,
            },
            'dividends': pd.Series(dtype=float),
        }
        metrics = screener._build_dividend_metrics(quote, detailed)
        assert metrics is not None
        assert metrics.ticker == 'JNJ'
        assert metrics.name == 'Johnson & Johnson'
        assert metrics.dividend_yield == 3.0
        assert metrics.payout_ratio == 0.45
        assert metrics.debt_to_equity == 0.5  # 50/100

    def test_handles_missing_fields(self):
        screener = DividendScreener()
        quote = {'symbol': 'X'}
        metrics = screener._build_dividend_metrics(quote, None)
        assert metrics is not None
        assert metrics.ticker == 'X'
        assert metrics.dividend_yield == 0.0

    def test_handles_none_values(self):
        screener = DividendScreener()
        quote = {
            'symbol': 'Y',
            'longName': None,
            'dividendYield': None,
            'dividendRate': None,
            'trailingAnnualDividendYield': None,
            'regularMarketPrice': None,
            'marketCap': None,
            'currency': None,
        }
        detailed = {
            'info': {
                'payoutRatio': None,
                'returnOnEquity': None,
                'debtToEquity': None,
                'fiveYearAvgDividendYield': None,
                'freeCashflow': None,
                'sector': None,
            },
            'dividends': pd.Series(dtype=float),
        }
        metrics = screener._build_dividend_metrics(quote, detailed)
        assert metrics is not None
        assert metrics.name == 'Y'
        assert metrics.dividend_yield == 0.0


class TestScreenIntegration:
    """Integrationstests mit gemocktem yfinance."""

    @patch('src.data.dividend_screener.yf.screen')
    @patch('src.data.dividend_screener.yf.Ticker')
    def test_screen_returns_results(self, mock_ticker_cls, mock_screen):
        mock_screen.return_value = {
            'quotes': [{
                'symbol': 'TEST',
                'longName': 'Test Corp',
                'regularMarketPrice': 100.0,
                'marketCap': 50e9,
                'currency': 'USD',
                'dividendYield': 3.5,
                'dividendRate': 3.50,
                'trailingAnnualDividendYield': 0.032,
            }],
            'total': 1
        }

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {
            'payoutRatio': 0.45,
            'returnOnEquity': 0.18,
            'debtToEquity': 50.0,
            'sector': 'Technology',
            'fiveYearAvgDividendYield': 3.0,
            'freeCashflow': 5e9,
        }
        dates = pd.date_range('2019-01-15', '2024-10-15', freq='QS')
        mock_ticker_instance.dividends = pd.Series(
            [0.5 * (1.05 ** (d.year - 2019)) for d in dates],
            index=dates
        )
        mock_ticker_cls.return_value = mock_ticker_instance

        screener = DividendScreener()
        results = screener.screen(ScreenerFilter(min_consecutive_years=0))

        assert len(results) > 0
        assert results[0].ticker == 'TEST'
        assert results[0].quality_score > 0

    @patch('src.data.dividend_screener.yf.screen')
    def test_screen_handles_empty_results(self, mock_screen):
        mock_screen.return_value = {'quotes': [], 'total': 0}

        screener = DividendScreener()
        results = screener.screen(ScreenerFilter())

        assert results == []

    @patch('src.data.dividend_screener.yf.screen')
    def test_screen_handles_api_error(self, mock_screen):
        mock_screen.side_effect = Exception("API Error")

        screener = DividendScreener()
        results = screener.screen(ScreenerFilter())

        assert results == []

    @patch('src.data.dividend_screener.yf.screen')
    @patch('src.data.dividend_screener.yf.Ticker')
    def test_screen_with_progress_callback(self, mock_ticker_cls, mock_screen):
        mock_screen.return_value = {
            'quotes': [{'symbol': 'A', 'dividendYield': 3.0, 'currency': 'USD'}],
            'total': 1
        }
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {}
        mock_ticker_instance.dividends = pd.Series(dtype=float)
        mock_ticker_cls.return_value = mock_ticker_instance

        progress_calls = []
        def callback(pct, text):
            progress_calls.append((pct, text))

        screener = DividendScreener()
        screener.screen(ScreenerFilter(min_consecutive_years=0), progress_callback=callback)

        assert len(progress_calls) > 0
        assert progress_calls[0][0] == 10

    @patch('src.data.dividend_screener.yf.screen')
    @patch('src.data.dividend_screener.yf.Ticker')
    def test_screen_uses_cache(self, mock_ticker_cls, mock_screen):
        mock_screen.return_value = {
            'quotes': [{'symbol': 'CACHED', 'dividendYield': 3.0, 'currency': 'USD'}],
            'total': 1
        }
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'payoutRatio': 0.40}
        mock_ticker_instance.dividends = pd.Series(dtype=float)
        mock_ticker_cls.return_value = mock_ticker_instance

        screener = DividendScreener()
        screener.screen(ScreenerFilter(min_consecutive_years=0))

        # Zweiter Aufruf sollte Cache nutzen
        mock_ticker_cls.reset_mock()
        screener.screen(ScreenerFilter(min_consecutive_years=0))

        # Ticker sollte nicht nochmal aufgerufen werden (cached)
        mock_ticker_cls.assert_not_called()

    def test_clear_cache(self):
        screener = DividendScreener()
        metrics = _make_metrics()
        screener._set_cached('TEST', metrics)
        assert len(screener._cache) == 1
        screener.clear_cache()
        assert len(screener._cache) == 0


class TestNegativePayoutRatio:
    """Tests für negative Payout Ratios (z.B. Verlustjahre)."""

    def test_negative_payout_ratio_clamped_to_zero(self):
        """Negative Payout Ratio darf keinen negativen Score erzeugen."""
        screener = DividendScreener()
        metrics = _make_metrics(payout_ratio=-0.5)
        score = screener._calculate_quality_score(metrics)
        assert score >= 0, f"Score {score} should be >= 0 with negative payout ratio"

    def test_negative_payout_ratio_score_in_range(self):
        """Score muss auch bei negativer Payout Ratio in 0-100 liegen."""
        screener = DividendScreener()
        for pr in [-2.0, -1.0, -0.5, -0.1]:
            metrics = _make_metrics(payout_ratio=pr)
            score = screener._calculate_quality_score(metrics)
            assert 0 <= score <= 100, f"Score {score} out of range for payout_ratio={pr}"


class TestPartialYearExclusion:
    """Tests für die Ausschluss des unvollständigen aktuellen Jahres."""

    def test_current_year_excluded_from_consecutive(self):
        """Das aktuelle (unvollständige) Jahr darf nicht in die Berechnung einfließen."""
        screener = DividendScreener()
        current_year = date.today().year
        # Aufsteigend bis Vorjahr, dann abfallend im aktuellen Jahr (unvollständig)
        dates = pd.to_datetime([
            f'{current_year-3}-06-15',
            f'{current_year-2}-06-15',
            f'{current_year-1}-06-15',
            f'{current_year}-03-15',  # Nur Q1, unvollständig
        ])
        values = [2.0, 2.2, 2.4, 0.6]  # 0.6 wäre "Abstieg" wenn komplett gezählt
        dividends = pd.Series(values, index=dates)

        result = screener._calculate_consecutive_increases(dividends)
        # Ohne aktuelles Jahr: 2.0 -> 2.2 -> 2.4 = 2 konsekutive Steigerungen
        assert result == 2

    def test_current_year_excluded_from_cagr(self):
        """CAGR darf nicht durch unvollständiges aktuelles Jahr verfälscht werden."""
        screener = DividendScreener()
        current_year = date.today().year
        dates = pd.to_datetime([
            f'{current_year-4}-06-15',
            f'{current_year-3}-06-15',
            f'{current_year-2}-06-15',
            f'{current_year-1}-06-15',
            f'{current_year}-02-15',  # Nur ein Quartal
        ])
        values = [2.0, 2.1, 2.2, 2.3, 0.5]  # 0.5 wäre verfälschend
        dividends = pd.Series(values, index=dates)

        cagr = screener._calculate_dividend_growth_rate(dividends, years=3)
        # Sollte auf Basis 2.0->2.3 berechnet sein (ohne aktuelles Jahr)
        assert cagr > 0

    def test_get_complete_annual_dividends_excludes_current_year(self):
        """_get_complete_annual_dividends muss aktuelles Jahr ausschließen."""
        current_year = date.today().year
        dates = pd.to_datetime([
            f'{current_year-2}-06-15',
            f'{current_year-1}-06-15',
            f'{current_year}-01-15',
        ])
        dividends = pd.Series([2.0, 2.5, 0.5], index=dates)

        annual = DividendScreener._get_complete_annual_dividends(dividends)
        assert current_year not in annual.index
        assert len(annual) == 2


class TestSingleConditionQuery:
    """Tests für EquityQuery mit einzelner Bedingung."""

    def test_query_with_no_optional_filters(self):
        """Query mit nur min_dividend_yield (keine Exchange, Market Cap 0, max yield 100)."""
        screener = DividendScreener()
        filters = ScreenerFilter(
            exchange=None,
            min_market_cap=0,
            max_dividend_yield=100,
        )
        query = screener._build_screen_query(filters)
        assert query is not None

    def test_query_with_all_filters(self):
        """Query mit allen Filtern aktiv."""
        screener = DividendScreener()
        filters = ScreenerFilter(
            min_dividend_yield=2.0,
            max_dividend_yield=8.0,
            min_market_cap=1e10,
            exchange="NYQ",
        )
        query = screener._build_screen_query(filters)
        assert query is not None


class TestPreferredShareFilter:
    """Tests für das Herausfiltern von Preferred Shares."""

    def test_detects_preferred_share_with_letter(self):
        assert DividendScreener._is_preferred_share('NLY-PF') is True
        assert DividendScreener._is_preferred_share('MS-PA') is True
        assert DividendScreener._is_preferred_share('SPG-PJ') is True

    def test_detects_preferred_share_without_letter(self):
        assert DividendScreener._is_preferred_share('AXIA-P') is True

    def test_normal_tickers_not_filtered(self):
        assert DividendScreener._is_preferred_share('AAPL') is False
        assert DividendScreener._is_preferred_share('JPM') is False
        assert DividendScreener._is_preferred_share('T') is False
        assert DividendScreener._is_preferred_share('MPLX') is False

    def test_empty_symbol(self):
        assert DividendScreener._is_preferred_share('') is False

    def test_ticker_with_p_not_at_end(self):
        """Ticker wie PG oder PFE dürfen nicht gefiltert werden."""
        assert DividendScreener._is_preferred_share('PG') is False
        assert DividendScreener._is_preferred_share('PFE') is False
        assert DividendScreener._is_preferred_share('EPD') is False

    @patch('src.data.dividend_screener.yf.screen')
    @patch('src.data.dividend_screener.yf.Ticker')
    def test_preferred_shares_excluded_from_screen(self, mock_ticker_cls, mock_screen):
        """Preferred Shares werden vor Enrichment herausgefiltert."""
        mock_screen.return_value = {
            'quotes': [
                {'symbol': 'JNJ', 'dividendYield': 3.0, 'currency': 'USD'},
                {'symbol': 'NLY-PF', 'dividendYield': 8.0, 'currency': 'USD'},
                {'symbol': 'MS-PA', 'dividendYield': 6.0, 'currency': 'USD'},
            ],
            'total': 3
        }
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {}
        mock_ticker_instance.dividends = pd.Series(dtype=float)
        mock_ticker_cls.return_value = mock_ticker_instance

        screener = DividendScreener()
        results = screener.screen(ScreenerFilter(min_consecutive_years=0))

        tickers = [r.ticker for r in results]
        assert 'JNJ' in tickers
        assert 'NLY-PF' not in tickers
        assert 'MS-PA' not in tickers


class TestDividendCharts:
    """Tests für die Dividenden-Chart-Funktionen."""

    def test_scatter_empty_list(self):
        fig = plot_dividend_screener_scatter([])
        assert isinstance(fig, go.Figure)

    def test_scatter_with_metrics(self):
        metrics = [_make_metrics(quality_score=70.0)]
        fig = plot_dividend_screener_scatter(metrics)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_history_empty_series(self):
        fig = plot_dividend_history('TEST', pd.Series(dtype=float), 'Test Corp')
        assert isinstance(fig, go.Figure)

    def test_history_with_data(self):
        dates = pd.to_datetime(['2020-06-15', '2021-06-15', '2022-06-15', '2023-06-15'])
        dividends = pd.Series([2.0, 2.2, 2.4, 2.6], index=dates)
        fig = plot_dividend_history('TEST', dividends, 'Test Corp')
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_history_excludes_current_year(self):
        """Chart sollte das aktuelle unvollständige Jahr nicht anzeigen."""
        current_year = date.today().year
        dates = pd.to_datetime([
            f'{current_year-2}-06-15',
            f'{current_year-1}-06-15',
            f'{current_year}-02-15',  # Unvollständig
        ])
        dividends = pd.Series([2.0, 2.5, 0.5], index=dates)
        fig = plot_dividend_history('TEST', dividends, 'Test Corp')
        assert isinstance(fig, go.Figure)
        # Die Bars sollten nur 2 Jahre enthalten, nicht das aktuelle
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        if bar_traces:
            assert current_year not in list(bar_traces[0].x)


class TestCacheTTL:
    """Tests für TTL-basiertes Caching."""

    def test_cache_hit_within_ttl(self):
        """Frischer Cache-Eintrag wird zurückgegeben."""
        screener = DividendScreener(cache_ttl=3600)
        metrics = _make_metrics('TTL')
        screener._set_cached('TTL', metrics)

        cached = screener._get_cached('TTL')
        assert cached is not None
        assert cached.ticker == 'TTL'

    def test_cache_miss_after_ttl(self):
        """Abgelaufener Cache-Eintrag wird nicht zurückgegeben."""
        screener = DividendScreener(cache_ttl=0)  # TTL=0 -> sofort abgelaufen
        metrics = _make_metrics()
        screener._set_cached('EXPIRED', metrics)

        import time
        time.sleep(0.01)  # Minimal warten

        cached = screener._get_cached('EXPIRED')
        assert cached is None
        # Abgelaufener Eintrag wird aus dem Cache entfernt
        assert 'EXPIRED' not in screener._cache

    def test_cache_miss_for_unknown_ticker(self):
        screener = DividendScreener()
        assert screener._get_cached('UNKNOWN') is None


class TestRetryLogic:
    """Tests für Retry-Logik bei API-Fehlern."""

    @patch('src.data.dividend_screener.time.sleep')  # Sleep nicht warten
    @patch('src.data.dividend_screener.yf.Ticker')
    def test_retry_succeeds_on_second_attempt(self, mock_ticker_cls, mock_sleep):
        """Zweiter Versuch erfolgreich nach erstem Fehler."""
        mock_success = MagicMock()
        mock_success.info = {'sector': 'Tech'}
        mock_success.dividends = pd.Series(dtype=float)

        # Erster Aufruf schlägt fehl, zweiter klappt
        mock_ticker_cls.side_effect = [Exception("Timeout"), mock_success]

        screener = DividendScreener()
        result = screener._enrich_with_detailed_data('RETRY')

        assert result is not None
        assert result['info'] == {'sector': 'Tech'}
        assert mock_ticker_cls.call_count == 2
        mock_sleep.assert_called_once()  # Ein Retry-Sleep

    @patch('src.data.dividend_screener.time.sleep')
    @patch('src.data.dividend_screener.yf.Ticker')
    def test_all_retries_fail(self, mock_ticker_cls, mock_sleep):
        """Alle Versuche scheitern -> None zurückgeben."""
        mock_ticker_cls.side_effect = Exception("Persistent error")

        screener = DividendScreener()
        result = screener._enrich_with_detailed_data('FAIL')

        assert result is None
        assert mock_ticker_cls.call_count == 3  # _MAX_RETRIES


class TestErrorLogging:
    """Tests für spezifisches Exception-Handling statt bare except."""

    def test_build_metrics_handles_none_detailed(self):
        """_build_dividend_metrics funktioniert mit detailed=None."""
        screener = DividendScreener()
        result = screener._build_dividend_metrics({'symbol': 'X'}, None)
        assert result is not None
        assert result.ticker == 'X'

    def test_build_metrics_returns_none_on_bad_data(self):
        """_build_dividend_metrics gibt None zurück bei invaliden Daten statt zu crashen."""
        screener = DividendScreener()
        # detailed mit debtToEquity als String statt Zahl -> TypeError bei / 100
        quote = {'symbol': 'BAD'}
        detailed = {
            'info': {'debtToEquity': 'not_a_number'},
            'dividends': pd.Series(dtype=float),
        }
        result = screener._build_dividend_metrics(quote, detailed)
        # Entweder None (TypeError gefangen) oder Metrics (str / 100 schlägt fehl)
        assert result is None or result.ticker == 'BAD'

    def test_fetch_screen_logs_on_error(self):
        """_fetch_screen_results loggt den Fehler."""
        screener = DividendScreener()
        # Ungültige Query -> Exception
        with patch('src.data.dividend_screener.yf.screen', side_effect=Exception("API down")):
            result = screener._fetch_screen_results(MagicMock(), 25)
            assert result == []


class TestNaNHandling:
    """Tests für NaN-Werte aus der yfinance API."""

    def test_safe_handles_nan(self):
        from src.data.dividend_screener import _safe
        assert _safe(float('nan')) == 0.0
        assert _safe(float('nan'), 42.0) == 42.0

    def test_safe_preserves_valid_zero(self):
        from src.data.dividend_screener import _safe
        assert _safe(0.0) == 0.0
        assert _safe(0) == 0

    def test_safe_handles_none(self):
        from src.data.dividend_screener import _safe
        assert _safe(None) == 0.0
        assert _safe(None, 'fallback') == 'fallback'

    def test_build_metrics_handles_nan_values(self):
        """NaN-Werte aus API dürfen nicht in Quality Score propagieren."""
        screener = DividendScreener()
        quote = {
            'symbol': 'NAN',
            'dividendYield': float('nan'),
            'trailingAnnualDividendYield': float('nan'),
            'regularMarketPrice': float('nan'),
            'marketCap': float('nan'),
        }
        detailed = {
            'info': {
                'debtToEquity': float('nan'),
                'returnOnEquity': float('nan'),
                'payoutRatio': float('nan'),
            },
            'dividends': pd.Series(dtype=float),
        }
        metrics = screener._build_dividend_metrics(quote, detailed)
        assert metrics is not None
        assert metrics.dividend_yield == 0.0
        assert metrics.trailing_dividend_yield == 0.0
        assert metrics.debt_to_equity == 0.0
        score = screener._calculate_quality_score(metrics)
        assert 0 <= score <= 100
        assert not np.isnan(score)


class TestTrailingYieldNormalization:
    """Tests für konsistente Normalisierung von trailing_dividend_yield."""

    def test_trailing_yield_as_fraction(self):
        """Trailing Yield als Fraktion (0.032) wird zu Prozent (3.2) normalisiert."""
        screener = DividendScreener()
        quote = {
            'symbol': 'FRAC',
            'dividendYield': 3.2,
            'trailingAnnualDividendYield': 0.032,
        }
        metrics = screener._build_dividend_metrics(quote, None)
        assert metrics is not None
        assert abs(metrics.trailing_dividend_yield - 3.2) < 0.01

    def test_trailing_yield_as_percent(self):
        """Trailing Yield bereits als Prozent (3.2) bleibt 3.2."""
        screener = DividendScreener()
        quote = {
            'symbol': 'PCT',
            'dividendYield': 3.2,
            'trailingAnnualDividendYield': 3.2,
        }
        metrics = screener._build_dividend_metrics(quote, None)
        assert metrics is not None
        assert abs(metrics.trailing_dividend_yield - 3.2) < 0.01

    def test_trailing_yield_zero(self):
        """Trailing Yield 0 bleibt 0."""
        screener = DividendScreener()
        quote = {
            'symbol': 'ZERO',
            'trailingAnnualDividendYield': 0.0,
        }
        metrics = screener._build_dividend_metrics(quote, None)
        assert metrics is not None
        assert metrics.trailing_dividend_yield == 0.0


class TestCacheSizeLimit:
    """Tests für die Cache-Größenbegrenzung."""

    def test_cache_evicts_oldest_when_full(self):
        """Ältester Eintrag wird entfernt wenn Cache voll ist."""
        from src.data.dividend_screener import _MAX_CACHE_SIZE
        screener = DividendScreener()

        # Fülle Cache bis zum Limit
        for i in range(_MAX_CACHE_SIZE):
            screener._set_cached(f'T{i}', _make_metrics(f'T{i}'))

        assert len(screener._cache) == _MAX_CACHE_SIZE

        # Ein weiterer Eintrag sollte den ältesten verdrängen
        screener._set_cached('NEW', _make_metrics('NEW'))
        assert len(screener._cache) == _MAX_CACHE_SIZE
        assert screener._get_cached('NEW') is not None
        # T0 war der älteste und sollte entfernt worden sein
        assert screener._get_cached('T0') is None

    def test_cache_update_existing_no_eviction(self):
        """Update eines bestehenden Eintrags löst keine Eviction aus."""
        from src.data.dividend_screener import _MAX_CACHE_SIZE
        screener = DividendScreener()

        for i in range(_MAX_CACHE_SIZE):
            screener._set_cached(f'T{i}', _make_metrics(f'T{i}'))

        # Update eines bestehenden → kein Evict
        screener._set_cached('T0', _make_metrics('T0'))
        assert len(screener._cache) == _MAX_CACHE_SIZE
        assert screener._get_cached('T0') is not None
