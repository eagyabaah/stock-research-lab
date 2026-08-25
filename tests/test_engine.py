from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from stock_model.engine import ModelConfig, analyze_us_swing, assess_market_regime
from stock_model.ghana import analyze_ghana_long_term
from stock_model.ghana_data import _parse_profile, headline_risk, parse_stockanalysis_pages
from stock_model.indicators import add_indicators
from stock_model.data import _normalize_news
from scripts.generate_closing_report import compare_reports, json_safe


def synthetic_history(
    start: float = 100.0,
    end: float = 160.0,
    periods: int = 300,
    volume: float = 2_000_000,
) -> pd.DataFrame:
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    trend = np.linspace(start, end, periods)
    wave = np.sin(np.linspace(0, 12, periods)) * 1.5
    close = trend + wave
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(periods, volume),
        },
        index=index,
    )


class IndicatorTests(unittest.TestCase):
    def test_indicators_are_calculated(self) -> None:
        enriched = add_indicators(synthetic_history())
        for column in ["SMA20", "SMA50", "SMA200", "RSI14", "ATR14", "PRIOR_HIGH20"]:
            self.assertIn(column, enriched.columns)
            self.assertFalse(pd.isna(enriched[column].iloc[-1]))

    def test_requires_enough_history(self) -> None:
        with self.assertRaises(ValueError):
            add_indicators(synthetic_history(periods=100))


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stock = synthetic_history(volume=3_000_000)
        self.spy = synthetic_history(start=400, end=520, volume=50_000_000)
        self.fundamentals = {
            "revenue_growth": 0.18,
            "earnings_growth": 0.22,
            "profit_margin": 0.25,
            "free_cash_flow": 10_000_000_000,
            "total_cash": 50_000_000_000,
            "total_debt": 40_000_000_000,
            "trailing_pe": 25,
            "forward_pe": 22,
            "peg_ratio": 1.4,
            "earnings_date": datetime.now(timezone.utc) + timedelta(days=30),
        }
        self.news = [
            {
                "title": "Company expands a major customer partnership",
                "published": datetime.now(timezone.utc).isoformat(),
                "publisher": "Example",
                "link": "https://example.com",
            }
        ]

    def test_bullish_market_regime(self) -> None:
        self.assertEqual(assess_market_regime(self.spy).status, "PASS")

    def test_position_size_respects_caps(self) -> None:
        result = analyze_us_swing(
            "TEST",
            self.stock,
            self.fundamentals,
            self.news,
            self.spy,
            ModelConfig(),
        )
        self.assertLessEqual(result.score, 100)
        if result.trade_plan:
            self.assertLessEqual(result.trade_plan.notional, 200.02)
            self.assertLessEqual(result.trade_plan.planned_risk, 5.02)

    def test_near_earnings_fails_news_gate(self) -> None:
        fundamentals = dict(self.fundamentals)
        fundamentals["earnings_date"] = datetime.now(timezone.utc) + timedelta(days=1)
        result = analyze_us_swing(
            "TEST", self.stock, fundamentals, self.news, self.spy, ModelConfig()
        )
        news_gate = next(gate for gate in result.gates if gate.name == "News/catalyst")
        self.assertEqual(news_gate.status, "FAIL")
        self.assertNotEqual(result.status, "QUALIFIED")

    def test_drawdown_halt_blocks_new_trade(self) -> None:
        result = analyze_us_swing(
            "TEST",
            self.stock,
            self.fundamentals,
            self.news,
            self.spy,
            ModelConfig(current_drawdown_pct=10),
        )
        portfolio_gate = next(gate for gate in result.gates if gate.name == "Portfolio risk")
        self.assertEqual(portfolio_gate.status, "FAIL")
        self.assertNotEqual(result.status, "QUALIFIED")

    def test_total_open_risk_cap_blocks_new_trade(self) -> None:
        result = analyze_us_swing(
            "TEST",
            self.stock,
            self.fundamentals,
            self.news,
            self.spy,
            ModelConfig(current_open_risk=20),
        )
        portfolio_gate = next(gate for gate in result.gates if gate.name == "Portfolio risk")
        self.assertEqual(portfolio_gate.status, "FAIL")
        self.assertIsNone(result.trade_plan)

    def test_bearish_view_does_not_enable_short_execution(self) -> None:
        falling_stock = synthetic_history(start=180, end=80, volume=3_000_000)
        result = analyze_us_swing(
            "BEAR",
            falling_stock,
            self.fundamentals,
            self.news,
            self.spy,
            ModelConfig(),
        )
        self.assertEqual(result.directional_view.bias, "BEARISH / AVOID LONG")
        self.assertIsNotNone(result.directional_view.bearish_trigger)
        self.assertFalse(result.directional_view.short_execution_allowed)

    def test_result_json_includes_directional_view(self) -> None:
        result = analyze_us_swing(
            "TEST",
            self.stock,
            self.fundamentals,
            self.news,
            self.spy,
            ModelConfig(),
        )
        payload = result.to_dict()
        self.assertIn("directional_view", payload)
        self.assertIn("bias", payload["directional_view"])


class GhanaTests(unittest.TestCase):
    def test_complete_strong_candidate_accumulates(self) -> None:
        result = analyze_ghana_long_term(
            {
                "ticker": "EXAMPLE",
                "revenue_growth_pct": 20,
                "earnings_growth_pct": 25,
                "roe_pct": 24,
                "pe_ratio": 10,
                "dividend_yield_pct": 6,
                "operating_cashflow_positive": True,
                "debt_to_equity": 0.8,
                "governance_flag": False,
                "regulatory_flag": False,
                "avg_daily_value_ghs": 2_000_000,
                "price_vs_200d_pct": 5,
            }
        )
        self.assertEqual(result["status"], "ACCUMULATE GRADUALLY")
        self.assertEqual(result["score"], 100)

    def test_missing_data_does_not_pass(self) -> None:
        result = analyze_ghana_long_term({"ticker": "EMPTY"})
        self.assertEqual(result["status"], "INSUFFICIENT DATA")

    def test_automated_pages_extract_auditable_metrics(self) -> None:
        statistics = """
        <html><head><title>Scancom Statistics & Valuation Metrics</title></head><body>
        Ghana · Delayed Price · Currency is GHS 7.00 At close: Aug 24, 2026
        <table>
          <tr><td>PE Ratio</td><td>11.05</td></tr>
          <tr><td>Debt / Equity</td><td>0.30</td></tr>
          <tr><td>Return on Equity (ROE)</td><td>82.76%</td></tr>
          <tr><td>200-Day Moving Average</td><td>5.58</td></tr>
          <tr><td>Average Volume (20 Days)</td><td>1,659,223</td></tr>
          <tr><td>Operating Cash Flow</td><td>12.21B</td></tr>
          <tr><td>Dividend Yield</td><td>6.86%</td></tr>
        </table>
        Data Source: S&amp;P Global Last updated: Aug 24, 2026
        </body></html>
        """
        income = """
        <html><body><table>
          <tr><td>Revenue Growth</td><td>49.96%</td><td>36.17%</td></tr>
          <tr><td>Net Income Growth</td><td>36.78%</td><td>55.89%</td></tr>
        </table>Data Source: S&amp;P Global Last updated: Jun 30, 2026</body></html>
        """
        company, price, price_date, inputs, metrics = parse_stockanalysis_pages(
            "MTNGH", statistics, income
        )
        self.assertEqual(company, "Scancom")
        self.assertEqual(price, 7.0)
        self.assertEqual(price_date, "2026-08-24")
        self.assertAlmostEqual(inputs["revenue_growth_pct"], 49.96)
        self.assertAlmostEqual(inputs["earnings_growth_pct"], 36.78)
        self.assertTrue(inputs["operating_cashflow_positive"])
        self.assertGreater(inputs["avg_daily_value_ghs"], 10_000_000)
        self.assertGreater(inputs["price_vs_200d_pct"], 20)
        self.assertTrue(all(item.url.startswith("https://") for item in metrics))

    def test_governance_keyword_screen_is_explicit(self) -> None:
        headlines = [{"title": "Issuer announces regulatory investigation"}]
        self.assertTrue(headline_risk(headlines, ("investigation",)))
        self.assertFalse(headline_risk(headlines, ("dividend",)))

    def test_business_profile_parser_keeps_source_description(self) -> None:
        html = """
        <html><body><h1>Scancom Company Description</h1>
        <p>Scancom Plc provides telecommunications and mobile-money services in Ghana.</p>
        <table><tr><td>Country</td><td>Ghana</td></tr>
        <tr><td>Industry</td><td>Radiotelephone Communications</td></tr></table>
        Data Source: S&amp;P Global Last updated: Aug 7, 2026
        </body></html>
        """
        industry, description, as_of = _parse_profile(html)
        self.assertEqual(industry, "Radiotelephone Communications")
        self.assertIn("mobile-money services", description)
        self.assertEqual(as_of, "2026-08-07")


class DataAdapterTests(unittest.TestCase):
    def test_normalizes_current_nested_news_shape(self) -> None:
        normalized = _normalize_news(
            [
                {
                    "content": {
                        "title": "Example filing update",
                        "provider": {"displayName": "Example Wire"},
                        "pubDate": "2026-08-25T12:00:00Z",
                        "canonicalUrl": {"url": "https://example.com/filing"},
                    }
                }
            ]
        )
        self.assertEqual(normalized[0]["publisher"], "Example Wire")
        self.assertEqual(normalized[0]["link"], "https://example.com/filing")


class ScheduledReportTests(unittest.TestCase):
    def test_comparison_records_score_and_entry_changes(self) -> None:
        previous = {
            "reports": [
                {
                    "ticker": "RDW",
                    "status": "WATCH",
                    "score": 70,
                    "directional_view": {"bias": "MIXED / WAIT"},
                    "trade_plan": {"entry_trigger": 50, "stop": 45, "target_1": 60},
                    "facts": {"earnings_date": "2026-09-01"},
                }
            ]
        }
        current = [
            {
                "ticker": "RDW",
                "status": "WATCH",
                "score": 76,
                "directional_view": {"bias": "BULLISH WATCH"},
                "trade_plan": {"entry_trigger": 52, "stop": 47, "target_1": 62},
                "facts": {"earnings_date": "2026-09-01"},
            }
        ]
        fields = {item["field"] for item in compare_reports(previous, current)}
        self.assertIn("Score", fields)
        self.assertIn("Entry", fields)
        self.assertIn("Directional view", fields)
        self.assertNotIn("Decision", fields)

    def test_json_safe_replaces_non_finite_values(self) -> None:
        self.assertIsNone(json_safe(float("nan")))


if __name__ == "__main__":
    unittest.main()
