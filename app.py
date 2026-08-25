from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_model import ModelConfig, analyze_ghana_long_term, analyze_us_swing
from stock_model.data import MarketBundle, fetch_history, fetch_us_bundle
from stock_model.indicators import add_indicators


st.set_page_config(page_title="Stock Research Lab", page_icon="📈", layout="wide")


@st.cache_data(ttl=1_800, show_spinner=False)
def cached_bundle(ticker: str) -> MarketBundle:
    return fetch_us_bundle(ticker)


@st.cache_data(ttl=1_800, show_spinner=False)
def cached_history(ticker: str) -> pd.DataFrame:
    return fetch_history(ticker)


def status_icon(status: str) -> str:
    return {
        "PASS": "✅",
        "CAUTION": "⚠️",
        "FAIL": "❌",
        "QUALIFIED": "✅",
        "WATCH": "⚠️",
        "REJECT FOR NOW": "❌",
        "ACCUMULATE GRADUALLY": "✅",
        "INSUFFICIENT DATA": "⚠️",
        "AVOID FOR NOW": "❌",
        "LONG CANDIDATE": "✅",
        "BULLISH WATCH": "⚠️",
        "MIXED / WAIT": "⚠️",
        "BEARISH / AVOID LONG": "🔻",
    }.get(status, "•")


def gate_frame(result: Any) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Gate": gate.name,
                "Result": f"{status_icon(gate.status)} {gate.status}",
                "Score": "—" if gate.max_score == 0 else f"{gate.score:.1f}/{gate.max_score:.0f}",
                "Evidence": " ".join(gate.evidence[:3]),
            }
            for gate in result.gates
        ]
    )


def price_chart(bundle: MarketBundle, result: Any) -> go.Figure:
    data = add_indicators(bundle.history).tail(150)
    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        )
    )
    for column in ["SMA20", "SMA50", "SMA200"]:
        figure.add_trace(go.Scatter(x=data.index, y=data[column], mode="lines", name=column))
    if result.trade_plan:
        levels = [
            ("Entry", result.trade_plan.entry_trigger, "dash"),
            ("Stop", result.trade_plan.stop, "dot"),
            ("Target 1", result.trade_plan.target_1, "dashdot"),
        ]
        for label, value, dash in levels:
            figure.add_hline(y=value, line_dash=dash, annotation_text=f"{label} {value:.2f}")
    figure.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        hovermode="x unified",
    )
    return figure


def render_directional_view(result: Any) -> None:
    view = result.directional_view
    st.markdown("#### Directional conclusion")
    st.info(f"{status_icon(view.bias)} **{view.bias}**")
    for item in view.evidence:
        st.write(f"- {item}")
    if view.bearish_trigger is not None:
        st.markdown("##### Short-side research scenario")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Strategy": view.bearish_strategy,
                        "Bearish trigger": f"${view.bearish_trigger:.2f}",
                        "Invalidation": f"${view.bearish_invalidation:.2f}",
                        "Downside objective 1": f"${view.bearish_target_1:.2f}",
                        "Downside objective 2": f"${view.bearish_target_2:.2f}",
                    }
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.warning(
            "Short-sale execution and position sizing are disabled. Your active mandate prohibits "
            "shorting; these levels explain the bearish thesis and when it would be wrong."
        )


def render_us_detail(bundle: MarketBundle, result: Any) -> None:
    st.subheader(f"{result.ticker} — {status_icon(result.status)} {result.status}")
    left, middle, right, fourth = st.columns(4)
    left.metric("Model score", f"{result.score:.0f}/100")
    middle.metric("Latest close", f"${result.price:,.2f}")
    right.metric("Strategy", result.strategy)
    fourth.metric("Data through", pd.Timestamp(result.as_of).date().isoformat())

    st.dataframe(gate_frame(result), hide_index=True, use_container_width=True)
    render_directional_view(result)
    st.plotly_chart(price_chart(bundle, result), use_container_width=True)

    if result.trade_plan:
        plan = result.trade_plan
        st.markdown("#### Conditional trade plan")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Entry trigger": f"${plan.entry_trigger:.2f}",
                        "Stop/invalidation": f"${plan.stop:.2f}",
                        "Target 1": f"${plan.target_1:.2f}",
                        "Target 2": f"${plan.target_2:.2f}",
                        "Shares": f"{plan.shares:.3f}",
                        "Notional": f"${plan.notional:.2f}",
                        "Planned risk": f"${plan.planned_risk:.2f}",
                    }
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Recalculate from the actual broker fill. A model trigger is not a market order.")

    thesis_column, risk_column = st.columns(2)
    with thesis_column:
        st.markdown("#### Why it deserves consideration")
        for line in result.thesis:
            st.write(f"- {line}")
    with risk_column:
        st.markdown("#### Strongest bear case")
        for line in result.bear_case:
            st.write(f"- {line}")
        st.markdown("#### What invalidates it")
        for line in result.invalidation:
            st.write(f"- {line}")

    with st.expander("Business model and company profile"):
        fundamentals = bundle.fundamentals
        st.write(
            f"**{fundamentals.get('company_name') or result.ticker}** · "
            f"{fundamentals.get('sector') or 'Sector unavailable'} · "
            f"{fundamentals.get('industry') or 'Industry unavailable'}"
        )
        st.write(
            fundamentals.get("business_summary")
            or "A detailed business description was not returned by the current data source."
        )

    with st.expander("Full gate evidence"):
        for gate in result.gates:
            st.markdown(f"**{status_icon(gate.status)} {gate.name}: {gate.status}**")
            for item in gate.evidence:
                st.write(f"- {item}")

    fundamentals = bundle.fundamentals
    with st.expander("Fundamental data used by the model"):
        rows = []
        for key, label in [
            ("revenue_growth", "Revenue growth"),
            ("earnings_growth", "Earnings growth"),
            ("profit_margin", "Profit margin"),
            ("free_cash_flow", "Free cash flow"),
            ("total_cash", "Total cash"),
            ("total_debt", "Total debt"),
            ("trailing_pe", "Trailing P/E"),
            ("forward_pe", "Forward P/E"),
            ("peg_ratio", "PEG ratio"),
            ("earnings_date", "Next earnings date"),
        ]:
            rows.append({"Metric": label, "Value": fundamentals.get(key)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with st.expander(f"Recent headlines reviewed ({len(result.news)})"):
        if not result.news:
            st.warning("No headlines were returned, so the news gate cannot receive a full pass.")
        else:
            display = pd.DataFrame(result.news)[["published", "publisher", "title", "link"]]
            st.dataframe(
                display,
                hide_index=True,
                use_container_width=True,
                column_config={"link": st.column_config.LinkColumn("Source")},
            )
    st.caption(bundle.source_note)
    for warning in result.warnings:
        st.caption(f"• {warning}")
    st.download_button(
        "Download this analysis as JSON",
        data=json.dumps(result.to_dict(), default=str, indent=2),
        file_name=f"{result.ticker.lower()}-analysis.json",
        mime="application/json",
    )


def saved_gate_frame(report: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Gate": gate["name"],
                "Result": f"{status_icon(gate['status'])} {gate['status']}",
                "Score": "—" if gate["max_score"] == 0 else f"{gate['score']:.1f}/{gate['max_score']:.0f}",
                "Evidence": " ".join(gate.get("evidence", [])[:3]),
            }
            for gate in report.get("gates", [])
        ]
    )


def render_saved_report(report: dict[str, Any]) -> None:
    st.subheader(
        f"{report['ticker']} — {status_icon(report['status'])} {report['status']}"
    )
    first, second, third, fourth = st.columns(4)
    first.metric("Model score", f"{report['score']:.0f}/100")
    second.metric("Closing price", f"${report['price']:,.2f}")
    third.metric("Strategy", report["strategy"])
    direction = report.get("directional_view", {})
    fourth.metric("Directional view", direction.get("bias", "Unavailable"))
    st.dataframe(saved_gate_frame(report), hide_index=True, use_container_width=True)

    plan = report.get("trade_plan")
    if plan:
        st.markdown("#### Conditional long plan")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Entry": plan["entry_trigger"],
                        "Stop": plan["stop"],
                        "Target 1": plan["target_1"],
                        "Target 2": plan["target_2"],
                        "Shares": plan["shares"],
                        "Notional": plan["notional"],
                        "Planned risk": plan["planned_risk"],
                    }
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    if direction:
        st.markdown(f"#### {status_icon(direction.get('bias', ''))} {direction.get('bias')}")
        for item in direction.get("evidence", []):
            st.write(f"- {item}")
        if direction.get("bearish_trigger") is not None:
            st.write(
                f"Bearish research levels: trigger **${direction['bearish_trigger']:.2f}**, "
                f"invalidation **${direction['bearish_invalidation']:.2f}**, objectives "
                f"**${direction['bearish_target_1']:.2f}** and **${direction['bearish_target_2']:.2f}**."
            )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Why consider it")
        for item in report.get("thesis", []):
            st.write(f"- {item}")
    with right:
        st.markdown("#### Bear case and invalidation")
        for item in report.get("bear_case", []):
            st.write(f"- {item}")
        for item in report.get("invalidation", []):
            st.write(f"- {item}")


def load_latest_report() -> dict[str, Any] | None:
    path = Path("reports/latest.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def optional_number(label: str, key: str, help_text: str = "") -> float | None:
    text = st.text_input(label, key=key, help=help_text, placeholder="Leave blank if unknown")
    if not text.strip():
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        st.error(f"{label} must be a number or blank.")
        return None


def optional_bool(label: str, key: str) -> bool | None:
    answer = st.selectbox(label, ["Unknown", "No", "Yes"], key=key)
    return None if answer == "Unknown" else answer == "Yes"


st.title("Stock Research Lab")
st.caption(
    "Transparent US swing-trade qualification and Ghana long-term accumulation scoring. "
    "This tool supports research; it does not place trades."
)

search_tab, us_tab, report_tab, ghana_tab, methodology_tab = st.tabs(
    [
        "Analyze a stock",
        "US watchlist",
        "Closing reports",
        "Ghana long-term",
        "Methodology",
    ]
)

with us_tab:
    with st.sidebar:
        st.header("Risk mandate")
        portfolio = st.number_input("Portfolio value ($)", min_value=100.0, value=1_000.0, step=100.0)
        risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        risk_cap = st.number_input("Dollar risk cap per trade ($)", min_value=1.0, value=5.0, step=1.0)
        max_position = st.number_input("Maximum US position ($)", min_value=25.0, value=200.0, step=25.0)
        current_open_risk = st.number_input("Current total open risk ($)", min_value=0.0, value=0.0, step=1.0)
        current_drawdown = st.number_input("Current portfolio drawdown (%)", min_value=0.0, value=0.0, step=0.5)
        minimum_rr = st.number_input("Minimum reward/risk", min_value=1.0, value=2.0, step=0.25)
        st.caption("Portfolio halt threshold: 10% drawdown. Maximum total open risk: $20.")

    watchlist_text = st.text_input(
        "US watchlist",
        value="MSFT, AMZN, GOOGL, EOG, NVDA, JPM, V, RDW",
        help="Up to eight liquid US tickers. The model does not scan penny stocks.",
    )
    tickers = list(dict.fromkeys(item.strip().upper() for item in watchlist_text.split(",") if item.strip()))[:8]
    if st.button("Run current screen", type="primary", disabled=not tickers):
        config = ModelConfig(
            portfolio_value=portfolio,
            risk_per_trade_pct=risk_pct,
            risk_dollar_cap=risk_cap,
            max_position=max_position,
            current_open_risk=current_open_risk,
            current_drawdown_pct=current_drawdown,
            min_reward_risk=minimum_rr,
        )
        reports = {}
        bundles = {}
        errors = {}
        with st.spinner("Retrieving market, company, calendar, and headline data..."):
            try:
                benchmark = cached_history("SPY")
            except Exception as exc:
                benchmark = None
                st.error(f"SPY benchmark failed: {exc}")
            if benchmark is not None:
                for ticker in tickers:
                    try:
                        bundle = cached_bundle(ticker)
                        report = analyze_us_swing(
                            ticker,
                            bundle.history,
                            bundle.fundamentals,
                            bundle.news,
                            benchmark,
                            config,
                        )
                        bundles[ticker] = bundle
                        reports[ticker] = report
                    except Exception as exc:
                        errors[ticker] = str(exc)
        st.session_state["us_reports"] = reports
        st.session_state["us_bundles"] = bundles
        st.session_state["us_errors"] = errors

    reports = st.session_state.get("us_reports", {})
    bundles = st.session_state.get("us_bundles", {})
    errors = st.session_state.get("us_errors", {})
    if errors:
        with st.expander("Data errors"):
            for ticker, message in errors.items():
                st.write(f"- **{ticker}:** {message}")
    if reports:
        ranking = pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Status": f"{status_icon(report.status)} {report.status}",
                    "Score": report.score,
                    "Strategy": report.strategy,
                    "Close": report.price,
                    "Entry": report.trade_plan.entry_trigger if report.trade_plan else None,
                    "Stop": report.trade_plan.stop if report.trade_plan else None,
                    "Target": report.trade_plan.target_1 if report.trade_plan else None,
                }
                for ticker, report in sorted(reports.items(), key=lambda item: item[1].score, reverse=True)
            ]
        )
        st.markdown("#### Ranked decision table")
        st.dataframe(ranking, hide_index=True, use_container_width=True)
        selected = st.selectbox("Open detailed evidence", ranking["Ticker"].tolist())
        render_us_detail(bundles[selected], reports[selected])
    else:
        st.info("Run the screen to retrieve current data and apply all strategy gates.")

with search_tab:
    st.subheader("Research any US-listed stock")
    st.write(
        "Enter a ticker such as RDW. The report applies the same technical, fundamental, "
        "valuation, news, liquidity, and portfolio-risk gates as the watchlist screen."
    )
    searched_ticker = st.text_input(
        "Ticker symbol",
        value="RDW",
        key="single_ticker",
        help="Use the exchange ticker, for example RDW, MSFT, or JPM.",
    ).strip().upper()
    if st.button("Run full stock analysis", type="primary", disabled=not searched_ticker):
        config = ModelConfig(
            portfolio_value=portfolio,
            risk_per_trade_pct=risk_pct,
            risk_dollar_cap=risk_cap,
            max_position=max_position,
            current_open_risk=current_open_risk,
            current_drawdown_pct=current_drawdown,
            min_reward_risk=minimum_rr,
        )
        with st.spinner(f"Building the evidence report for {searched_ticker}..."):
            try:
                single_benchmark = cached_history("SPY")
                single_bundle = cached_bundle(searched_ticker)
                single_report = analyze_us_swing(
                    searched_ticker,
                    single_bundle.history,
                    single_bundle.fundamentals,
                    single_bundle.news,
                    single_benchmark,
                    config,
                )
                st.session_state["single_bundle"] = single_bundle
                st.session_state["single_report"] = single_report
                st.session_state.pop("single_error", None)
            except Exception as exc:
                st.session_state["single_error"] = str(exc)
                st.session_state.pop("single_bundle", None)
                st.session_state.pop("single_report", None)

    if st.session_state.get("single_error"):
        st.error(
            f"The data source could not complete this ticker: {st.session_state['single_error']}"
        )
    if st.session_state.get("single_report"):
        render_us_detail(
            st.session_state["single_bundle"], st.session_state["single_report"]
        )

with report_tab:
    st.subheader("Latest scheduled closing report")
    latest_report = load_latest_report()
    if latest_report is None:
        st.info(
            "No scheduled report has been generated yet. After the GitHub workflow is installed, "
            "run it once manually from GitHub Actions or wait for the next weekday close."
        )
    else:
        st.caption(
            f"Generated {latest_report.get('generated_at', 'time unavailable')} · "
            f"Market date {latest_report.get('market_date', 'unavailable')}"
        )
        if latest_report.get("market_was_closed"):
            st.warning(
                "The market was closed on the workflow run date; this report uses the latest "
                "available completed session."
            )
        action = latest_report.get("action", "REVIEW REPORT")
        if action == "NO TRADE":
            st.warning("**Closing decision: NO TRADE** — no stock passed every hard gate.")
        else:
            st.success(f"**Closing decision: {action}**")
        summary = latest_report.get("summary", {})
        first, second, third, fourth = st.columns(4)
        first.metric("Qualified", summary.get("qualified", 0))
        second.metric("Watch", summary.get("watch", 0))
        third.metric("Rejected", summary.get("rejected", 0))
        fourth.metric("Data errors", len(latest_report.get("errors", {})))

        scheduled_reports = latest_report.get("reports", [])
        if scheduled_reports:
            ranking = pd.DataFrame(
                [
                    {
                        "Ticker": report["ticker"],
                        "Decision": report["status"],
                        "Directional view": report.get("directional_view", {}).get("bias"),
                        "Score": report["score"],
                        "Close": report["price"],
                        "Strategy": report["strategy"],
                    }
                    for report in scheduled_reports
                ]
            )
            st.dataframe(ranking, hide_index=True, use_container_width=True)
            selected_ticker = st.selectbox(
                "Open a scheduled report",
                ranking["Ticker"].tolist(),
                key="scheduled_selected",
            )
            selected_report = next(
                report for report in scheduled_reports if report["ticker"] == selected_ticker
            )
            render_saved_report(selected_report)
        changes = latest_report.get("changes_from_previous_close", [])
        with st.expander(f"Changes from previous closing report ({len(changes)})"):
            if changes:
                st.dataframe(pd.DataFrame(changes), hide_index=True, use_container_width=True)
            else:
                st.write("No tracked decision, score, level, or earnings-date changes.")
        if latest_report.get("errors"):
            with st.expander("Scheduled data errors"):
                for symbol, message in latest_report["errors"].items():
                    st.write(f"- **{symbol}:** {message}")
        st.caption(latest_report.get("source_note", ""))

with ghana_tab:
    st.subheader("Ghana long-term evidence form")
    st.write(
        "Enter values from the latest official GSE filing and the current IC Wealth quote. "
        "Unknown fields stay blank; the model will report data completeness instead of inventing a pass."
    )
    with st.form("ghana_form"):
        ticker = st.text_input("Ticker", value="MTNGH").strip().upper()
        first, second, third = st.columns(3)
        with first:
            revenue = optional_number("Revenue growth (%)", "gh_revenue")
            earnings = optional_number("Earnings growth (%)", "gh_earnings")
            roe = optional_number("Return on equity (%)", "gh_roe")
            pe = optional_number("P/E ratio", "gh_pe")
        with second:
            dividend = optional_number("Dividend yield (%)", "gh_dividend")
            debt_equity = optional_number("Debt/equity", "gh_de")
            liquidity = optional_number("Average daily traded value (GHS)", "gh_liquidity")
            timing = optional_number("Price vs 200-day average (%)", "gh_timing")
        with third:
            cashflow = optional_bool("Is operating cash flow positive?", "gh_cashflow")
            governance = optional_bool("Unresolved governance concern?", "gh_governance")
            regulatory = optional_bool("Unresolved regulatory concern?", "gh_regulatory")
        submitted = st.form_submit_button("Score Ghana candidate", type="primary")
    if submitted:
        ghana_result = analyze_ghana_long_term(
            {
                "ticker": ticker,
                "revenue_growth_pct": revenue,
                "earnings_growth_pct": earnings,
                "roe_pct": roe,
                "pe_ratio": pe,
                "dividend_yield_pct": dividend,
                "operating_cashflow_positive": cashflow,
                "debt_to_equity": debt_equity,
                "governance_flag": governance,
                "regulatory_flag": regulatory,
                "avg_daily_value_ghs": liquidity,
                "price_vs_200d_pct": timing,
            }
        )
        st.subheader(f"{ticker} — {status_icon(ghana_result['status'])} {ghana_result['status']}")
        score_col, completeness_col = st.columns(2)
        score_col.metric("Long-term score", f"{ghana_result['score']:.0f}/100")
        completeness_col.metric("Data completeness", f"{ghana_result['completeness_pct']}%")
        gate_rows = []
        for gate in ghana_result["gates"]:
            gate_rows.append(
                {
                    "Gate": gate["gate"],
                    "Result": f"{status_icon(gate['status'])} {gate['status']}",
                    "Score": f"{gate['score']:.1f}/{gate['max_score']}",
                    "Evidence": " ".join(gate["evidence"]),
                }
            )
        st.dataframe(pd.DataFrame(gate_rows), hide_index=True, use_container_width=True)
        for warning in ghana_result["warnings"]:
            st.caption(f"• {warning}")
        st.download_button(
            "Download Ghana score as JSON",
            json.dumps(ghana_result, indent=2),
            file_name=f"{ticker.lower()}-ghana-long-term.json",
            mime="application/json",
        )

with methodology_tab:
    st.subheader("Qualification logic")
    st.markdown(
        """
**US score (100 points):** technical 30, fundamentals 25, news/catalysts 15,
valuation 10, liquidity/execution 10, and risk/setup 10.

**Qualified** requires a score of at least 75 plus hard passes for technicals,
fundamentals, news/catalysts, liquidity, and risk. The SPY market regime cannot
be in a failed state. Valuation affects the score but is not a standalone hard gate.

The engine tests three distinct setups—trend pullback, confirmed breakout, and
recovery/reclaim—and reports only the highest-scoring applicable setup. It sizes
the position using the smaller of the risk budget and the maximum-position cap.

The directional layer can classify a stock as long candidate, bullish watch,
mixed/wait, or bearish/avoid long. Bearish breakdown levels are research only:
short-sale execution remains disabled under the active no-shorting mandate.

Scheduled U.S. reports are generated by GitHub Actions after the market close and
saved into the repository for this site to display. They use the same deterministic
engine and do not replace primary-source review.

**Ghana score (100 points):** fundamentals 35, valuation 20, balance sheet and
cash flow 15, news/governance 15, liquidity 10, and technical entry timing 5.
Ghana output is long-term accumulation only.
"""
    )
    st.warning(
        "Data-source limitations are intentional hard constraints. Missing news or financial data "
        "cannot silently become a pass. Read primary filings before acting."
    )
