from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_model import ModelConfig, analyze_ghana_long_term, analyze_us_swing
from stock_model.data import MarketBundle, fetch_history, fetch_us_bundle
from stock_model.ghana_data import GhanaResearchBundle, fetch_ghana_bundle
from stock_model.indicators import add_indicators


st.set_page_config(page_title="Stock Research Lab", page_icon="📈", layout="wide")


@st.cache_data(ttl=1_800, show_spinner=False)
def cached_bundle(ticker: str) -> MarketBundle:
    return fetch_us_bundle(ticker)


@st.cache_data(ttl=1_800, show_spinner=False)
def cached_history(ticker: str) -> pd.DataFrame:
    return fetch_history(ticker)


@st.cache_data(ttl=21_600, show_spinner=False)
def cached_ghana_bundle(ticker: str) -> GhanaResearchBundle:
    return fetch_ghana_bundle(ticker)


def status_icon(status: str) -> str:
    return {
        "PASS": "✅",
        "CAUTION": "⚠️",
        "FAIL": "❌",
        "QUALIFIED": "✅",
        "WATCH": "⚠️",
        "ACCUMULATE GRADUALLY": "✅",
        "INSUFFICIENT DATA": "⚠️",
        "AVOID FOR NOW": "❌",
        "LONG CANDIDATE": "✅",
        "BULLISH WATCH": "⚠️",
        "MIXED / WAIT": "⚠️",
        "BEARISH / AVOID LONG": "🔻",
        "STRONG LONG": "🟢",
        "LONG": "🟢",
        "STRONG SHORT": "🔴",
        "SHORT": "🔴",
        "NO TRADE": "⚪",
        "REJECT FOR NOW": "⛔",
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
    first, second, third = st.columns(3)
    first.metric("Recommendation", result.recommendation)
    second.metric("Long score", f"{result.long_score:.0f}/100")
    third.metric("Short score", f"{result.short_score:.0f}/100")
    st.info(f"{status_icon(result.recommendation)} **{result.recommendation}** · confidence {result.confidence}")
    for item in view.evidence:
        st.write(f"- {item}")
    if view.short_risk_flags:
        st.warning("**Short-specific risk flags:** " + ", ".join(view.short_risk_flags))


def render_us_detail(bundle: MarketBundle, result: Any) -> None:
    st.subheader(f"{result.ticker} — {status_icon(result.recommendation)} {result.recommendation}")
    left, middle, right, fourth = st.columns(4)
    left.metric("Selected score", f"{result.score:.0f}/100")
    middle.metric("Latest close", f"${result.price:,.2f}")
    right.metric("Confidence", result.confidence)
    fourth.metric("Strategy", result.strategy)

    st.dataframe(gate_frame(result), hide_index=True, use_container_width=True)
    render_directional_view(result)
    st.plotly_chart(price_chart(bundle, result), use_container_width=True)

    if result.trade_plan:
        plan = result.trade_plan
        st.markdown(f"#### Conditional {plan.direction.lower()} trade plan")
        breakeven_note = "" if plan.direction == "LONG" else "Short sale: no option premium/breakeven is assumed."
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Direction": plan.direction,
                        "Entry trigger": f"${plan.entry_trigger:.2f}",
                        "Stop / invalidation": f"${plan.stop:.2f}",
                        "Target 1": f"${plan.target_1:.2f}",
                        "Target 2": f"${plan.target_2:.2f}",
                        "Target 3": f"${plan.target_3:.2f}",
                        "Risk/share": f"${plan.risk_per_share:.2f}",
                        "R:R to T1": f"{plan.reward_risk_1:.1f}:1",
                        "Shares": f"{plan.shares:.3f}",
                        "Notional": f"${plan.notional:.2f}",
                        "Planned max risk": f"${plan.planned_risk:.2f}",
                    }
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Recalculate from the actual broker fill. " + breakeven_note)
    else:
        st.info("No executable setup was generated because the selected direction did not clear the model's hard gates.")

    thesis_column, risk_column = st.columns(2)
    with thesis_column:
        st.markdown("#### Trade thesis")
        for line in result.thesis:
            st.write(f"- {line}")
    with risk_column:
        st.markdown("#### Risks and invalidation")
        for line in result.bear_case:
            st.write(f"- {line}")
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
    with st.expander("Fundamental and short-interest data used by the model"):
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
            ("short_percent_of_float", "Short % of float"),
            ("short_ratio", "Short ratio / days to cover"),
            ("shares_short", "Shares short"),
            ("float_shares", "Float shares"),
            ("beta", "Beta"),
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
    recommendation = report.get("recommendation", report.get("status", "NO TRADE"))
    st.subheader(f"{report['ticker']} — {status_icon(recommendation)} {recommendation}")
    first, second, third, fourth = st.columns(4)
    first.metric("Selected score", f"{report.get('score', 0):.0f}/100")
    second.metric("Long score", f"{report.get('long_score', 0):.0f}/100")
    third.metric("Short score", f"{report.get('short_score', 0):.0f}/100")
    fourth.metric("Confidence", report.get("confidence", "LOW"))
    st.dataframe(saved_gate_frame(report), hide_index=True, use_container_width=True)

    plan = report.get("trade_plan")
    if plan:
        st.markdown(f"#### Conditional {plan.get('direction', 'trade').lower()} plan")
        st.dataframe(pd.DataFrame([plan]), hide_index=True, use_container_width=True)

    direction = report.get("directional_view", {})
    if direction:
        st.markdown(f"#### {status_icon(direction.get('bias', ''))} {direction.get('bias')}")
        for item in direction.get("evidence", []):
            st.write(f"- {item}")
        flags = direction.get("short_risk_flags") or []
        if flags:
            st.warning("**Short-specific risk flags:** " + ", ".join(flags))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Trade thesis")
        for item in report.get("thesis", []):
            st.write(f"- {item}")
    with right:
        st.markdown("#### Risks and invalidation")
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


def load_latest_ghana_report() -> dict[str, Any] | None:
    path = Path("reports/ghana_latest.json")
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


def ghana_rationale(bundle: GhanaResearchBundle) -> tuple[list[str], list[str], list[str]]:
    data = bundle.inputs
    thesis: list[str] = []
    risks: list[str] = []
    revenue = data.get("revenue_growth_pct")
    earnings = data.get("earnings_growth_pct")
    roe = data.get("roe_pct")
    pe = data.get("pe_ratio")
    dividend = data.get("dividend_yield_pct")
    timing = data.get("price_vs_200d_pct")
    liquidity = data.get("avg_daily_value_ghs")

    if revenue is not None and earnings is not None:
        thesis.append(
            f"Reported revenue growth is {revenue:+.1f}% and net-income growth is "
            f"{earnings:+.1f}%; the model rewards positive, earnings-backed expansion."
        )
    if roe is not None:
        thesis.append(f"Reported return on equity is {roe:.1f}%, a direct profitability check.")
    if pe is not None or dividend is not None:
        parts = []
        if pe is not None:
            parts.append(f"P/E {pe:.1f}x")
        if dividend is not None:
            parts.append(f"dividend yield {dividend:.1f}%")
        thesis.append("Valuation context: " + " and ".join(parts) + ".")
    if timing is not None:
        thesis.append(
            f"The delayed close is {timing:+.1f}% versus the 200-day average; "
            "this affects entry timing, not business quality."
        )

    if revenue is not None and revenue <= 0:
        risks.append("Revenue is contracting, weakening the long-term compounding case.")
    if earnings is not None and earnings <= 0:
        risks.append("Net income is contracting or negative despite any top-line progress.")
    if data.get("operating_cashflow_positive") is False:
        risks.append("Operating cash flow is not positive; earnings may not be converting to cash.")
    if data.get("governance_flag") is True or data.get("regulatory_flag") is True:
        risks.append("The official-headline screen found language requiring governance or regulatory review.")
    if liquidity is not None and liquidity < 250_000:
        risks.append("Estimated daily traded value is low, so exits may be slow and limit orders are essential.")
    if timing is not None and timing > 20:
        risks.append("Price is more than 20% above its 200-day average, increasing entry-timing risk.")
    if not risks:
        risks.append(
            "The strongest general risk is stale or incomplete Ghana market data; the IC Wealth "
            "quote and official filing remain controlling evidence."
        )

    invalidation = [
        "Revenue and earnings quality deteriorate materially in the next official filing.",
        "Operating cash flow turns negative or leverage rises beyond the acceptable range.",
        "A material unresolved governance or regulatory issue is confirmed.",
        "Liquidity falls below a practical level for gradual accumulation and eventual exit.",
    ]
    return thesis, risks, invalidation


def render_ghana_detail(bundle: GhanaResearchBundle, result: dict[str, Any]) -> None:
    st.subheader(f"{bundle.ticker} — {status_icon(result['status'])} {result['status']}")
    first, second, third, fourth = st.columns(4)
    first.metric("Long-term score", f"{result['score']:.0f}/100")
    second.metric("Data completeness", f"{result['completeness_pct']}%")
    third.metric(
        "Delayed close",
        "Unknown" if bundle.price is None else f"GHS {bundle.price:,.2f}",
    )
    fourth.metric("Price date", bundle.price_as_of or "Unknown")
    st.caption(f"{bundle.company_name} · Retrieved {bundle.fetched_at}")

    gate_rows = [
        {
            "Gate": gate["gate"],
            "Result": f"{status_icon(gate['status'])} {gate['status']}",
            "Score": f"{gate['score']:.1f}/{gate['max_score']}",
            "Evidence": " ".join(gate["evidence"]) or "Required evidence unavailable.",
        }
        for gate in result["gates"]
    ]
    st.markdown("#### Gate decision")
    st.dataframe(pd.DataFrame(gate_rows), hide_index=True, use_container_width=True)

    thesis, risks, invalidation = ghana_rationale(bundle)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Why it deserves consideration")
        if bundle.business_summary:
            st.write(bundle.business_summary)
            if bundle.industry:
                st.caption(f"Industry: {bundle.industry}")
        for item in thesis:
            st.write(f"- {item}")
    with right:
        st.markdown("#### Strongest bear case")
        for item in risks:
            st.write(f"- {item}")
        st.markdown("#### What invalidates the thesis")
        for item in invalidation:
            st.write(f"- {item}")

    if result["status"] == "ACCUMULATE GRADUALLY":
        st.success(
            "The evidence passes for gradual long-term accumulation. Confirm the live IC Wealth "
            "quote, use limit orders, and divide the intended allocation into scheduled tranches. "
            "Do not add merely because price falls."
        )
    elif result["status"] == "WATCH":
        st.warning("Do not accumulate yet. Wait for the caution or failed gates to improve.")
    elif result["status"] == "INSUFFICIENT DATA":
        st.warning("NO BUY DECISION: too much required evidence is unavailable to justify accumulation.")
    else:
        st.error("Avoid for now; one or more material long-term gates failed.")

    st.markdown("#### Data used and source trail")
    evidence_rows = [
        {
            "Metric": item.label,
            "Value": item.display,
            "As of": item.as_of or "Unknown",
            "Provider": item.source,
            "Source": item.url,
        }
        for item in bundle.metrics
    ]
    st.dataframe(
        pd.DataFrame(evidence_rows),
        hide_index=True,
        use_container_width=True,
        column_config={"Source": st.column_config.LinkColumn("Source")},
    )

    with st.expander(f"Official issuer/regulatory results reviewed ({len(bundle.headlines)})"):
        if bundle.headlines:
            st.dataframe(
                pd.DataFrame(bundle.headlines),
                hide_index=True,
                use_container_width=True,
                column_config={"link": st.column_config.LinkColumn("Source")},
            )
        else:
            st.write("No matching results were returned. Check the review-completeness rows above.")

    for warning in bundle.warnings + result["warnings"]:
        st.caption(f"• {warning}")
    export = {
        "analysis": result,
        "company_name": bundle.company_name,
        "industry": bundle.industry,
        "business_summary": bundle.business_summary,
        "price": bundle.price,
        "price_as_of": bundle.price_as_of,
        "fetched_at": bundle.fetched_at,
        "metrics": [item.to_dict() for item in bundle.metrics],
        "headlines": bundle.headlines,
        "rationale": {"thesis": thesis, "risks": risks, "invalidation": invalidation},
    }
    st.download_button(
        "Download Ghana research as JSON",
        json.dumps(export, default=str, indent=2),
        file_name=f"{bundle.ticker.lower()}-ghana-long-term.json",
        mime="application/json",
    )


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
        allow_shorts = st.checkbox("Enable short-trade analysis", value=True, help="The model will score short setups and can generate a conditional short plan when all short hard gates pass.")
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
            allow_shorts=allow_shorts,
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
                    "Recommendation": f"{status_icon(report.recommendation)} {report.recommendation}",
                    "Confidence": report.confidence,
                    "Long": report.long_score,
                    "Short": report.short_score,
                    "Strategy": report.strategy,
                    "Close": report.price,
                    "Entry": report.trade_plan.entry_trigger if report.trade_plan else None,
                    "Stop": report.trade_plan.stop if report.trade_plan else None,
                    "Target": report.trade_plan.target_1 if report.trade_plan else None,
                }
                for ticker, report in sorted(reports.items(), key=lambda item: max(item[1].long_score, item[1].short_score), reverse=True)
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
        "Enter a ticker such as RDW. The report scores both a long and short thesis, then "
        "selects LONG, SHORT, or NO TRADE using the same technical, fundamental, valuation, "
        "news, liquidity, squeeze-risk, and portfolio-risk framework."
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
            allow_shorts=allow_shorts,
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
            st.warning("**Closing decision: NO TRADE** — no stock passed the directional hard gates.")
        else:
            st.success(f"**Closing decision: {action}**")
        summary = latest_report.get("summary", {})
        first, second, third, fourth, fifth = st.columns(5)
        first.metric("Long setups", summary.get("qualified_long", 0))
        second.metric("Short setups", summary.get("qualified_short", 0))
        third.metric("Watch", summary.get("watch", 0))
        fourth.metric("No trade", summary.get("no_trade", 0))
        fifth.metric("Data errors", len(latest_report.get("errors", {})))

        scheduled_reports = latest_report.get("reports", [])
        if scheduled_reports:
            ranking = pd.DataFrame(
                [
                    {
                        "Ticker": report["ticker"],
                        "Recommendation": report.get("recommendation", report["status"]),
                        "Confidence": report.get("confidence"),
                        "Long": report.get("long_score"),
                        "Short": report.get("short_score"),
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
    st.subheader("Automated Ghana long-term research")
    st.write(
        "Enter a GSE ticker. The app retrieves delayed market and financial data, reviews matching "
        "GSE/SEC evidence, and applies the long-term accumulation gates automatically."
    )
    scheduled_ghana = load_latest_ghana_report()
    if scheduled_ghana:
        with st.expander("Latest scheduled Ghana watchlist", expanded=True):
            st.caption(f"Generated {scheduled_ghana.get('generated_at', 'time unavailable')}")
            scheduled_rows = [
                {
                    "Ticker": report["ticker"],
                    "Decision": f"{status_icon(report['status'])} {report['status']}",
                    "Score": report["score"],
                    "Completeness": f"{report['completeness_pct']}%",
                    "Delayed close": report.get("price"),
                    "Price date": report.get("price_as_of"),
                }
                for report in scheduled_ghana.get("reports", [])
            ]
            if scheduled_rows:
                st.dataframe(pd.DataFrame(scheduled_rows), hide_index=True, use_container_width=True)
            if scheduled_ghana.get("errors"):
                for symbol, message in scheduled_ghana["errors"].items():
                    st.caption(f"• {symbol}: {message}")
            st.caption(scheduled_ghana.get("source_note", ""))
    else:
        st.info(
            "The scheduled Ghana watchlist will appear here after the updated GitHub workflow "
            "completes once. You can run an individual ticker immediately below."
        )
    ticker = st.text_input(
        "GSE ticker",
        value="MTNGH",
        key="ghana_ticker",
        help="Examples: MTNGH, GCB, EGH, SCB, GOIL, or UNIL.",
    ).strip().upper()
    if st.button("Run Ghana long-term analysis", type="primary", disabled=not ticker):
        with st.spinner(f"Retrieving financial, market, GSE, and SEC evidence for {ticker}..."):
            try:
                ghana_bundle = cached_ghana_bundle(ticker)
                ghana_result = analyze_ghana_long_term(ghana_bundle.inputs)
                st.session_state["ghana_bundle"] = ghana_bundle
                st.session_state["ghana_result"] = ghana_result
                st.session_state.pop("ghana_error", None)
            except Exception as exc:
                st.session_state["ghana_error"] = str(exc)
                st.session_state.pop("ghana_bundle", None)
                st.session_state.pop("ghana_result", None)
    if st.session_state.get("ghana_error"):
        st.error(
            "The Ghana sources could not complete this ticker. No result was manufactured. "
            f"Details: {st.session_state['ghana_error']}"
        )
    if st.session_state.get("ghana_result"):
        render_ghana_detail(
            st.session_state["ghana_bundle"], st.session_state["ghana_result"]
        )

with methodology_tab:
    st.subheader("Qualification logic")
    st.markdown(
        """
**US direction-neutral scoring:** the model calculates separate Long and Short scores.
Long: technical 30, fundamentals 25, news/catalysts 15, valuation 10, liquidity 10,
and risk/setup 10. Short: technical 30, fundamentals 25, news/catalysts 15,
valuation 10, liquidity 5, short-squeeze risk 5, and risk/setup 10.

**Trade selection:** a direction must reach at least 75 and pass its hard gates.
When both directions qualify, the higher score must lead by at least five points;
otherwise the result is NO TRADE. SPY market regime blocks new longs when it fails
but does not automatically block shorts. Short borrow/locate availability is not
verified by the current adapter and remains broker-controlled.

**Ghana score (100 points):** fundamentals 35, valuation 20, balance sheet and
cash flow 15, news/governance 15, liquidity 10, and technical entry timing 5.
Ghana output is long-term accumulation only.
"""
    )
    st.warning(
        "Data-source limitations are intentional hard constraints. Missing news or financial data "
        "cannot silently become a pass. Read primary filings before acting."
    )
