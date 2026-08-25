from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
import re
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


STOCK_ANALYSIS_ROOT = "https://stockanalysis.com/quote/ghse"
GSE_SEARCH_ROOT = "https://gse.com.gh"
SEC_SEARCH_ROOT = "https://sec.gov.gh"


# Search aliases improve the official-feed review when the listed name differs from
# the exchange ticker. An unknown ticker still works; the symbol itself is searched.
ISSUER_SEARCH_TERMS = {
    "ACCESS": "Access Bank Ghana",
    "ADB": "Agricultural Development Bank",
    "BOPP": "Benso Oil Palm Plantation",
    "CAL": "CalBank",
    "CPC": "Cocoa Processing Company",
    "DASPHARMA": "Dannex Ayrton Starwin",
    "EGH": "Ecobank Ghana",
    "EGL": "Enterprise Group",
    "FML": "Fan Milk",
    "GCB": "GCB Bank",
    "GGBL": "Guinness Ghana Breweries",
    "GOIL": "GOIL",
    "MTNGH": "Scancom MTN Ghana",
    "RBGH": "Republic Bank Ghana",
    "SCB": "Standard Chartered Bank Ghana",
    "SIC": "SIC Insurance",
    "SOGEGH": "Societe Generale Ghana",
    "TOTAL": "TotalEnergies Ghana",
    "UNIL": "Unilever Ghana",
}


@dataclass
class GhanaMetric:
    key: str
    label: str
    value: Any
    display: str
    as_of: str | None
    source: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "display": self.display,
            "as_of": self.as_of,
            "source": self.source,
            "url": self.url,
        }


@dataclass
class GhanaResearchBundle:
    ticker: str
    company_name: str
    industry: str | None
    business_summary: str | None
    price: float | None
    price_as_of: str | None
    inputs: dict[str, Any]
    metrics: list[GhanaMetric]
    headlines: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class _DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.title: list[str] = []
        self.rows: list[list[str]] = []
        self.links: list[tuple[str, str]] = []
        self._skip_depth = 0
        self._in_title = False
        self._in_row = False
        self._in_cell = False
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._current_href: str | None = None
        self._current_link: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag == "tr":
            self._in_row = True
            self._current_row = []
        elif tag in {"th", "td"} and self._in_row:
            self._in_cell = True
            self._current_cell = []
        elif tag == "a":
            self._current_href = dict(attrs).get("href")
            self._current_link = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag in {"th", "td"} and self._in_cell:
            self._current_row.append(_clean(" ".join(self._current_cell)))
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if len(self._current_row) >= 2:
                self.rows.append(self._current_row)
            self._in_row = False
        elif tag == "a" and self._current_href is not None:
            self.links.append((self._current_href, _clean(" ".join(self._current_link))))
            self._current_href = None
            self._current_link = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        value = _clean(data)
        if not value:
            return
        self.text.append(value)
        if self._in_title:
            self.title.append(value)
        if self._in_cell:
            self._current_cell.append(value)
        if self._current_href is not None:
            self._current_link.append(value)


class _HttpClient:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    }

    def get_text(self, url: str, timeout: int = 20) -> str:
        request = Request(url, headers=self.headers)
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")


def _fetch_html(client: _HttpClient, url: str, timeout: int = 20) -> str:
    content = client.get_text(url, timeout=timeout)
    if not content.strip():
        raise ValueError(f"Empty response from {url}")
    return content


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _table_rows(html: str) -> dict[str, list[str]]:
    parser = _DocumentParser()
    parser.feed(html)
    rows: dict[str, list[str]] = {}
    for cells in parser.rows:
        if len(cells) >= 2 and cells[0]:
            rows.setdefault(cells[0].lower(), cells[1:])
    return rows


def _first(rows: dict[str, list[str]], *labels: str) -> str | None:
    for label in labels:
        values = rows.get(label.lower())
        if values:
            value = next((item for item in values if item not in {"", "-", "n/a", "—"}), None)
            if value is not None:
                return value
    return None


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.replace(",", "").replace("GH₵", "").replace("GHS", "").strip()
    if text.lower() in {"", "-", "n/a", "—"}:
        return None
    multiplier = 1.0
    suffix = text[-1:].upper()
    if suffix in {"K", "M", "B", "T"}:
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[suffix]
        text = text[:-1]
    text = text.replace("%", "").replace("×", "").strip()
    try:
        return float(text) * multiplier
    except ValueError:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        return float(match.group()) * multiplier if match else None


def _date_from_text(text: str, label: str) -> str | None:
    match = re.search(
        rf"{re.escape(label)}\s*:?\s*([A-Z][a-z]{{2}}\s+\d{{1,2}},\s+\d{{4}})",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%b %d, %Y").date().isoformat()
    except ValueError:
        return match.group(1)


def _company_and_price(html: str, ticker: str) -> tuple[str, float | None, str | None]:
    parser = _DocumentParser()
    parser.feed(html)
    text = _clean(" ".join(parser.text))
    title = _clean(" ".join(parser.title)) or ticker
    company_match = re.match(r"(.+?)\s+(?:Statistics|\(GHSE:)", title, re.IGNORECASE)
    company = _clean(company_match.group(1)) if company_match else ticker
    price_match = re.search(
        r"Currency is GHS\s+([0-9][0-9,.]*)", text, re.IGNORECASE
    )
    price = _number(price_match.group(1)) if price_match else None
    price_date = _date_from_text(text, "At close")
    return company, price, price_date


def _source_as_of(html: str) -> str | None:
    parser = _DocumentParser()
    parser.feed(html)
    text = _clean(" ".join(parser.text))
    return _date_from_text(text, "Last updated")


def _parse_profile(html: str) -> tuple[str | None, str | None, str | None]:
    parser = _DocumentParser()
    parser.feed(html)
    text = _clean(" ".join(parser.text))
    rows = _table_rows(html)
    industry = _first(rows, "Industry")
    match = re.search(
        r"Company Description\s+(.+?)(?:\s+Country\s+Ghana|\s+Contact Details)",
        text,
        re.IGNORECASE,
    )
    description = _clean(match.group(1)) if match else None
    if description and len(description) > 1_200:
        description = description[:1_197].rstrip() + "..."
    return industry, description, _source_as_of(html)


def _format_number(value: float | None, kind: str) -> str:
    if value is None:
        return "Unknown"
    if kind == "pct":
        return f"{value:+.2f}%"
    if kind == "ratio":
        return f"{value:.2f}x"
    if kind == "money":
        return f"GHS {value:,.2f}"
    if kind == "money0":
        return f"GHS {value:,.0f}"
    return f"{value:,.2f}"


def parse_stockanalysis_pages(
    ticker: str, statistics_html: str, income_html: str
) -> tuple[str, float | None, str | None, dict[str, Any], list[GhanaMetric]]:
    """Extract only labeled, auditable fields from two rendered statistics pages."""
    symbol = ticker.upper()
    statistics_url = f"{STOCK_ANALYSIS_ROOT}/{symbol}/statistics/"
    income_url = f"{STOCK_ANALYSIS_ROOT}/{symbol}/financials/income-statement/"
    company, price, price_date = _company_and_price(statistics_html, symbol)
    stats_rows = _table_rows(statistics_html)
    income_rows = _table_rows(income_html)
    stats_as_of = _source_as_of(statistics_html) or price_date
    income_as_of = _source_as_of(income_html)

    revenue = _number(_first(income_rows, "Revenue Growth"))
    earnings = _number(_first(income_rows, "Net Income Growth", "EPS Growth"))
    roe = _number(_first(stats_rows, "Return on Equity (ROE)", "ROE"))
    pe = _number(_first(stats_rows, "PE Ratio", "P/E Ratio", "P/E"))
    dividend = _number(_first(stats_rows, "Dividend Yield", "Trailing Dividend Yield"))
    debt_equity = _number(_first(stats_rows, "Debt / Equity", "Debt / Equity Ratio"))
    ocf = _number(_first(stats_rows, "Operating Cash Flow"))
    sma200 = _number(_first(stats_rows, "200-Day Moving Average"))
    avg_volume = _number(
        _first(stats_rows, "Average Volume (20 Days)", "Average Volume (30 Days)")
    )
    timing = None
    if price is not None and sma200 not in {None, 0}:
        timing = (price / sma200 - 1) * 100
    traded_value = price * avg_volume if price is not None and avg_volume is not None else None

    inputs: dict[str, Any] = {
        "ticker": symbol,
        "_automated": True,
        "revenue_growth_pct": revenue,
        "earnings_growth_pct": earnings,
        "roe_pct": roe,
        "pe_ratio": pe,
        "dividend_yield_pct": dividend,
        "operating_cashflow_positive": None if ocf is None else ocf > 0,
        "debt_to_equity": debt_equity,
        "avg_daily_value_ghs": traded_value,
        "price_vs_200d_pct": timing,
    }

    metric_specs = [
        ("price", "Delayed closing price", price, _format_number(price, "money"), price_date, statistics_url),
        ("revenue_growth_pct", "Revenue growth", revenue, _format_number(revenue, "pct"), income_as_of, income_url),
        ("earnings_growth_pct", "Net-income growth", earnings, _format_number(earnings, "pct"), income_as_of, income_url),
        ("roe_pct", "Return on equity", roe, _format_number(roe, "pct"), stats_as_of, statistics_url),
        ("pe_ratio", "Trailing P/E", pe, _format_number(pe, "ratio"), stats_as_of, statistics_url),
        ("dividend_yield_pct", "Dividend yield", dividend, _format_number(dividend, "pct"), stats_as_of, statistics_url),
        ("operating_cashflow_positive", "Operating cash flow positive", None if ocf is None else ocf > 0, "Unknown" if ocf is None else ("Yes" if ocf > 0 else "No"), stats_as_of, statistics_url),
        ("debt_to_equity", "Debt/equity", debt_equity, _format_number(debt_equity, "ratio"), stats_as_of, statistics_url),
        ("avg_daily_value_ghs", "Estimated average daily traded value", traded_value, _format_number(traded_value, "money0"), stats_as_of, statistics_url),
        ("price_vs_200d_pct", "Price versus 200-day average", timing, _format_number(timing, "pct"), stats_as_of, statistics_url),
    ]
    metrics = [
        GhanaMetric(key, label, value, display, as_of, "StockAnalysis / S&P Global", url)
        for key, label, value, display, as_of, url in metric_specs
    ]
    return company, price, price_date, inputs, metrics


def _official_search(
    client: _HttpClient,
    root: str,
    query: str,
    source_name: str,
) -> tuple[list[dict[str, Any]], bool]:
    url = f"{root}/?s={quote_plus(query)}"
    try:
        html = _fetch_html(client, url)
    except Exception:
        return [], False
    parser = _DocumentParser()
    parser.feed(html)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    query_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", query)
        if len(token) >= 3
    }
    for href, title in parser.links:
        lower = title.lower()
        if len(title) < 12 or href in seen:
            continue
        if query_tokens and not any(token in lower for token in query_tokens):
            continue
        if not href.startswith("http"):
            continue
        results.append(
            {
                "title": title,
                "published": None,
                "publisher": source_name,
                "link": href,
            }
        )
        seen.add(href)
        if len(results) >= 8:
            return results, True
    return results, True


GOVERNANCE_TERMS = (
    "audit qualification",
    "fraud",
    "investigation",
    "restatement",
    "director resignation",
    "late filing",
    "suspension",
    "governance",
)
REGULATORY_TERMS = (
    "sanction",
    "penalty",
    "breach",
    "non-compliance",
    "violation",
    "suspension",
    "caution",
    "investigation",
)


def headline_risk(headlines: list[dict[str, Any]], terms: tuple[str, ...]) -> bool:
    text = " ".join(str(item.get("title", "")).lower() for item in headlines)
    return any(term in text for term in terms)


def fetch_ghana_bundle(ticker: str) -> GhanaResearchBundle:
    symbol = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,15}", symbol):
        raise ValueError("Enter a valid GSE ticker, for example MTNGH, GCB, or EGH.")
    statistics_url = f"{STOCK_ANALYSIS_ROOT}/{symbol}/statistics/"
    income_url = f"{STOCK_ANALYSIS_ROOT}/{symbol}/financials/income-statement/"
    profile_url = f"{STOCK_ANALYSIS_ROOT}/{symbol}/company/"
    client = _HttpClient()
    statistics_html = _fetch_html(client, statistics_url)
    income_html = _fetch_html(client, income_url)
    company, price, price_date, inputs, metrics = parse_stockanalysis_pages(
        symbol, statistics_html, income_html
    )
    industry = None
    business_summary = None
    try:
        profile_html = _fetch_html(client, profile_url)
        industry, business_summary, profile_as_of = _parse_profile(profile_html)
        metrics.append(
            GhanaMetric(
                "business_profile",
                "Business profile",
                industry,
                industry or "Available—open source",
                profile_as_of,
                "StockAnalysis / S&P Global",
                profile_url,
            )
        )
    except Exception:
        profile_as_of = None

    search_term = ISSUER_SEARCH_TERMS.get(symbol, company or symbol)
    gse_headlines, gse_complete = _official_search(
        client, GSE_SEARCH_ROOT, search_term, "Ghana Stock Exchange"
    )
    sec_headlines, sec_complete = _official_search(
        client, SEC_SEARCH_ROOT, search_term, "SEC Ghana"
    )
    inputs["governance_flag"] = (
        headline_risk(gse_headlines, GOVERNANCE_TERMS) if gse_complete else None
    )
    inputs["regulatory_flag"] = (
        headline_risk(sec_headlines, REGULATORY_TERMS) if sec_complete else None
    )

    gse_search_url = f"{GSE_SEARCH_ROOT}/?s={quote_plus(search_term)}"
    sec_search_url = f"{SEC_SEARCH_ROOT}/?s={quote_plus(search_term)}"
    metrics.extend(
        [
            GhanaMetric(
                "governance_flag",
                "Potential governance concern in reviewed GSE results",
                inputs["governance_flag"],
                "Unknown" if inputs["governance_flag"] is None else ("Review required" if inputs["governance_flag"] else "None detected"),
                datetime.now(timezone.utc).date().isoformat() if gse_complete else None,
                "Ghana Stock Exchange search",
                gse_search_url,
            ),
            GhanaMetric(
                "regulatory_flag",
                "Potential regulatory concern in reviewed SEC results",
                inputs["regulatory_flag"],
                "Unknown" if inputs["regulatory_flag"] is None else ("Review required" if inputs["regulatory_flag"] else "None detected"),
                datetime.now(timezone.utc).date().isoformat() if sec_complete else None,
                "SEC Ghana search",
                sec_search_url,
            ),
        ]
    )

    warnings = [
        "StockAnalysis supplies delayed research data; the final executable price is the IC Wealth quote.",
        "The governance and regulatory checks screen matching official search results; open the linked evidence before accumulating.",
    ]
    if not gse_complete:
        warnings.append("The GSE issuer search was unavailable, so governance evidence remains Unknown.")
    if not sec_complete:
        warnings.append("The SEC Ghana search was unavailable, so regulatory evidence remains Unknown.")
    if price is None:
        warnings.append("A delayed closing price could not be extracted from the data page.")
    if business_summary is None:
        warnings.append("The business-profile page was unavailable; the quantitative gates still ran.")

    return GhanaResearchBundle(
        ticker=symbol,
        company_name=company,
        industry=industry,
        business_summary=business_summary,
        price=price,
        price_as_of=price_date,
        inputs=inputs,
        metrics=metrics,
        headlines=(gse_headlines + sec_headlines)[:12],
        warnings=warnings,
    )
