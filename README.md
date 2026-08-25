# Stock Research Lab

A runnable Streamlit website for transparent US swing-trade qualification and
Ghana long-term stock scoring. It implements the mandate we defined:

- $1,000 initial portfolio and $500 monthly contribution context;
- 10% drawdown halt;
- $5 initial US risk per trade and $20 maximum concurrent open risk;
- $200 maximum US position;
- 2:1 minimum reward/risk;
- no leverage, options, shorts, penny stocks, or averaging down;
- no Ghana swing trading.

The site shows the evidence behind every gate. It does **not** expose or depend
on hidden model reasoning, and it does not place orders.

## Features

- Batch screen of up to eight US tickers.
- SPY market-regime gate.
- Trend-pullback, confirmed-breakout, and recovery/reclaim strategies.
- SMA20/50/200, RSI14, ATR14, volume ratio, 20-day breakout level, and 63-day
  relative strength.
- Fundamentals, valuation, news/event, liquidity, and risk gates.
- Entry, stop, 2R/3R targets, fractional shares, notional, and planned loss.
- Portfolio drawdown and existing-open-risk breakers.
- Candlestick chart with model levels.
- Evidence, bull thesis, bear case, and invalidation conditions.
- Manual Ghana long-term scoring that refuses to pass incomplete data.
- JSON export of each report.

## Run locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` if it does not open automatically.

## Run with Docker

```bash
docker build -t stock-research-lab .
docker run --rm -p 8501:8501 stock-research-lab
```

Then open `http://localhost:8501`.

## Deploy as a public website

The simplest path is Streamlit Community Cloud:

1. Put this folder in a GitHub repository.
2. Sign in at `share.streamlit.io`.
3. Choose **Create app**, select the repository, and set `app.py` as the entrypoint.
4. Deploy and use the assigned `streamlit.app` URL.

The included `Dockerfile` also works with hosts that accept containers.

## Tests

The strategy-engine tests use only Python, pandas, and NumPy; they do not call a
live data source:

```bash
python -m unittest discover -s tests -v
```

## Data and safety

The default adapter uses the community-maintained `yfinance` package. Its own
documentation describes it as an open-source research/personal-use interface,
not an official Yahoo product. Therefore:

- treat prices as research data, not executable quotes;
- confirm entries and position sizes in Robinhood;
- verify earnings and financial data in company filings;
- review every flagged article instead of trusting keyword sentiment;
- enter Ghana metrics from official GSE filings and the actual IC Wealth quote.

See `MODEL_RULES.md` for the exact scoring rules and limitations.
