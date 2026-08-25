# Model rules

This document is the inspectable specification behind the website. The code is
deterministic: the same inputs produce the same gate decisions and position size.

## Portfolio mandate

- Initial portfolio: $1,000.
- Monthly contribution: $500 (tracked outside this single-run engine).
- Portfolio halt: 10% drawdown.
- US risk per trade: the smallest of 0.5% of portfolio, the $5 dollar cap, and
  remaining capacity under the $20 total-open-risk limit.
- Total concurrent open risk: $20 maximum. The website sizes one proposed trade;
  the user must compare the planned risk with existing positions.
- Maximum US position: $200.
- Minimum reward/risk: 2:1.
- No leverage, options, short selling, penny stocks, or averaging down.
- Ghana analysis is long-term accumulation only.

## US scoring

| Gate | Points | Hard gate? |
|---|---:|---|
| Technical | 30 | Yes |
| Fundamentals | 25 | Yes |
| News/catalyst | 15 | Yes |
| Valuation | 10 | No |
| Liquidity/execution | 10 | Yes |
| Risk/setup | 10 | Yes |

`QUALIFIED` requires at least 75 points, every hard gate at `PASS`, and a market
regime and portfolio-risk state that are not `FAIL`. A valuation failure can therefore be outweighed only
by exceptionally strong evidence elsewhere; it remains visible as a risk.

### Market regime

- `PASS`: SPY is above SMA200, SMA50 is above SMA200, and SMA50 is rising over
  the last 20 sessions.
- `CAUTION`: SPY is above SMA200 but intermediate confirmation is mixed.
- `FAIL`: SPY is below SMA200; new long positions are blocked.

### Portfolio risk

- New trades are blocked at a 10% portfolio drawdown.
- New trades are also blocked when existing positions already use the $20
  total-open-risk allowance.
- At a drawdown of 6% or more, the gate changes to `CAUTION`.

### Technical strategies

The engine separately scores three strategies and displays the highest-scoring
one. A stock does not pass merely because one attractive indicator exists.

1. **Trend pullback:** established trend, rising SMA50, price near SMA20, RSI in
   a controlled range, improving close, and positive 63-day relative strength.
2. **Confirmed breakout:** close above the prior 20-day high, volume of at least
   1.3 times its 20-day average, aligned moving averages, confirming RSI, limited
   ATR extension, and positive relative strength.
3. **Recovery/reclaim:** price above SMA200 and back above SMA20/SMA50 after a
   genuine correction, with momentum, volume, and relative-strength confirmation.

The stop is placed algorithmically below the nearest strategy-specific support
with an ATR buffer. Target 1 is two risk units above entry and Target 2 is three.

### Fundamental gate

The current no-key data adapter uses reported revenue growth, earnings growth,
profit margin, free cash flow, and debt relative to cash. At least three fields
must be present. Missing data can only produce `CAUTION`, never `PASS`.

This is a first-pass quality guardrail, not a full discounted-cash-flow model.
Primary filings should replace or verify every provider field before money is
committed.

### News gate

- Earnings or a comparable binary event within two trading sessions blocks a
  new entry.
- A small explicit keyword set identifies headlines requiring review.
- Severe accounting, fraud, bankruptcy, or investigation language fails the gate.
- No recent headlines means `CAUTION`, not an automatic pass.

This rule is deliberately conservative and cannot replace reading the filing or
article. It reports the titles that influenced the gate.

### Liquidity and risk

- Preferred 20-day average dollar volume: at least $50 million.
- Position shares are the smaller of:
  - risk budget divided by entry minus stop; and
  - maximum position dollars divided by entry.
- Robinhood-compatible fractional sizing is rounded down to 0.001 share.

## Ghana long-term scoring

| Component | Points |
|---|---:|
| Fundamentals and earnings quality | 35 |
| Valuation | 20 |
| Balance sheet and cash flow | 15 |
| News, governance, and regulation | 15 |
| Liquidity | 10 |
| Technical entry timing | 5 |

`ACCUMULATE GRADUALLY` requires at least 75 points, at least 80% data
completeness, a fundamentals score of at least 24/35, no unresolved governance
or regulatory flags, and minimum liquidity. With less than 70% completeness the
only possible result is `INSUFFICIENT DATA`.

## Known limitations

- yfinance is an unofficial, personal/research-use adapter and is not an
  exchange-grade feed.
- Headline keyword classification is a triage tool, not semantic due diligence.
- Adjusted daily data can differ from the prices and order types shown by a broker.
- Ghana inputs are manual until a licensed GSE/IC data integration is available.
- Backtesting, transaction costs, taxes, FX changes, portfolio correlation, and
  slippage are not yet modeled.
