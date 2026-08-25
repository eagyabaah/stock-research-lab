from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


def _validate(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Price history is missing columns: {sorted(missing)}")
    if len(frame) < 220:
        raise ValueError("At least 220 daily observations are required.")


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Return OHLCV data with deterministic technical indicators."""
    _validate(frame)
    data = frame.copy().sort_index()
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    close = data["Close"]
    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()
    data["SMA200"] = close.rolling(200).mean()

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI14"] = 100 - (100 / (1 + relative_strength))
    data.loc[avg_loss.eq(0) & avg_gain.gt(0), "RSI14"] = 100.0

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["ATR14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data["AVG_VOLUME20"] = data["Volume"].rolling(20).mean()
    data["VOLUME_RATIO"] = data["Volume"] / data["AVG_VOLUME20"].replace(0, np.nan)
    data["PRIOR_HIGH20"] = data["High"].shift(1).rolling(20).max()
    data["PRIOR_LOW20"] = data["Low"].shift(1).rolling(20).min()
    data["LOW10"] = data["Low"].rolling(10).min()
    data["RETURN63"] = close.pct_change(63)
    data["SMA50_SLOPE20"] = data["SMA50"] - data["SMA50"].shift(20)
    return data


def relative_strength_63(stock: pd.DataFrame, benchmark: pd.DataFrame | None) -> float | None:
    if benchmark is None or len(benchmark) < 64:
        return None
    stock_return = float(stock["Close"].iloc[-1] / stock["Close"].iloc[-64] - 1)
    benchmark_return = float(
        benchmark["Close"].iloc[-1] / benchmark["Close"].iloc[-64] - 1
    )
    return stock_return - benchmark_return
