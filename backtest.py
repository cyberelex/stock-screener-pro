"""Backtesting engine — replays price-based signals against historical data."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

from screener import _rsi, _return_pct, SCORE_WEIGHTS, compute_ai_setup_score


BACKTEST_NOTE = (
    "Price history only: RSI, trend, drawdown, volume, and (for AI Momentum) "
    "the setup checklist. Live P/E, yield, and growth are not available "
    "historically, so Value Hunting and Dividend Income here are technical "
    "approximations, not a replay of the live screener."
)


def _close_series(frame: pd.DataFrame | pd.Series) -> pd.Series:
    """Return a 1-d close series from yfinance's shifting column layouts."""
    if isinstance(frame, pd.Series):
        return frame
    if "Close" in frame.columns:
        close = frame["Close"]
    else:
        try:
            close = frame.xs("Close", axis=1, level=0)
        except Exception:
            close = frame.iloc[:, 0]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close


def _field_series(ticker_frame: pd.DataFrame, field: str) -> pd.Series:
    val = ticker_frame[field]
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]
    return val


def _price_at(prices_df: pd.DataFrame, ticker: str, date: pd.Timestamp) -> float:
    close = _close_series(prices_df[ticker]).loc[:date].dropna()
    return float(close.iloc[-1])


def _metrics_at(prices_df: pd.DataFrame, ticker: str, date: pd.Timestamp) -> dict | None:
    """Price-derived metrics at *date*. Returns None if there is not enough history."""
    try:
        close = _close_series(prices_df[ticker]).loc[:date].dropna()
        volume = _field_series(prices_df[ticker], "Volume").loc[:date].dropna()
    except Exception:
        return None
    if len(close) < 50:
        return None

    price = float(close.iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan
    rsi_val = _rsi(close)
    high_52w = float(close.iloc[-min(252, len(close)):].max())
    pct_from_high = ((price - high_52w) / high_52w) * 100 if high_52w else np.nan
    avg_vol = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else np.nan
    vol_ratio = (float(volume.iloc[-1]) / avg_vol) if not np.isnan(avg_vol) and avg_vol > 0 else np.nan
    week = _return_pct(close, 5)
    month = _return_pct(close, 21)

    return {
        "Ticker": ticker,
        "Price": price,
        "RSI (14)": rsi_val,
        "% from 52w High": pct_from_high,
        "Vol vs Avg": vol_ratio,
        "1W %": week,
        "1M %": month,
        "Above 50-MA": price > ma50 if not np.isnan(ma50) else False,
        "Above 200-MA": price > ma200 if not np.isnan(ma200) else False,
        "50-day MA": ma50 if not np.isnan(ma50) else None,
        "200-day MA": ma200 if not np.isnan(ma200) else None,
        "_price": price,
    }


def _nvda_1m(prices_df: pd.DataFrame, date: pd.Timestamp) -> float | None:
    tickers = set(prices_df.columns.get_level_values(0).unique())
    if "NVDA" not in tickers:
        return None
    try:
        close = _close_series(prices_df["NVDA"]).loc[:date].dropna()
        return _return_pct(close, 21)
    except Exception:
        return None


def _compute_historical_scores(
    prices_df: pd.DataFrame,
    preset: str,
    date: pd.Timestamp,
) -> pd.Series:
    """Score the universe at *date* from price history only."""
    records = []
    nvda_1m = _nvda_1m(prices_df, date)
    tickers = prices_df.columns.get_level_values(0).unique()

    for ticker in tickers:
        metrics = _metrics_at(prices_df, ticker, date)
        if not metrics:
            continue
        month = metrics.get("1M %")
        if nvda_1m is not None and month is not None:
            metrics["RS vs NVDA"] = round(float(month) - nvda_1m, 1)
        else:
            metrics["RS vs NVDA"] = np.nan
        records.append(metrics)

    if not records:
        return pd.Series(dtype=float)

    df = pd.DataFrame(records).set_index("Ticker")

    if preset == "AI Momentum":
        setup = compute_ai_setup_score(df)
        return setup["Setup Score"].sort_values(ascending=False)

    weights = SCORE_WEIGHTS.get(preset, SCORE_WEIGHTS["No Preset"])
    usable = [(col, weight, lower) for col, weight, lower in weights if col in df.columns]
    if not usable:
        return pd.Series(dtype=float)

    composite = pd.Series(0.0, index=df.index)
    used = 0.0
    for col, weight, lower_is_better in usable:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() < 2:
            continue
        pct = s.rank(pct=True).fillna(0.5)
        if lower_is_better:
            pct = 1 - pct
        composite += pct * weight
        used += weight

    if used <= 0:
        return pd.Series(dtype=float)
    return (composite / used * 100).sort_values(ascending=False)


def _fifo_win_rate(trades: list[dict]) -> float:
    lots: dict[str, list[list[float]]] = defaultdict(list)
    sells = 0
    wins = 0
    for trade in trades:
        ticker = trade["Ticker"]
        shares = float(trade["Shares"])
        price = float(trade["Price"])
        if trade["Side"] == "buy":
            lots[ticker].append([shares, price])
            continue
        remaining = shares
        cost = 0.0
        proceeds = 0.0
        while remaining > 1e-8 and lots[ticker]:
            lot_shares, lot_price = lots[ticker][0]
            take = min(remaining, lot_shares)
            cost += take * lot_price
            proceeds += take * price
            lot_shares -= take
            remaining -= take
            if lot_shares <= 1e-8:
                lots[ticker].pop(0)
            else:
                lots[ticker][0][0] = lot_shares
        if cost <= 0:
            continue
        sells += 1
        if proceeds > cost:
            wins += 1
    if sells == 0:
        return 0.0
    return wins / sells * 100


def run_backtest(
    tickers: list[str],
    preset: str,
    top_n: int = 10,
    rebalance_freq: str = "monthly",
    lookback_years: int = 2,
    starting_cash: float = 100_000,
    progress_callback=None,
) -> dict:
    """Run a full backtest and return results dict.

    Returns:
        {
            "equity_curve": pd.DataFrame with Date, Portfolio, Benchmark columns,
            "trades": list of trade dicts,
            "stats": dict of performance metrics,
            "note": limitation string,
        }
    """
    if progress_callback:
        progress_callback(0.05, "Downloading historical data...")

    period = f"{lookback_years}y"
    raw = yf.download(tickers, period=period, group_by="ticker", progress=False, threads=True)

    if raw.empty:
        return {"equity_curve": pd.DataFrame(), "trades": [], "stats": {}, "note": BACKTEST_NOTE}

    if len(tickers) == 1:
        raw = pd.concat({tickers[0]: raw}, axis=1)

    spy = yf.download("SPY", period=period, progress=False)
    if spy.empty:
        return {"equity_curve": pd.DataFrame(), "trades": [], "stats": {}, "note": BACKTEST_NOTE}

    dates = raw.index
    if len(dates) < 60:
        return {"equity_curve": pd.DataFrame(), "trades": [], "stats": {}, "note": BACKTEST_NOTE}

    if progress_callback:
        progress_callback(0.3, "Running backtest simulation...")

    if rebalance_freq == "weekly":
        rebal_dates = dates[dates.weekday == 0]
    else:
        month_groups = dates.to_series().groupby(dates.to_period("M"))
        rebal_dates = month_groups.apply(lambda g: g.iloc[0]).values
        rebal_dates = pd.DatetimeIndex(rebal_dates)

    rebal_dates = rebal_dates[rebal_dates >= dates[min(252, len(dates) - 1)]]

    cash = starting_cash
    holdings: dict[str, float] = {}
    trades: list[dict] = []
    equity_records: list[dict] = []
    spy_start = None

    for i, dt in enumerate(rebal_dates):
        if progress_callback:
            progress_callback(
                0.3 + 0.6 * (i / max(len(rebal_dates), 1)),
                f"Simulating {pd.Timestamp(dt).strftime('%Y-%m-%d')}...",
            )

        scores = _compute_historical_scores(raw, preset, dt)
        if scores.empty:
            continue

        top_picks = scores.head(top_n).index.tolist()
        prices: dict[str, float] = {}
        names = set(holdings) | set(top_picks)
        for ticker in names:
            try:
                prices[ticker] = _price_at(raw, ticker, dt)
            except Exception:
                continue

        portfolio_value = cash + sum(
            shares * prices[ticker]
            for ticker, shares in holdings.items()
            if ticker in prices
        )

        for ticker, shares in list(holdings.items()):
            if ticker in top_picks or ticker not in prices:
                continue
            price = prices[ticker]
            cash += shares * price
            trades.append({
                "Date": pd.Timestamp(dt).strftime("%Y-%m-%d"), "Ticker": ticker,
                "Side": "sell", "Shares": round(shares, 2),
                "Price": round(price, 2), "Reason": "Dropped from top N",
            })
            del holdings[ticker]

        portfolio_value = cash + sum(
            shares * prices[ticker]
            for ticker, shares in holdings.items()
            if ticker in prices
        )
        if not top_picks or portfolio_value <= 0:
            continue

        target_dollars = portfolio_value / len(top_picks)
        for ticker in top_picks:
            price = prices.get(ticker)
            if not price or price <= 0:
                continue
            current_shares = holdings.get(ticker, 0.0)
            current_val = current_shares * price
            diff = target_dollars - current_val
            if abs(diff) < max(price * 0.01, 1.0):
                continue
            if diff > 0:
                spend = min(diff, cash)
                shares = spend / price
                if shares <= 0:
                    continue
                cash -= shares * price
                holdings[ticker] = current_shares + shares
                trades.append({
                    "Date": pd.Timestamp(dt).strftime("%Y-%m-%d"), "Ticker": ticker,
                    "Side": "buy", "Shares": round(shares, 2),
                    "Price": round(price, 2), "Reason": f"Rebalance to equal weight ({preset})",
                })
            else:
                shares = min(current_shares, abs(diff) / price)
                if shares <= 0:
                    continue
                cash += shares * price
                leftover = current_shares - shares
                if leftover <= 1e-8:
                    holdings.pop(ticker, None)
                else:
                    holdings[ticker] = leftover
                trades.append({
                    "Date": pd.Timestamp(dt).strftime("%Y-%m-%d"), "Ticker": ticker,
                    "Side": "sell", "Shares": round(shares, 2),
                    "Price": round(price, 2), "Reason": "Trim to equal weight",
                })

        portfolio_value = cash
        for ticker, shares in holdings.items():
            try:
                portfolio_value += shares * _price_at(raw, ticker, dt)
            except Exception:
                continue

        try:
            spy_price = float(_close_series(spy).loc[:dt].dropna().iloc[-1])
        except Exception:
            continue
        if spy_start is None or spy_start == 0:
            spy_start = spy_price
        benchmark_value = starting_cash * (spy_price / spy_start)

        equity_records.append({
            "Date": dt, "Portfolio": round(portfolio_value, 2),
            "Benchmark": round(benchmark_value, 2),
        })

    if progress_callback:
        progress_callback(0.95, "Computing stats...")

    equity_df = pd.DataFrame(equity_records)
    if equity_df.empty:
        return {"equity_curve": equity_df, "trades": trades, "stats": {}, "note": BACKTEST_NOTE}

    port_returns = equity_df["Portfolio"].pct_change().dropna()
    total_return = (equity_df["Portfolio"].iloc[-1] / starting_cash - 1) * 100
    bench_total = (equity_df["Benchmark"].iloc[-1] / starting_cash - 1) * 100

    first_dt = pd.to_datetime(equity_df["Date"].iloc[0])
    last_dt = pd.to_datetime(equity_df["Date"].iloc[-1])
    years = max((last_dt - first_dt).days / 365.25, 0.25)

    ann_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
    ann_bench = ((1 + bench_total / 100) ** (1 / years) - 1) * 100

    peak = equity_df["Portfolio"].expanding().max()
    drawdown = ((equity_df["Portfolio"] - peak) / peak) * 100
    max_dd = drawdown.min()

    periods_per_year = 52 if rebalance_freq == "weekly" else 12
    sharpe = (
        port_returns.mean() / port_returns.std() * np.sqrt(periods_per_year)
        if port_returns.std() > 0 else 0
    )

    win_rate = _fifo_win_rate(trades)

    stats = {
        "Total Return": f"{total_return:.1f}%",
        "Benchmark Return": f"{bench_total:.1f}%",
        "Annualized Return": f"{ann_return:.1f}%",
        "Annualized Benchmark": f"{ann_bench:.1f}%",
        "Max Drawdown": f"{max_dd:.1f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Win Rate": f"{win_rate:.0f}%",
        "Total Trades": len(trades),
        "Final Value": f"${equity_df['Portfolio'].iloc[-1]:,.0f}",
    }

    if progress_callback:
        progress_callback(1.0, "Done")

    return {
        "equity_curve": equity_df,
        "trades": trades,
        "stats": stats,
        "note": BACKTEST_NOTE,
    }
