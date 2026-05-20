"""Backtesting engine — replays a strategy preset against historical data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from screener import _rsi, SCORE_WEIGHTS


def _compute_historical_scores(
    prices_df: pd.DataFrame,
    preset: str,
    date: pd.Timestamp,
    lookback: int = 252,
) -> pd.Series:
    """Compute composite scores at a specific historical date using price data only.

    Since we can't get historical fundamentals from yfinance, we use
    price-derived technicals (RSI, MA signals, drawdown, volume) for
    backtesting. This is an approximation — forward bot uses live fundamentals.
    """
    scores = {}
    for ticker in prices_df.columns.get_level_values(0).unique():
        try:
            close = prices_df[ticker]["Close"].loc[:date].dropna()
            volume = prices_df[ticker]["Volume"].loc[:date].dropna()
            if len(close) < 50:
                continue

            price = close.iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
            rsi_val = _rsi(close)
            high_52w = close.iloc[-min(252, len(close)):].max()
            pct_from_high = ((price - high_52w) / high_52w) * 100
            avg_vol = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else np.nan
            vol_ratio = (volume.iloc[-1] / avg_vol) if not np.isnan(avg_vol) and avg_vol > 0 else 1.0

            metrics = {
                "RSI (14)": rsi_val,
                "% from 52w High": pct_from_high,
                "Vol vs Avg": vol_ratio,
                "P/E": np.nan,
                "Fwd P/E": np.nan,
                "Div Yield %": np.nan,
                "P/B": np.nan,
                "Revenue Growth %": np.nan,
                "Profit Margin %": np.nan,
            }

            weights = SCORE_WEIGHTS.get(preset, SCORE_WEIGHTS["No Preset"])
            score = 0.0
            usable_weight = 0.0
            for col, weight, lower_is_better in weights:
                val = metrics.get(col)
                if val is None or np.isnan(val):
                    continue
                usable_weight += weight
                score += weight

            scores[ticker] = {**metrics, "_price": price, "_ma50": ma50, "_ma200": ma200}
        except Exception:
            continue

    if not scores:
        return pd.Series(dtype=float)

    df = pd.DataFrame(scores).T
    weights = SCORE_WEIGHTS.get(preset, SCORE_WEIGHTS["No Preset"])
    total_weight = sum(w for _, w, _ in weights)
    composite = pd.Series(0.0, index=df.index)

    for col, weight, lower_is_better in weights:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        pct = s.rank(pct=True).fillna(0.5)
        if lower_is_better:
            pct = 1 - pct
        composite += pct * (weight / total_weight)

    return composite.sort_values(ascending=False)


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
        }
    """
    if progress_callback:
        progress_callback(0.05, "Downloading historical data...")

    period = f"{lookback_years}y"
    raw = yf.download(tickers, period=period, group_by="ticker", progress=False, threads=True)

    if raw.empty:
        return {"equity_curve": pd.DataFrame(), "trades": [], "stats": {}}

    if len(tickers) == 1:
        raw = pd.concat({tickers[0]: raw}, axis=1)

    spy = yf.download("SPY", period=period, progress=False)
    if spy.empty:
        return {"equity_curve": pd.DataFrame(), "trades": [], "stats": {}}

    dates = raw.index
    if len(dates) < 60:
        return {"equity_curve": pd.DataFrame(), "trades": [], "stats": {}}

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

    spy_start = spy["Close"].iloc[0]

    for i, dt in enumerate(rebal_dates):
        if progress_callback:
            progress_callback(0.3 + 0.6 * (i / len(rebal_dates)),
                              f"Simulating {dt.strftime('%Y-%m-%d')}...")

        scores = _compute_historical_scores(raw, preset, dt)
        if scores.empty:
            continue

        top_picks = scores.head(top_n).index.tolist()

        for ticker, shares in list(holdings.items()):
            if ticker not in top_picks:
                try:
                    price = raw[ticker]["Close"].loc[:dt].dropna().iloc[-1]
                    cash += shares * price
                    trades.append({
                        "Date": dt.strftime("%Y-%m-%d"), "Ticker": ticker,
                        "Side": "sell", "Shares": round(shares, 2),
                        "Price": round(price, 2), "Reason": "Dropped from top N",
                    })
                    del holdings[ticker]
                except Exception:
                    continue

        current_tickers = [t for t in top_picks if t not in holdings]
        if current_tickers:
            alloc_per_stock = cash / max(len(current_tickers), 1)
            for ticker in current_tickers:
                try:
                    price = raw[ticker]["Close"].loc[:dt].dropna().iloc[-1]
                    if price <= 0:
                        continue
                    shares = alloc_per_stock / price
                    max_position = starting_cash * 0.10
                    shares = min(shares, max_position / price)
                    cost = shares * price
                    if cost > cash:
                        continue
                    cash -= cost
                    holdings[ticker] = holdings.get(ticker, 0) + shares
                    trades.append({
                        "Date": dt.strftime("%Y-%m-%d"), "Ticker": ticker,
                        "Side": "buy", "Shares": round(shares, 2),
                        "Price": round(price, 2), "Reason": f"Top {top_n} by {preset}",
                    })
                except Exception:
                    continue

        portfolio_value = cash
        for ticker, shares in holdings.items():
            try:
                price = raw[ticker]["Close"].loc[:dt].dropna().iloc[-1]
                portfolio_value += shares * price
            except Exception:
                continue

        spy_price = spy["Close"].loc[:dt].dropna().iloc[-1]
        benchmark_value = starting_cash * (spy_price / spy_start)

        equity_records.append({
            "Date": dt, "Portfolio": round(portfolio_value, 2),
            "Benchmark": round(benchmark_value, 2),
        })

    if progress_callback:
        progress_callback(0.95, "Computing stats...")

    equity_df = pd.DataFrame(equity_records)
    if equity_df.empty:
        return {"equity_curve": equity_df, "trades": trades, "stats": {}}

    port_returns = equity_df["Portfolio"].pct_change().dropna()
    bench_returns = equity_df["Benchmark"].pct_change().dropna()

    total_return = (equity_df["Portfolio"].iloc[-1] / starting_cash - 1) * 100
    bench_total = (equity_df["Benchmark"].iloc[-1] / starting_cash - 1) * 100
    n_periods = len(equity_df)
    years = lookback_years

    ann_return = ((1 + total_return / 100) ** (1 / max(years, 0.5)) - 1) * 100
    ann_bench = ((1 + bench_total / 100) ** (1 / max(years, 0.5)) - 1) * 100

    peak = equity_df["Portfolio"].expanding().max()
    drawdown = ((equity_df["Portfolio"] - peak) / peak) * 100
    max_dd = drawdown.min()

    sharpe = (port_returns.mean() / port_returns.std() * np.sqrt(12)) if port_returns.std() > 0 else 0

    winning_trades = [t for t in trades if t["Side"] == "sell"]
    win_rate = 0
    if winning_trades:
        buy_prices = {}
        for t in trades:
            if t["Side"] == "buy":
                buy_prices[t["Ticker"]] = t["Price"]
        wins = sum(1 for t in winning_trades if t["Price"] > buy_prices.get(t["Ticker"], float("inf")))
        win_rate = (wins / len(winning_trades)) * 100

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

    return {"equity_curve": equity_df, "trades": trades, "stats": stats}
