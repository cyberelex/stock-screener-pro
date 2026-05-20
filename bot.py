"""Forward trading bot — auto-picks top N from live screener scores and
executes paper trades on a configurable schedule."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from screener import compute_score, detect_regime, SCORE_WEIGHTS
from database import (
    get_or_create_portfolio, get_portfolio, execute_trade,
    get_holdings, save_snapshot, enrich_holdings_with_prices,
)


# Maps market regimes to which presets get a bonus when auto-selecting.
# The bot scores every preset, then adds regime bonus points to tilt
# toward strategies that historically work better in that environment.
_REGIME_AFFINITY = {
    "normal": {
        "Momentum / Growth": 10,
        "Value Hunting": 2,
        "Dividend Income": 3,
    },
    "stressed": {
        "Value Hunting": 8,
        "Dividend Income": 6,
        "Oversold Bounce": 4,
        "Momentum / Growth": -5,
    },
    "selloff": {
        "Oversold Bounce": 12,
        "Value Hunting": 6,
        "Dividend Income": 4,
        "Momentum / Growth": -10,
    },
}


def auto_select_strategy(screener_df: pd.DataFrame) -> dict:
    """Evaluate every preset against the current universe and regime,
    return the best one with reasoning.

    Returns:
        {
            "preset": chosen preset name,
            "regime": detected regime label,
            "scores": {preset_name: composite_score, ...},
            "reasoning": human-readable explanation,
        }
    """
    if screener_df.empty:
        return {"preset": "No Preset", "regime": "Unknown",
                "scores": {}, "reasoning": "No data to evaluate."}

    regime_info = detect_regime(screener_df)
    regime = regime_info["regime"]
    affinity = _REGIME_AFFINITY.get(regime, {})

    presets_to_test = [p for p in SCORE_WEIGHTS if p != "No Preset"]
    preset_scores = {}

    for preset_name in presets_to_test:
        scores = compute_score(screener_df, preset_name)
        top_10_avg = scores.nlargest(10).mean() if len(scores) >= 10 else scores.mean()
        spread = scores.nlargest(10).std() if len(scores) >= 10 else scores.std()

        # Higher average among the top picks = better signal clarity.
        # Lower spread = more consistent picks (the bot likes consensus).
        signal = top_10_avg - (spread * 0.3)

        regime_bonus = affinity.get(preset_name, 0)
        preset_scores[preset_name] = round(signal + regime_bonus, 1)

    best = max(preset_scores, key=preset_scores.get)

    parts = [f"Market regime: **{regime_info['label']}**"]
    ranked = sorted(preset_scores.items(), key=lambda x: -x[1])
    parts.append("Strategy scores: " + ", ".join(
        f"{name} ({sc:.0f})" for name, sc in ranked
    ))
    parts.append(f"Selected **{best}** — strongest signal for current conditions.")

    return {
        "preset": best,
        "regime": regime_info["label"],
        "scores": preset_scores,
        "reasoning": " | ".join(parts),
    }


def bot_rebalance(
    screener_df: pd.DataFrame,
    preset: str,
    top_n: int = 10,
    max_position_pct: float = 0.10,
    portfolio_name: str = "Robo Bot",
) -> dict:
    """Run one rebalance cycle: score universe, sell exits, buy new picks.

    Args:
        screener_df: Current screener results with all columns.
        preset: Strategy preset name for scoring.
        top_n: How many stocks to hold.
        max_position_pct: Max % of portfolio per stock.
        portfolio_name: Paper portfolio name.

    Returns:
        Dict with keys: picks, buys, sells, portfolio_value, cash.
    """
    pid = get_or_create_portfolio(portfolio_name, ptype="bot", strategy=preset)
    portfolio = get_portfolio(pid)
    if not portfolio:
        return {"error": "Portfolio not found"}

    if screener_df.empty:
        return {"error": "No screener data to work with"}

    scored = screener_df.copy()
    scored["Score"] = compute_score(scored, preset)
    scored = scored.sort_values("Score", ascending=False).head(top_n)
    picks = scored["Ticker"].tolist()

    holdings_df = get_holdings(pid)
    held_tickers = set(holdings_df["Ticker"].tolist()) if not holdings_df.empty else set()

    sells = []
    for ticker in held_tickers:
        if ticker not in picks:
            shares = holdings_df.loc[holdings_df["Ticker"] == ticker, "Shares"].iloc[0]
            try:
                price = _get_current_price(ticker)
            except Exception:
                continue
            err = execute_trade(pid, ticker, "sell", shares, price,
                                reason="Dropped from top N", strategy=preset)
            if not err:
                sells.append({"ticker": ticker, "shares": shares, "price": price})

    portfolio = get_portfolio(pid)
    cash = portfolio["cash"]

    new_tickers = [t for t in picks if t not in held_tickers or t in [s["ticker"] for s in sells]]
    buys = []
    if new_tickers:
        alloc_each = cash / len(new_tickers)
        max_alloc = portfolio["starting_cash"] * max_position_pct

        for ticker in new_tickers:
            try:
                price = _get_current_price(ticker)
            except Exception:
                continue
            if price <= 0:
                continue
            budget = min(alloc_each, max_alloc)
            shares = budget / price
            err = execute_trade(pid, ticker, "buy", round(shares, 4), price,
                                reason=f"Top {top_n} by {preset}", strategy=preset)
            if not err:
                buys.append({"ticker": ticker, "shares": round(shares, 4), "price": price})

    total_value = _compute_portfolio_value(pid)
    save_snapshot(pid, total_value, date.today().isoformat())

    return {
        "picks": picks,
        "buys": buys,
        "sells": sells,
        "portfolio_value": total_value,
        "cash": get_portfolio(pid)["cash"],
    }


def get_bot_status(portfolio_name: str = "Robo Bot") -> dict:
    """Return current bot portfolio status."""
    pid = get_or_create_portfolio(portfolio_name, ptype="bot")
    portfolio = get_portfolio(pid)
    holdings_df = get_holdings(pid)

    if not holdings_df.empty:
        holdings_df = enrich_holdings_with_prices(holdings_df)

    total_value = _compute_portfolio_value(pid)
    starting = portfolio["starting_cash"]
    pnl = total_value - starting
    pnl_pct = (pnl / starting) * 100 if starting > 0 else 0

    return {
        "pid": pid,
        "portfolio": portfolio,
        "holdings": holdings_df,
        "total_value": total_value,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
    }


def _get_current_price(ticker: str) -> float:
    tk = yf.Ticker(ticker)
    price = tk.info.get("currentPrice")
    if price:
        return price
    hist = tk.history(period="1d")
    if not hist.empty:
        return hist["Close"].iloc[-1]
    raise ValueError(f"No price for {ticker}")


def _compute_portfolio_value(pid: int) -> float:
    portfolio = get_portfolio(pid)
    cash = portfolio["cash"]
    holdings_df = get_holdings(pid)
    if holdings_df.empty:
        return cash

    market_value = 0
    for _, row in holdings_df.iterrows():
        try:
            price = _get_current_price(row["Ticker"])
            market_value += row["Shares"] * price
        except Exception:
            continue

    return cash + market_value
