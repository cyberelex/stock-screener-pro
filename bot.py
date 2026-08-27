"""Forward trading bot — auto-picks top N from live screener scores and
executes paper trades on a configurable schedule."""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf

from screener import compute_score, detect_regime, SCORE_WEIGHTS, apply_preset_filters
from database import (
    get_or_create_portfolio, get_portfolio, execute_trade,
    get_holdings, snapshot_portfolio, enrich_holdings_with_prices,
    portfolio_total_value,
)


# Maps market regimes to which presets get a bonus when auto-selecting.
# The bot scores every preset, then adds regime bonus points to tilt
# toward strategies that historically work better in that environment.
_REGIME_AFFINITY = {
    "normal": {
        "Momentum / Growth": 10,
        "AI Momentum": 6,
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
    """Pick a strategy from regime fit plus how many names currently qualify.

    Does not compare raw Score values across presets. AI Momentum is an
    absolute 0–100 checklist; the other presets are percentile ranks inside
    their own list. Mixing those numbers would always tilt toward AI.
    """
    if screener_df.empty:
        return {"preset": "No Preset", "regime": "Unknown",
                "scores": {}, "reasoning": "No data to evaluate."}

    regime_info = detect_regime(screener_df)
    regime = regime_info["regime"]
    affinity = _REGIME_AFFINITY.get(regime, {})

    presets_to_test = [p for p in SCORE_WEIGHTS if p != "No Preset"]
    preset_scores = {}
    passing = {}

    for preset_name in presets_to_test:
        filtered = apply_preset_filters(screener_df, preset_name, regime)
        n_pass = len(filtered)
        passing[preset_name] = n_pass

        if n_pass < 3:
            opportunity = -15.0
        elif preset_name == "AI Momentum":
            setup_avg = float(compute_score(filtered, preset_name).mean())
            opportunity = setup_avg * 0.1
        else:
            opportunity = min(10.0, n_pass / 2)

        preset_scores[preset_name] = round(affinity.get(preset_name, 0) + opportunity, 1)

    best = max(preset_scores, key=preset_scores.get)

    parts = [
        f"Universe regime: **{regime_info['label']}** "
        "(this loaded list, not the S&P)"
    ]
    ranked = sorted(preset_scores.items(), key=lambda x: -x[1])
    parts.append("Fit scores (regime bonus + opportunity, not raw stock scores): " + ", ".join(
        f"{name} ({sc:.0f}, {passing[name]} names)" for name, sc in ranked
    ))
    parts.append(f"Selected **{best}** — best regime fit among strategies with enough names.")

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

    regime = detect_regime(screener_df)["regime"]
    candidates = apply_preset_filters(screener_df, preset, regime)
    if candidates.empty:
        candidates = screener_df.copy()

    scored = candidates.copy()
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
    current_value = portfolio_total_value(pid)

    new_tickers = [t for t in picks if t not in held_tickers or t in [s["ticker"] for s in sells]]
    buys = []
    if new_tickers:
        alloc_each = cash / len(new_tickers)
        max_alloc = current_value * max_position_pct

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

    total_value = snapshot_portfolio(pid, date.today().isoformat())

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

    total_value = portfolio_total_value(pid)
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
