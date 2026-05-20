"""SQLite persistence layer for paper trading portfolios, trades, and snapshots."""

from __future__ import annotations

import sqlite3
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import yfinance as yf

DB_PATH = Path(__file__).parent / "portfolio.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                type        TEXT    NOT NULL CHECK(type IN ('manual','bot')),
                strategy    TEXT,
                cash        REAL    NOT NULL DEFAULT 100000,
                starting_cash REAL  NOT NULL DEFAULT 100000,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL REFERENCES portfolios(id),
                ticker      TEXT    NOT NULL,
                side        TEXT    NOT NULL CHECK(side IN ('buy','sell')),
                shares      REAL    NOT NULL,
                price       REAL    NOT NULL,
                reason      TEXT,
                strategy    TEXT,
                ts          TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS daily_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL REFERENCES portfolios(id),
                dt          TEXT    NOT NULL,
                total_value REAL    NOT NULL,
                UNIQUE(portfolio_id, dt)
            );
            CREATE TABLE IF NOT EXISTS challenges (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT    NOT NULL,
                trade_start     TEXT    NOT NULL,
                trade_end       TEXT    NOT NULL,
                challenge_end   TEXT    NOT NULL,
                starting_cash   REAL    NOT NULL DEFAULT 100000,
                status          TEXT    NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active','locked','completed')),
                winner          TEXT,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
            );
        """)


# ── Portfolio helpers ─────────────────────────────────────────────────────

def get_or_create_portfolio(name: str, ptype: str = "manual",
                            strategy: str | None = None,
                            starting_cash: float = 100_000) -> int:
    with _conn() as c:
        row = c.execute(
            "SELECT id FROM portfolios WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return row["id"]
        cur = c.execute(
            "INSERT INTO portfolios (name, type, strategy, cash, starting_cash) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, ptype, strategy, starting_cash, starting_cash),
        )
        return cur.lastrowid


def get_portfolio(pid: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM portfolios WHERE id = ?", (pid,)).fetchone()
        return dict(row) if row else None


def list_portfolios() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM portfolios ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]


def reset_portfolio(pid: int) -> None:
    with _conn() as c:
        p = c.execute("SELECT starting_cash FROM portfolios WHERE id = ?", (pid,)).fetchone()
        if not p:
            return
        c.execute("DELETE FROM trades WHERE portfolio_id = ?", (pid,))
        c.execute("DELETE FROM daily_snapshots WHERE portfolio_id = ?", (pid,))
        c.execute("UPDATE portfolios SET cash = ? WHERE id = ?", (p["starting_cash"], pid))


# ── Challenge helpers ─────────────────────────────────────────────────────

def create_challenge(name: str, trade_days: int, hold_days: int,
                     starting_cash: float = 100_000) -> int:
    """Create a new challenge. Resets both portfolios to start fresh."""
    trade_start = date.today()
    from datetime import timedelta
    trade_end = trade_start + timedelta(days=trade_days)
    challenge_end = trade_end + timedelta(days=hold_days)

    manual_pid = get_or_create_portfolio("My Portfolio", ptype="manual",
                                         starting_cash=starting_cash)
    bot_pid = get_or_create_portfolio("Robo Bot", ptype="bot",
                                      starting_cash=starting_cash)
    reset_portfolio(manual_pid)
    reset_portfolio(bot_pid)

    with _conn() as c:
        c.execute("UPDATE portfolios SET cash = ?, starting_cash = ? WHERE id IN (?, ?)",
                  (starting_cash, starting_cash, manual_pid, bot_pid))
        cur = c.execute(
            "INSERT INTO challenges (name, trade_start, trade_end, challenge_end, "
            "starting_cash, status) VALUES (?, ?, ?, ?, ?, 'active')",
            (name, trade_start.isoformat(), trade_end.isoformat(),
             challenge_end.isoformat(), starting_cash),
        )
        return cur.lastrowid


def get_active_challenge() -> dict | None:
    """Return the most recent active/locked challenge, or None."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM challenges WHERE status IN ('active', 'locked') "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None

    ch = dict(row)
    today = date.today().isoformat()

    if ch["status"] == "active" and today >= ch["trade_end"]:
        with _conn() as c:
            c.execute("UPDATE challenges SET status = 'locked' WHERE id = ?", (ch["id"],))
        ch["status"] = "locked"

    if ch["status"] == "locked" and today >= ch["challenge_end"]:
        winner = _determine_winner(ch["id"])
        with _conn() as c:
            c.execute("UPDATE challenges SET status = 'completed', winner = ? WHERE id = ?",
                      (winner, ch["id"]))
        ch["status"] = "completed"
        ch["winner"] = winner

    return ch


def _determine_winner(challenge_id: int) -> str:
    manual_pid = get_or_create_portfolio("My Portfolio", ptype="manual")
    bot_pid = get_or_create_portfolio("Robo Bot", ptype="bot")
    manual_snaps = get_snapshots(manual_pid)
    bot_snaps = get_snapshots(bot_pid)

    manual_final = manual_snaps["Value"].iloc[-1] if not manual_snaps.empty else 100_000
    bot_final = bot_snaps["Value"].iloc[-1] if not bot_snaps.empty else 100_000

    if manual_final > bot_final:
        return "You"
    elif bot_final > manual_final:
        return "Bot"
    return "Tie"


def get_challenge_history() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM challenges ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def cancel_challenge(challenge_id: int) -> None:
    with _conn() as c:
        c.execute("UPDATE challenges SET status = 'completed', winner = 'Cancelled' "
                  "WHERE id = ?", (challenge_id,))


def is_trading_allowed() -> tuple[bool, str]:
    """Check if trading is currently allowed based on challenge state.

    Returns (allowed, reason).
    """
    ch = get_active_challenge()
    if ch is None:
        return True, "No active challenge — free trading."

    today = date.today().isoformat()

    if ch["status"] == "active" and today < ch["trade_end"]:
        days_left = (date.fromisoformat(ch["trade_end"]) - date.today()).days
        return True, f"Trading window open — {days_left} day(s) left to trade."

    if ch["status"] == "locked":
        days_left = (date.fromisoformat(ch["challenge_end"]) - date.today()).days
        return False, f"Portfolios locked — {days_left} day(s) until challenge ends."

    return True, "Challenge complete — free trading."


# ── Trade helpers ─────────────────────────────────────────────────────────

def execute_trade(pid: int, ticker: str, side: str, shares: float,
                  price: float, reason: str = "", strategy: str = "") -> str | None:
    """Execute a paper trade. Returns an error string or None on success."""
    with _conn() as c:
        p = c.execute("SELECT cash FROM portfolios WHERE id = ?", (pid,)).fetchone()
        if not p:
            return "Portfolio not found"

        if side == "buy":
            cost = shares * price
            if cost > p["cash"]:
                return f"Not enough cash (${p['cash']:,.2f} available, ${cost:,.2f} needed)"
            c.execute("UPDATE portfolios SET cash = cash - ? WHERE id = ?", (cost, pid))
        elif side == "sell":
            held = _shares_held(c, pid, ticker)
            if shares > held:
                return f"Only {held:.2f} shares held"
            proceeds = shares * price
            c.execute("UPDATE portfolios SET cash = cash + ? WHERE id = ?", (proceeds, pid))

        c.execute(
            "INSERT INTO trades (portfolio_id, ticker, side, shares, price, reason, strategy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pid, ticker, side, shares, price, reason, strategy),
        )
    return None


def _shares_held(c: sqlite3.Connection, pid: int, ticker: str) -> float:
    row = c.execute("""
        SELECT COALESCE(SUM(CASE WHEN side='buy' THEN shares ELSE -shares END), 0) AS held
        FROM trades WHERE portfolio_id = ? AND ticker = ?
    """, (pid, ticker)).fetchone()
    return row["held"]


def get_holdings(pid: int) -> pd.DataFrame:
    with _conn() as c:
        rows = c.execute("""
            SELECT ticker,
                   SUM(CASE WHEN side='buy' THEN shares ELSE -shares END) AS shares,
                   SUM(CASE WHEN side='buy' THEN shares * price ELSE 0 END) AS total_cost,
                   SUM(CASE WHEN side='buy' THEN shares ELSE 0 END) AS total_bought
            FROM trades
            WHERE portfolio_id = ?
            GROUP BY ticker
            HAVING shares > 0.001
        """, (pid,)).fetchall()

    if not rows:
        return pd.DataFrame(columns=["Ticker", "Shares", "Avg Cost", "Current Price",
                                      "Market Value", "P&L ($)", "P&L (%)"])

    records = []
    for r in rows:
        avg_cost = r["total_cost"] / r["total_bought"] if r["total_bought"] else 0
        records.append({
            "Ticker": r["ticker"],
            "Shares": round(r["shares"], 2),
            "Avg Cost": round(avg_cost, 2),
        })

    df = pd.DataFrame(records)
    return df


def get_trade_log(pid: int, limit: int = 100) -> pd.DataFrame:
    with _conn() as c:
        rows = c.execute(
            "SELECT ts, ticker, side, shares, price, reason, strategy "
            "FROM trades WHERE portfolio_id = ? ORDER BY ts DESC LIMIT ?",
            (pid, limit),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["Time", "Ticker", "Side", "Shares", "Price", "Reason", "Strategy"])
    return pd.DataFrame([dict(r) for r in rows]).rename(columns={
        "ts": "Time", "ticker": "Ticker", "side": "Side",
        "shares": "Shares", "price": "Price", "reason": "Reason", "strategy": "Strategy",
    })


# ── Snapshot helpers ──────────────────────────────────────────────────────

def save_snapshot(pid: int, total_value: float, dt: str | None = None) -> None:
    dt = dt or date.today().isoformat()
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO daily_snapshots (portfolio_id, dt, total_value) "
            "VALUES (?, ?, ?)",
            (pid, dt, total_value),
        )


def get_snapshots(pid: int) -> pd.DataFrame:
    with _conn() as c:
        rows = c.execute(
            "SELECT dt, total_value FROM daily_snapshots "
            "WHERE portfolio_id = ? ORDER BY dt",
            (pid,),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["Date", "Value"])
    df = pd.DataFrame([dict(r) for r in rows])
    df.columns = ["Date", "Value"]
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def snapshot_all_portfolios() -> None:
    """Record today's value for every portfolio that doesn't have a snapshot yet.

    Called once per app load so the equity curve fills in even on days
    with no trades.  Skips portfolios with zero holdings and no trades.
    """
    today = date.today().isoformat()
    for p in list_portfolios():
        pid = p["id"]
        with _conn() as c:
            exists = c.execute(
                "SELECT 1 FROM daily_snapshots WHERE portfolio_id = ? AND dt = ?",
                (pid, today),
            ).fetchone()
        if exists:
            continue

        holdings = get_holdings(pid)
        if holdings.empty:
            save_snapshot(pid, p["cash"], today)
            continue

        market_value = 0
        for _, row in holdings.iterrows():
            try:
                tk = yf.Ticker(row["Ticker"])
                price = tk.info.get("currentPrice") or \
                        tk.history(period="1d")["Close"].iloc[-1]
                market_value += row["Shares"] * price
            except Exception:
                continue
        save_snapshot(pid, p["cash"] + market_value, today)


def enrich_holdings_with_prices(holdings_df: pd.DataFrame) -> pd.DataFrame:
    """Add current prices and P&L to a holdings DataFrame."""
    if holdings_df.empty:
        return holdings_df

    df = holdings_df.copy()
    prices = {}
    for ticker in df["Ticker"].unique():
        try:
            prices[ticker] = yf.Ticker(ticker).info.get("currentPrice") or \
                             yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        except Exception:
            prices[ticker] = None

    df["Current Price"] = df["Ticker"].map(prices)
    df["Market Value"] = (df["Shares"] * df["Current Price"]).round(2)
    df["P&L ($)"] = ((df["Current Price"] - df["Avg Cost"]) * df["Shares"]).round(2)
    df["P&L (%)"] = (((df["Current Price"] / df["Avg Cost"]) - 1) * 100).round(1)
    return df


init_db()
