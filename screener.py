"""Stock screening engine — fetches data from Yahoo Finance, computes
fundamental + technical indicators, and applies user-defined filters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

# ── Ticker universes ──────────────────────────────────────────────────────

SP500_TICKERS: list[str] = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEP",
    "AFL", "AIG", "AMAT", "AMD", "AMGN", "AMP", "AMZN", "ANET", "ANSS", "AON",
    "APD", "APH", "AVGO", "AXP", "BA", "BAC", "BDX", "BK", "BKNG", "BLK",
    "BMY", "BRK-B", "BSX", "C", "CAT", "CB", "CCI", "CDNS", "CEG", "CHTR",
    "CI", "CL", "CMCSA", "CME", "CMG", "COF", "COP", "COST", "CRM", "CSCO",
    "CTAS", "CVS", "CVX", "D", "DD", "DE", "DHR", "DIS", "DLR", "DOW",
    "DUK", "ECL", "EL", "EMR", "ENPH", "EOG", "EQR", "ETN", "EW", "EXC",
    "F", "FAST", "FCX", "FDX", "FI", "FISV", "GD", "GE", "GILD", "GIS",
    "GM", "GOOG", "GOOGL", "GPN", "GS", "HCA", "HD", "HON", "IBM", "ICE",
    "INTC", "INTU", "ISRG", "ITW", "JNJ", "JPM", "KHC", "KLAC", "KMB", "KO",
    "LHX", "LIN", "LLY", "LMT", "LOW", "LRCX", "MA", "MCD", "MCHP", "MCK",
    "MCO", "MDLZ", "MDT", "MET", "META", "MMC", "MMM", "MO", "MPC", "MRK",
    "MS", "MSFT", "MSI", "MU", "NEE", "NEM", "NFLX", "NKE", "NOC", "NOW",
    "NSC", "NVDA", "ORCL", "ORLY", "OXY", "PANW", "PEP", "PFE", "PG", "PGR",
    "PH", "PLD", "PM", "PNC", "PSA", "PSX", "PYPL", "QCOM", "REGN", "ROP",
    "ROST", "RTX", "SBUX", "SCHW", "SHW", "SLB", "SMCI", "SNPS", "SO", "SPG",
    "SPGI", "SRE", "SYK", "SYY", "T", "TDG", "TGT", "TJX", "TMO", "TMUS",
    "TRV", "TSLA", "TT", "TXN", "UNH", "UNP", "UPS", "URI", "USB", "V",
    "VICI", "VLO", "VRSK", "VRTX", "VZ", "WBA", "WEC", "WELL", "WFC", "WM",
    "WMT", "XEL", "XOM", "ZTS",
]

NASDAQ100_TICKERS: list[str] = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD",
    "AMGN", "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AZN", "BIIB",
    "BKNG", "BKR", "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COIN",
    "COST", "CPRT", "CRWD", "CSCO", "CSGP", "CTAS", "CTSH", "DASH", "DDOG",
    "DLTR", "DXCM", "EA", "EXC", "FANG", "FAST", "FTNT", "GEHC", "GFS",
    "GILD", "GOOG", "GOOGL", "HON", "IDXX", "ILMN", "INTC", "INTU", "ISRG",
    "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR", "MCHP", "MDB",
    "MDLZ", "MELI", "META", "MNST", "MRVL", "MSFT", "MU", "NFLX", "NVDA",
    "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP",
    "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SMCI", "SNPS", "TEAM",
    "TMUS", "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY",
    "XEL", "ZS",
]

MIDCAP_TICKERS: list[str] = [
    "ACM", "AES", "ALGN", "ALLY", "AMH", "APA", "AR", "AXON", "BALL", "BAX",
    "BIO", "BWA", "CAG", "CE", "CFG", "CHD", "CLX", "CMA", "CNP", "COO",
    "CPB", "CRL", "CZR", "DAL", "DECK", "DFS", "DG", "DINO", "DKS", "DOC",
    "DPZ", "DRI", "DVA", "EBAY", "EFX", "EIX", "ENPH", "EPAM", "ESS", "ETSY",
    "EXPE", "FFIV", "FMC", "FNF", "FSLR", "GNRC", "GPK", "GPC", "GRAB", "HAL",
    "HAS", "HBAN", "HOLX", "HPE", "HPQ", "HST", "HWM", "IEX", "INCY", "IPG",
    "IRM", "JBHT", "JBL", "JKHY", "KEY", "KIM", "KMI", "L", "LDOS", "LEA",
    "LKQ", "LNT", "LPLA", "LUV", "LVS", "LW", "MAA", "MAS", "MGM", "MKTX",
    "MOH", "MPWR", "MRO", "MTCH", "MTD", "NCLH", "NI", "NRG", "NTAP", "NTRS",
    "NVR", "NWS", "OKE", "OTIS", "PAYC", "PEAK", "PFG", "PKG", "POOL", "PPG",
    "PPL", "PTC", "PVH", "QDEL", "RCL", "REG", "RF", "RJF", "RMD", "ROL",
    "RVTY", "SBAC", "SEE", "SJM", "SNA", "STX", "SWK", "SYF", "TAP", "TECH",
    "TEL", "TER", "TFX", "TRGP", "TSCO", "TXT", "TYL", "UAL", "UDR", "ULTA",
    "VFC", "VTRS", "VTR", "WAB", "WAT", "WDC", "WYNN", "XYL", "YUM", "ZBRA",
]

INTERNATIONAL_ADRS: list[str] = [
    "ASML", "AZN", "BABA", "BIDU", "BP", "BTI", "BUD", "CIB", "DEO", "ERIC",
    "GRAB", "GSK", "HDB", "HSBK", "HSBC", "IBN", "INFY", "JD", "KB", "KT",
    "LI", "LOGI", "LPL", "MELI", "MFG", "NIO", "NOK", "NVO", "NVS", "ORAN",
    "PBR", "PDD", "PHG", "RIO", "SAP", "SE", "SHG", "SHOP", "SID", "SNY",
    "SONY", "STM", "SU", "TAK", "TM", "TME", "TOST", "TSM", "UBS", "UL",
    "VALE", "VOD", "WIT", "WPP", "XPEV",
]

UNIVERSES: dict[str, list[str]] = {
    "S&P 500 (~200)": SP500_TICKERS,
    "Nasdaq 100": NASDAQ100_TICKERS,
    "Mid-Caps (~150)": MIDCAP_TICKERS,
    "International ADRs (~55)": INTERNATIONAL_ADRS,
    "S&P 500 + Mid-Caps (~350)": sorted(set(SP500_TICKERS + MIDCAP_TICKERS)),
    "All Universes (~500)": sorted(set(SP500_TICKERS + NASDAQ100_TICKERS + MIDCAP_TICKERS + INTERNATIONAL_ADRS)),
}


def _rsi(series: pd.Series, period: int = 14) -> float:
    """Compute the latest RSI value from a price series."""
    if series is None or len(series) < period + 1:
        return np.nan
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


def fetch_screening_data(
    tickers: list[str] | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """Download price history + fundamentals for *tickers* and return a
    single DataFrame with one row per ticker."""

    tickers = tickers or SP500_TICKERS
    records: list[dict] = []

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i / len(tickers), f"Fetching {ticker}…")
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            hist = tk.history(period="1y")

            if hist.empty:
                continue

            close = hist["Close"]
            volume = hist["Volume"]
            latest_price = close.iloc[-1]

            ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
            ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
            avg_vol_20 = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else np.nan
            rsi_14 = _rsi(close)

            records.append(
                {
                    "Ticker": ticker,
                    "Price": round(latest_price, 2),
                    "Market Cap": info.get("marketCap"),
                    "P/E": info.get("trailingPE"),
                    "Fwd P/E": info.get("forwardPE"),
                    "EPS": info.get("trailingEps"),
                    "Div Yield %": round(info.get("dividendYield", 0) * 100, 2)
                    if info.get("dividendYield")
                    else 0.0,
                    "P/B": info.get("priceToBook"),
                    "Revenue Growth %": round(info.get("revenueGrowth", 0) * 100, 1)
                    if info.get("revenueGrowth")
                    else None,
                    "Profit Margin %": round(info.get("profitMargins", 0) * 100, 1)
                    if info.get("profitMargins")
                    else None,
                    "Beta": info.get("beta"),
                    "52w High": info.get("fiftyTwoWeekHigh"),
                    "52w Low": info.get("fiftyTwoWeekLow"),
                    "50-day MA": round(ma50, 2) if not np.isnan(ma50) else None,
                    "200-day MA": round(ma200, 2) if not np.isnan(ma200) else None,
                    "RSI (14)": round(rsi_14, 1) if not np.isnan(rsi_14) else None,
                    "Avg Vol (20d)": int(avg_vol_20) if not np.isnan(avg_vol_20) else None,
                    "Volume": int(volume.iloc[-1]),
                    "Sector": info.get("sector", "—"),
                    "Industry": info.get("industry", "—"),
                    "Name": info.get("shortName", ticker),
                }
            )
        except Exception:
            continue

    if progress_callback:
        progress_callback(1.0, "Done")

    df = pd.DataFrame(records)
    if not df.empty:
        df["Above 50-MA"] = df["Price"] > df["50-day MA"]
        df["Above 200-MA"] = df["Price"] > df["200-day MA"]
        df["Vol vs Avg"] = (df["Volume"] / df["Avg Vol (20d)"]).round(2)
        df["% from 52w High"] = (((df["Price"] - df["52w High"]) / df["52w High"]) * 100).round(1)
    return df


# ── Market regime detection ───────────────────────────────────────────────

REGIME_THRESHOLDS = {
    "median_rsi": 40,
    "median_drawdown": -15,
    "pct_below_200ma": 55,
}


def detect_regime(df: pd.DataFrame) -> dict:
    """Analyze the loaded universe and return regime classification + stats.

    Returns a dict with:
      - regime: "normal", "stressed", or "selloff"
      - stats: the underlying numbers
      - label / color: for UI display
    """
    if df.empty:
        return {"regime": "normal", "label": "Normal", "color": "green", "stats": {}}

    median_rsi = df["RSI (14)"].median() if df["RSI (14)"].notna().any() else 50.0
    median_drawdown = df["% from 52w High"].median() if df["% from 52w High"].notna().any() else 0.0

    below_200 = df["Above 200-MA"].eq(False).sum()
    total_with_200 = df["Above 200-MA"].notna().sum()
    pct_below_200 = (below_200 / total_with_200 * 100) if total_with_200 > 0 else 0.0

    stats = {
        "median_rsi": round(median_rsi, 1),
        "median_drawdown": round(median_drawdown, 1),
        "pct_below_200ma": round(pct_below_200, 1),
    }

    selloff_signals = 0
    if median_rsi < REGIME_THRESHOLDS["median_rsi"]:
        selloff_signals += 1
    if median_drawdown < REGIME_THRESHOLDS["median_drawdown"]:
        selloff_signals += 1
    if pct_below_200 > REGIME_THRESHOLDS["pct_below_200ma"]:
        selloff_signals += 1

    if selloff_signals >= 2:
        return {"regime": "selloff", "label": "Broad Selloff", "color": "red", "stats": stats}
    elif selloff_signals == 1:
        return {"regime": "stressed", "label": "Stressed", "color": "orange", "stats": stats}
    else:
        return {"regime": "normal", "label": "Normal", "color": "green", "stats": stats}


# Adjustments applied to preset filter values during stressed/selloff regimes
REGIME_ADJUSTMENTS = {
    "stressed": {
        "rsi_shift": -10,
        "pct_high_shift": -10,
        "div_min_mult": 0.7,
        "pe_hi_add": 5,
    },
    "selloff": {
        "rsi_shift": -20,
        "pct_high_shift": -25,
        "div_min_mult": 0.5,
        "pe_hi_add": 10,
    },
}


def adjust_preset_for_regime(preset: dict, regime: str) -> dict:
    """Return a copy of *preset* with filter ranges shifted for the regime."""
    if regime == "normal":
        return preset

    adj = REGIME_ADJUSTMENTS.get(regime, {})
    p = {**preset}

    rsi_lo, rsi_hi = p["rsi"]
    shift = adj.get("rsi_shift", 0)
    p["rsi"] = (max(0.0, rsi_lo + shift), max(10.0, rsi_hi + shift))

    p["pct_high"] = max(-80.0, p["pct_high"] + adj.get("pct_high_shift", 0))
    p["div_min"] = round(p["div_min"] * adj.get("div_min_mult", 1.0), 1)
    p["pe"] = (p["pe"][0], p["pe"][1] + adj.get("pe_hi_add", 0))

    return p


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply a dict of {column: (min, max)} range filters to *df*."""
    result = df.copy()
    for col, (lo, hi) in filters.items():
        if col not in result.columns:
            continue
        series = pd.to_numeric(result[col], errors="coerce")
        if lo is not None:
            result = result[series >= lo]
            series = series.loc[result.index]
        if hi is not None:
            result = result[series <= hi]
    return result


# ── Composite scoring ────────────────────────────────────────────────────

# Each weight tuple: (column, weight, lower_is_better)
SCORE_WEIGHTS: dict[str, list[tuple[str, float, bool]]] = {
    "No Preset": [
        ("P/E",              0.15, True),
        ("Div Yield %",      0.10, False),
        ("RSI (14)",         0.10, True),
        ("Revenue Growth %", 0.15, False),
        ("Profit Margin %",  0.15, False),
        ("% from 52w High",  0.10, False),
        ("Vol vs Avg",       0.05, False),
        ("Fwd P/E",          0.10, True),
        ("P/B",              0.10, True),
    ],
    "Value Hunting": [
        ("P/E",              0.30, True),
        ("Div Yield %",      0.25, False),
        ("RSI (14)",         0.15, True),
        ("Profit Margin %",  0.15, False),
        ("P/B",              0.15, True),
    ],
    "Momentum / Growth": [
        ("RSI (14)",         0.20, False),
        ("% from 52w High",  0.25, False),
        ("Revenue Growth %", 0.30, False),
        ("Vol vs Avg",       0.10, False),
        ("Profit Margin %",  0.15, False),
    ],
    "Dividend Income": [
        ("Div Yield %",      0.35, False),
        ("P/E",              0.20, True),
        ("Profit Margin %",  0.20, False),
        ("RSI (14)",         0.10, True),
        ("P/B",              0.15, True),
    ],
    "Oversold Bounce": [
        ("RSI (14)",         0.30, True),
        ("% from 52w High",  0.25, True),
        ("Vol vs Avg",       0.25, False),
        ("Revenue Growth %", 0.10, False),
        ("P/E",              0.10, True),
    ],
}


def compute_score(df: pd.DataFrame, preset: str) -> pd.Series:
    """Return a 0–100 composite score for each row based on *preset* weights.

    Uses percentile ranks within the filtered set so scores are always
    relative to the current universe.  Missing values get 50th-percentile
    (neutral) so they neither help nor hurt.
    """
    weights = SCORE_WEIGHTS.get(preset, SCORE_WEIGHTS["No Preset"])
    total_weight = sum(w for _, w, _ in weights)
    score = pd.Series(0.0, index=df.index)

    for col, weight, lower_is_better in weights:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        pct = s.rank(pct=True)
        pct = pct.fillna(0.5)
        if lower_is_better:
            pct = 1 - pct
        score += pct * (weight / total_weight)

    return (score * 100).round(1)
