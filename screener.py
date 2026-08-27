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

# Curated AI value-chain: chips/tools, data-center infra, and software platforms.
AI_SLEEVE_GROUPS: dict[str, list[str]] = {
    "Chips": [
        "NVDA", "AVGO", "TSM", "ASML", "AMAT", "LRCX", "KLAC", "AMD",
        "MU", "QCOM", "ARM", "MRVL", "INTC", "SNPS", "CDNS", "TXN",
    ],
    "Infra": [
        "VRT", "ETN", "GEV", "CEG", "VST", "NRG", "EQIX", "DLR",
        "SMCI", "ANET", "DELL", "HPE", "CSCO", "CRDO", "COHR", "CCJ", "PWR",
    ],
    "Software": [
        "MSFT", "GOOGL", "AMZN", "META", "ORCL", "NOW", "CRM", "PLTR",
        "SNOW", "PANW", "CRWD", "DDOG", "NET", "ADSK",
    ],
}
AI_SLEEVES: dict[str, str] = {
    ticker: sleeve
    for sleeve, tickers in AI_SLEEVE_GROUPS.items()
    for ticker in tickers
}
AI_STACK_TICKERS: list[str] = [
    ticker for tickers in AI_SLEEVE_GROUPS.values() for ticker in tickers
]
AI_SLEEVE_ORDER: list[str] = list(AI_SLEEVE_GROUPS.keys()) + ["—"]

UNIVERSES: dict[str, list[str]] = {
    "S&P 500 (~200)": SP500_TICKERS,
    "AI Stack": AI_STACK_TICKERS,
    "Nasdaq 100": NASDAQ100_TICKERS,
    "Mid-Caps (~150)": MIDCAP_TICKERS,
    "International ADRs (~55)": INTERNATIONAL_ADRS,
    "S&P 500 + Mid-Caps (~350)": sorted(set(SP500_TICKERS + MIDCAP_TICKERS)),
    "All Universes (~500)": sorted(set(
        SP500_TICKERS + NASDAQ100_TICKERS + MIDCAP_TICKERS
        + INTERNATIONAL_ADRS + AI_STACK_TICKERS
    )),
}


def _normalize_dividend_yield(raw) -> float:
    """Yahoo sometimes returns yield as a ratio (0.023) and sometimes as percent (2.3)."""
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value <= 0:
        return 0.0
    if value <= 1.0:
        value *= 100
    return round(value, 2)


def _return_pct(close: pd.Series, periods: int) -> float | None:
    """Percent change from *periods* trading days ago to the latest close."""
    if close is None or len(close) <= periods:
        return None
    prev = close.iloc[-1 - periods]
    last = close.iloc[-1]
    if pd.isna(prev) or pd.isna(last) or prev == 0:
        return None
    return round(float((last / prev - 1) * 100), 1)


def _attach_relative_strength(df: pd.DataFrame) -> pd.DataFrame:
    """1-month return minus NVDA's 1-month return. Positive = beating NVDA."""
    if df.empty or "1M %" not in df.columns:
        return df

    nvda_1m = None
    nvda_row = df.loc[df["Ticker"] == "NVDA", "1M %"]
    if not nvda_row.empty and pd.notna(nvda_row.iloc[0]):
        nvda_1m = float(nvda_row.iloc[0])
    else:
        try:
            hist = yf.Ticker("NVDA").history(period="3mo")
            if not hist.empty:
                nvda_1m = _return_pct(hist["Close"], 21)
        except Exception:
            nvda_1m = None

    if nvda_1m is None or pd.isna(nvda_1m):
        df["RS vs NVDA"] = None
    else:
        df["RS vs NVDA"] = (
            pd.to_numeric(df["1M %"], errors="coerce") - nvda_1m
        ).round(1)
    return df


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
            hist = tk.history(period="2y")

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
                    "Div Yield %": _normalize_dividend_yield(info.get("dividendYield")),
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
                    "AI Sleeve": AI_SLEEVES.get(ticker, "—"),
                    "1D %": _return_pct(close, 1),
                    "1W %": _return_pct(close, 5),
                    "1M %": _return_pct(close, 21),
                    "3M %": _return_pct(close, 63),
                    "6M %": _return_pct(close, 126),
                    "12M %": _return_pct(close, 252),
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
        df = _attach_relative_strength(df)
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
        return {
            "regime": "normal",
            "label": "Normal",
            "color": "green",
            "stats": {},
            "scope": "this universe",
        }

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
        return {
            "regime": "selloff",
            "label": "Broad Selloff",
            "color": "red",
            "stats": stats,
            "scope": "this universe",
        }
    elif selloff_signals == 1:
        return {
            "regime": "stressed",
            "label": "Stressed",
            "color": "orange",
            "stats": stats,
            "scope": "this universe",
        }
    else:
        return {
            "regime": "normal",
            "label": "Normal",
            "color": "green",
            "stats": stats,
            "scope": "this universe",
        }


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
    """Widen filters in stress; do not slide a momentum RSI band down into oversold.

    Lower RSI bound moves down. Upper RSI bound stays put when the preset already
    has a floor (Momentum). When the floor is 0 (Value / Oversold / Dividend), the
    ceiling rises so more names can pass instead of getting squeezed.
    """
    if regime == "normal":
        return preset

    adj = REGIME_ADJUSTMENTS.get(regime, {})
    p = {**preset}

    rsi_lo, rsi_hi = p["rsi"]
    shift = adj.get("rsi_shift", 0)
    new_lo = max(0.0, rsi_lo + shift)
    if rsi_lo > 0:
        new_hi = rsi_hi
    else:
        new_hi = min(100.0, rsi_hi - shift)
    p["rsi"] = (new_lo, max(new_lo + 5.0, new_hi))

    p["pct_high"] = max(-80.0, p["pct_high"] + adj.get("pct_high_shift", 0))
    p["div_min"] = round(p["div_min"] * adj.get("div_min_mult", 1.0), 1)
    p["pe"] = (p["pe"][0], p["pe"][1] + adj.get("pe_hi_add", 0))

    return p


# Fundamentals with missing values should not fail a range filter (unprofitable
# AI names often have no P/E). Technicals still drop NaN so we don't pass names
# we cannot actually measure.
KEEP_NA_FILTER_COLS = {
    "P/E", "Fwd P/E", "P/B", "Market Cap", "Div Yield %",
    "Revenue Growth %", "Profit Margin %", "EPS", "Beta",
}


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply a dict of {column: (min, max)} range filters to *df*."""
    result = df.copy()
    for col, (lo, hi) in filters.items():
        if col not in result.columns:
            continue
        series = pd.to_numeric(result[col], errors="coerce")
        mask = pd.Series(True, index=result.index)
        if lo is not None:
            mask &= series >= lo
        if hi is not None:
            mask &= series <= hi
        if col in KEEP_NA_FILTER_COLS:
            mask |= series.isna()
        result = result[mask]
    return result


DIVIDEND_SECTORS = [
    "Utilities", "Consumer Defensive", "Real Estate", "Energy",
    "Financial Services", "Communication Services",
]
ALL_MA_OPTIONS = ["None", "Above 50-MA", "Above 200-MA", "Golden Cross (50 > 200)"]
MARKET_CAP_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "Any": (0, None),
    "Mega (>200B)": (200_000_000_000, None),
    "Large (10B–200B)": (10_000_000_000, 200_000_000_000),
    "Mid (2B–10B)": (2_000_000_000, 10_000_000_000),
    "Small (<2B)": (0, 2_000_000_000),
}

PRESETS: dict[str, dict] = {
    "No Preset": {
        "pe": (0.0, 80.0),
        "mktcap": "Any",
        "div_min": 0.0,
        "rsi": (10.0, 90.0),
        "ma": "None",
        "vol_spike": 0.0,
        "pct_high": -80.0,
        "sectors": None,
    },
    "Value Hunting": {
        "pe": (2.0, 25.0),
        "mktcap": "Any",
        "div_min": 1.0,
        "rsi": (0.0, 55.0),
        "ma": "None",
        "vol_spike": 0.0,
        "pct_high": -80.0,
        "sectors": None,
    },
    "Momentum / Growth": {
        "pe": (0.0, 80.0),
        "mktcap": "Any",
        "div_min": 0.0,
        "rsi": (45.0, 75.0),
        "ma": "Above 50-MA",
        "vol_spike": 0.0,
        "pct_high": -15.0,
        "sectors": None,
    },
    "Dividend Income": {
        "pe": (2.0, 35.0),
        "mktcap": "Any",
        "div_min": 1.5,
        "rsi": (0.0, 70.0),
        "ma": "None",
        "vol_spike": 0.0,
        "pct_high": -80.0,
        "sectors": DIVIDEND_SECTORS,
    },
    "Oversold Bounce": {
        "pe": (0.0, 80.0),
        "mktcap": "Any",
        "div_min": 0.0,
        "rsi": (0.0, 40.0),
        "ma": "None",
        "vol_spike": 0.5,
        "pct_high": -80.0,
        "sectors": None,
    },
    "AI Momentum": {
        "pe": (0.0, 150.0),
        "mktcap": "Any",
        "div_min": 0.0,
        "rsi": (0.0, 100.0),
        "ma": "None",
        "vol_spike": 0.0,
        "pct_high": -80.0,
        "sectors": None,
    },
}


def apply_preset_filters(
    df: pd.DataFrame,
    preset_name: str,
    regime: str = "normal",
) -> pd.DataFrame:
    """Apply a strategy's default filters, including regime widening."""
    preset = PRESETS.get(preset_name, PRESETS["No Preset"])
    p = adjust_preset_for_regime(preset, regime)
    result = df.copy()

    if p.get("sectors"):
        sector_match = result["Sector"].isin(p["sectors"])
        if sector_match.any():
            result = result[sector_match]

    mktcap_lo, mktcap_hi = MARKET_CAP_BOUNDS.get(p.get("mktcap", "Any"), (0, None))
    range_filters = {
        "P/E": p["pe"],
        "Market Cap": (mktcap_lo, mktcap_hi),
        "Div Yield %": (p["div_min"], None),
        "RSI (14)": p["rsi"],
        "Vol vs Avg": (p["vol_spike"], None),
        "% from 52w High": (p["pct_high"], None),
    }
    result = apply_filters(result, range_filters)

    ma_filter = p.get("ma", "None")
    if ma_filter == "Above 50-MA":
        result = result[result["Above 50-MA"] == True]
    elif ma_filter == "Above 200-MA":
        result = result[result["Above 200-MA"] == True]
    elif ma_filter == "Golden Cross (50 > 200)":
        result = result[
            result["50-day MA"].notna()
            & result["200-day MA"].notna()
            & (result["50-day MA"] > result["200-day MA"])
        ]
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
    "AI Momentum": [
        ("Revenue Growth %", 0.25, False),
        ("% from 52w High",  0.20, False),
        ("1M %",             0.20, False),
        ("RS vs NVDA",       0.15, False),
        ("RSI (14)",         0.10, False),
        ("Profit Margin %",  0.10, False),
    ],
}


def compute_ai_setup_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score each row 0–100 against the AI 'ideal setup'.

    Points: trend 30, RSI zone 20, vs NVDA 20, near 52-week high 15,
    1-week/1-month participation 15. This is an absolute checklist, not a
    percentile rank against the current universe.
    """
    out = pd.DataFrame(index=df.index)
    if df.empty:
        out["Setup Score"] = pd.Series(dtype=float)
        out["Setup"] = pd.Series(dtype=str)
        out["Setup Note"] = pd.Series(dtype=str)
        return out

    above50 = df.get("Above 50-MA", False)
    if not isinstance(above50, pd.Series):
        above50 = pd.Series(False, index=df.index)
    above50 = above50.fillna(False).astype(bool)

    above200 = df.get("Above 200-MA", False)
    if not isinstance(above200, pd.Series):
        above200 = pd.Series(False, index=df.index)
    above200 = above200.fillna(False).astype(bool)

    trend_pts = np.where(above50, 15, 0) + np.where(above200, 15, 0)

    def _col(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[name], errors="coerce")

    rsi = _col("RSI (14)")
    rsi_pts = np.select(
        [
            (rsi >= 50) & (rsi <= 70),
            ((rsi >= 40) & (rsi < 50)) | ((rsi > 70) & (rsi <= 80)),
            ((rsi >= 30) & (rsi < 40)) | ((rsi > 80) & (rsi <= 85)),
        ],
        [20, 12, 5],
        default=0,
    )

    rs = _col("RS vs NVDA")
    rs_pts = np.select(
        [rs >= 5, rs >= 0, rs >= -5],
        [20, 15, 8],
        default=0,
    )

    drawdown = _col("% from 52w High")
    high_pts = np.select(
        [drawdown >= -8, drawdown >= -15, drawdown >= -25],
        [15, 10, 5],
        default=0,
    )

    week = _col("1W %")
    month = _col("1M %")
    align_pts = np.where(month > 0, 8, 0) + np.where(week > 0, 7, 0)

    total = (
        pd.Series(trend_pts, index=df.index)
        + pd.Series(rsi_pts, index=df.index)
        + pd.Series(rs_pts, index=df.index)
        + pd.Series(high_pts, index=df.index)
        + pd.Series(align_pts, index=df.index)
    )
    out["Setup Score"] = total.clip(0, 100).round(0)

    out["Setup"] = np.select(
        [out["Setup Score"] >= 80, out["Setup Score"] >= 60, out["Setup Score"] >= 40],
        ["Ideal", "Good", "Mixed"],
        default="Weak",
    )

    notes: list[str] = []
    for i in df.index:
        bits: list[str] = []
        if bool(above50.loc[i]) and bool(above200.loc[i]):
            bits.append("above 50 & 200")
        elif bool(above50.loc[i]):
            bits.append("above 50 only")
        elif bool(above200.loc[i]):
            bits.append("above 200, below 50")
        else:
            bits.append("below both MAs")

        rsi_i = rsi.loc[i] if i in rsi.index else np.nan
        if pd.isna(rsi_i):
            bits.append("no RSI")
        elif 50 <= rsi_i <= 70:
            bits.append(f"RSI {rsi_i:.0f} ideal")
        elif rsi_i > 80:
            bits.append(f"RSI {rsi_i:.0f} stretched")
        elif rsi_i < 40:
            bits.append(f"RSI {rsi_i:.0f} washed out")
        else:
            bits.append(f"RSI {rsi_i:.0f} OK")

        rs_i = rs.loc[i] if i in rs.index else np.nan
        if pd.notna(rs_i):
            bits.append("beating NVDA" if rs_i >= 0 else "lagging NVDA")

        dd_i = drawdown.loc[i] if i in drawdown.index else np.nan
        if pd.notna(dd_i):
            if dd_i >= -8:
                bits.append("near high")
            elif dd_i < -25:
                bits.append("far from high")

        notes.append(" · ".join(bits))

    out["Setup Note"] = notes
    return out


def compute_score(df: pd.DataFrame, preset: str) -> pd.Series:
    """Return a 0–100 composite score for each row based on *preset* weights.

    AI Momentum uses an absolute setup checklist. Other presets use percentile
    ranks within the filtered set. Missing values get 50th-percentile (neutral).
    """
    if preset == "AI Momentum":
        return compute_ai_setup_score(df)["Setup Score"]

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
