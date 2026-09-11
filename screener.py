"""Stock screening engine — fetches data from Yahoo Finance, computes
fundamental + technical indicators, and applies user-defined filters.

Known limits: all fundamentals come from yfinance's single point-in-time
`.info` snapshot (today's numbers only). There is no multi-year statement
history here (no revenue/margin/debt trend, just the latest figure), and
no point-in-time historical fundamentals, which is why Backtest Lab can't
properly replay fundamentals-driven presets (Long-Term Hold, Value
Hunting, Dividend Income): it can only see today's balance sheet, not
what it looked like on a rebalance date two years ago.
"""

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
    """Normalize Yahoo's dividendYield field to a plain percent (2.3 = 2.3%).

    As of 2025 Yahoo returns this field already scaled as a percent (0.46
    means 0.46%), not the pre-2025 ratio (0.023 meaning 2.3%). Verified by
    cross-checking dividendRate / price against dividendYield across low-
    and high-yield tickers. We no longer assume small values are ratios,
    since that turned MSFT's real ~0.7% yield into a fake "74%". A high
    sanity clamp guards against the field reverting to the old ratio
    style without silently showing an absurd number.
    """
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value <= 0:
        return 0.0
    if value > 40:
        value /= 100
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
                    "Debt/Equity %": info.get("debtToEquity"),
                    "Current Ratio": info.get("currentRatio"),
                    "ROE %": round(info.get("returnOnEquity", 0) * 100, 1)
                    if info.get("returnOnEquity") is not None
                    else None,
                    "FCF Yield %": round(info.get("freeCashflow") / info.get("marketCap") * 100, 2)
                    if info.get("freeCashflow") and info.get("marketCap")
                    else None,
                    "Payout Ratio %": round(info.get("payoutRatio", 0) * 100, 1)
                    if info.get("payoutRatio") is not None
                    else None,
                    "PEG": info.get("trailingPegRatio")
                    if info.get("trailingPegRatio") is not None
                    else info.get("pegRatio"),
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
    "Debt/Equity %", "Current Ratio", "ROE %", "FCF Yield %",
    "Payout Ratio %", "PEG",
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
    "Long-Term Hold": {
        "pe": (5.0, 45.0),
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


def compute_lt_hold_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score each row 0–100 for long-term hold quality (any sector).

    Points: scale 15, profitability 20, growth 15, valuation 15,
    balance sheet quality 20, stability 15. Absolute checklist, not a
    percentile rank. Uses Yahoo fundamentals only (see module docstring
    limits: no multi-year statement history, no point-in-time backtesting).
    """
    out = pd.DataFrame(index=df.index)
    if df.empty:
        out["Hold Score"] = pd.Series(dtype=float)
        out["Hold Tier"] = pd.Series(dtype=str)
        out["Hold Note"] = pd.Series(dtype=str)
        return out

    def _col(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[name], errors="coerce")

    sector = df.get("Sector", pd.Series("—", index=df.index))
    if not isinstance(sector, pd.Series):
        sector = pd.Series("—", index=df.index)
    is_financial = sector.eq("Financial Services")

    mktcap = _col("Market Cap")
    scale_pts = np.select(
        [mktcap >= 200_000_000_000, mktcap >= 10_000_000_000, mktcap >= 2_000_000_000],
        [15, 11, 6],
        default=0,
    )

    margin = _col("Profit Margin %")
    margin_pts = np.select(
        [margin >= 20, margin >= 15, margin >= 10, margin >= 5, margin > 0],
        [12, 10, 7, 4, 2],
        default=0,
    )
    roe = _col("ROE %")
    roe_pts = np.select([roe >= 20, roe >= 15, roe >= 10, roe > 0], [8, 6, 4, 2], default=0)
    profitability_pts = np.minimum(margin_pts + roe_pts, 20)

    growth = _col("Revenue Growth %")
    growth_pts = np.select(
        [growth >= 15, growth >= 10, growth >= 5, growth >= 0],
        [15, 12, 9, 5],
        default=0,
    )

    pe = _col("P/E")
    peg = _col("PEG")
    pe_pts = np.select(
        [
            (pe >= 12) & (pe <= 25),
            ((pe >= 8) & (pe < 12)) | ((pe > 25) & (pe <= 35)),
            ((pe >= 5) & (pe < 8)) | ((pe > 35) & (pe <= 45)),
        ],
        [12, 8, 4],
        default=2,
    )
    pe_pts = np.where(pe.isna() & (margin > 0), 4, pe_pts)
    # PEG rewards growth that justifies the price, penalizes growth that
    # doesn't. A 35 P/E growing 30%/yr should not score worse than a 20 P/E
    # growing 2%/yr; a rich multiple with no growth to back it should.
    peg_adj = np.select([peg <= 1.0, peg <= 2.0, peg > 3.0], [3, 1, -2], default=0)
    val_pts = np.clip(pe_pts + peg_adj, 0, 15)

    debt_eq = _col("Debt/Equity %")
    current_ratio = _col("Current Ratio")
    fcf_yield = _col("FCF Yield %")

    # Debt/Equity isn't meaningful for banks/insurers (deposits and float
    # look like "debt"); give them a neutral score instead of penalizing.
    debt_pts = np.select(
        [debt_eq <= 50, debt_eq <= 100, debt_eq <= 200],
        [8, 5, 2],
        default=0,
    )
    debt_pts = np.where(is_financial, 6, debt_pts)
    debt_pts = np.where((~is_financial) & debt_eq.isna(), 3, debt_pts)

    current_ratio_pts = np.select(
        [current_ratio >= 2.0, current_ratio >= 1.5, current_ratio >= 1.0, current_ratio > 0],
        [6, 5, 3, 1],
        default=0,
    )
    current_ratio_pts = np.where(current_ratio.isna(), 3, current_ratio_pts)

    fcf_pts = np.select(
        [fcf_yield >= 8, fcf_yield >= 5, fcf_yield >= 2, fcf_yield > 0],
        [6, 4, 2, 1],
        default=0,
    )
    fcf_pts = np.where(fcf_yield.isna(), 2, fcf_pts)

    balance_sheet_pts = np.minimum(debt_pts + current_ratio_pts + fcf_pts, 20)

    beta = _col("Beta")
    div = _col("Div Yield %")
    payout = _col("Payout Ratio %")
    above200 = df.get("Above 200-MA", False)
    if not isinstance(above200, pd.Series):
        above200 = pd.Series(False, index=df.index)
    above200 = above200.fillna(False).astype(bool)

    beta_pts = np.select([beta <= 1.0, beta <= 1.3], [6, 4], default=1)
    # A yield only counts as "safe" if it's covered by earnings. Payout
    # ratio over 100% means the dividend is being funded by debt/cash, not
    # profit, a common precursor to a cut.
    div_safety_pts = np.select(
        [
            (div >= 1.0) & (payout <= 60),
            (div >= 1.0) & (payout <= 80),
            (div >= 1.0) & (payout <= 100),
            (div >= 1.0) & (payout > 100),
        ],
        [5, 3, 1, 0],
        default=0,
    )
    div_safety_pts = np.where((div >= 1.0) & payout.isna(), 3, div_safety_pts)
    trend_pts = np.where(above200, 4, 0)
    stability_pts = np.minimum(beta_pts + div_safety_pts + trend_pts, 15)

    total = (
        pd.Series(scale_pts, index=df.index)
        + pd.Series(profitability_pts, index=df.index)
        + pd.Series(growth_pts, index=df.index)
        + pd.Series(val_pts, index=df.index)
        + pd.Series(balance_sheet_pts, index=df.index)
        + pd.Series(stability_pts, index=df.index)
    )
    out["Hold Score"] = total.clip(0, 100).round(0)

    out["Hold Tier"] = np.select(
        [
            out["Hold Score"] >= 75,
            out["Hold Score"] >= 60,
            out["Hold Score"] >= 45,
        ],
        ["Core", "Quality", "Watch"],
        default="Pass",
    )

    notes: list[str] = []
    for i in df.index:
        bits: list[str] = []
        mc_i = mktcap.loc[i] if i in mktcap.index else np.nan
        if pd.notna(mc_i):
            if mc_i >= 200_000_000_000:
                bits.append("mega cap")
            elif mc_i >= 10_000_000_000:
                bits.append("large cap")
            elif mc_i >= 2_000_000_000:
                bits.append("mid cap")
            else:
                bits.append("small cap")

        margin_i = margin.loc[i] if i in margin.index else np.nan
        if pd.notna(margin_i):
            if margin_i >= 15:
                bits.append(f"margin {margin_i:.0f}% strong")
            elif margin_i > 0:
                bits.append(f"margin {margin_i:.0f}%")
            else:
                bits.append("unprofitable")

        growth_i = growth.loc[i] if i in growth.index else np.nan
        if pd.notna(growth_i):
            bits.append(f"rev {growth_i:+.0f}%" if growth_i >= 0 else f"rev {growth_i:.0f}% shrink")

        pe_i = pe.loc[i] if i in pe.index else np.nan
        if pd.notna(pe_i):
            if 12 <= pe_i <= 25:
                bits.append(f"P/E {pe_i:.0f} fair")
            elif pe_i > 35:
                bits.append(f"P/E {pe_i:.0f} rich")
            else:
                bits.append(f"P/E {pe_i:.0f}")

        peg_i = peg.loc[i] if i in peg.index else np.nan
        if pd.notna(peg_i):
            if peg_i <= 1.0:
                bits.append(f"PEG {peg_i:.1f} cheap for growth")
            elif peg_i > 3.0:
                bits.append(f"PEG {peg_i:.1f} rich for growth")

        debt_i = debt_eq.loc[i] if i in debt_eq.index else np.nan
        if bool(is_financial.loc[i]):
            pass
        elif pd.isna(debt_i):
            bits.append("debt unknown")
        elif debt_i <= 50:
            bits.append("low debt")
        elif debt_i > 150:
            bits.append(f"debt/equity {debt_i:.0f}% high")

        roe_i = roe.loc[i] if i in roe.index else np.nan
        if pd.notna(roe_i) and roe_i >= 15:
            bits.append(f"ROE {roe_i:.0f}% strong")

        if bool(above200.loc[i]):
            bits.append("above 200-day")
        else:
            bits.append("below 200-day")

        div_i = div.loc[i] if i in div.index else np.nan
        payout_i = payout.loc[i] if i in payout.index else np.nan
        if pd.notna(div_i) and div_i >= 1.0:
            if pd.notna(payout_i) and payout_i > 100:
                bits.append(f"yield {div_i:.1f}% payout {payout_i:.0f}% unsustainable")
            else:
                bits.append(f"yield {div_i:.1f}%")

        notes.append(" · ".join(bits))

    out["Hold Note"] = notes
    return out


def compute_score(df: pd.DataFrame, preset: str) -> pd.Series:
    """Return a 0–100 composite score for each row based on *preset* weights.

    AI Momentum and Long-Term Hold use absolute checklists. Other presets use
    percentile ranks within the filtered set. Missing values get 50th-percentile.
    """
    if preset == "AI Momentum":
        return compute_ai_setup_score(df)["Setup Score"]
    if preset == "Long-Term Hold":
        return compute_lt_hold_score(df)["Hold Score"]

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
