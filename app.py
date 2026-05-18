"""Stock Screener — Streamlit app with fundamental + technical filters."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from screener import (
    SP500_TICKERS, UNIVERSES, fetch_screening_data, apply_filters,
    compute_score, detect_regime, adjust_preset_for_regime,
)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Screener", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetric"] {background: #f8f9fb; border-radius: .5rem; padding: .75rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Stock Screener")
st.caption("Yahoo Finance data · Fundamentals + Technicals")

# ── Sidebar: data loading ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Universe")

    universe_label = st.selectbox(
        "Stock universe",
        list(UNIVERSES.keys()) + ["Custom"],
        help="Pick a pre-built universe or choose Custom to type your own tickers.",
    )

    if universe_label == "Custom":
        custom_input = st.text_area(
            "Custom tickers (comma-separated)",
            placeholder="e.g. AAPL, TSLA, NVDA, MSFT",
        )
        selected_tickers = (
            [t.strip().upper() for t in custom_input.split(",") if t.strip()]
            if custom_input.strip()
            else None
        )
    else:
        selected_tickers = UNIVERSES[universe_label]
        st.caption(f"{len(selected_tickers)} tickers")

    load_btn = st.button("Load / Refresh Data", type="primary", use_container_width=True)


@st.cache_data(ttl=900, show_spinner=False)
def _load(tickers_key: str, tickers: list[str] | None):
    bar = st.progress(0, text="Starting…")
    df = fetch_screening_data(tickers=tickers, progress_callback=bar.progress)
    bar.empty()
    return df


cache_key = (
    ",".join(selected_tickers) if selected_tickers else "sp500"
)

if load_btn or "df" not in st.session_state:
    st.session_state["df"] = _load(cache_key, selected_tickers)

df: pd.DataFrame = st.session_state["df"]

if df.empty:
    st.warning("No data loaded yet. Click **Load / Refresh Data** in the sidebar.")
    st.stop()

# ── Market regime detection ────────────────────────────────────────────────
regime_info = detect_regime(df)
regime = regime_info["regime"]

if regime == "selloff":
    st.error(
        f"**Market Regime: {regime_info['label']}** — "
        f"Median RSI {regime_info['stats']['median_rsi']}, "
        f"Median drawdown {regime_info['stats']['median_drawdown']}%, "
        f"{regime_info['stats']['pct_below_200ma']}% below 200-MA. "
        "Preset filters have been widened automatically."
    )
elif regime == "stressed":
    st.warning(
        f"**Market Regime: {regime_info['label']}** — "
        f"Median RSI {regime_info['stats']['median_rsi']}, "
        f"Median drawdown {regime_info['stats']['median_drawdown']}%, "
        f"{regime_info['stats']['pct_below_200ma']}% below 200-MA. "
        "Preset filters slightly relaxed."
    )
else:
    st.success(
        f"**Market Regime: Normal** — "
        f"Median RSI {regime_info['stats']['median_rsi']}, "
        f"Median drawdown {regime_info['stats']['median_drawdown']}%, "
        f"{regime_info['stats']['pct_below_200ma']}% below 200-MA."
    )

# ── Presets ────────────────────────────────────────────────────────────────
DIVIDEND_SECTORS = ["Utilities", "Consumer Defensive", "Real Estate", "Energy",
                    "Financial Services", "Communication Services"]
ALL_MA_OPTIONS = ["None", "Above 50-MA", "Above 200-MA", "Golden Cross (50 > 200)"]

PRESETS = {
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
        "sectors": None,
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
}

# ── Sidebar: filters ──────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.header("Strategy Preset")
    preset_name = st.selectbox("Apply a preset", list(PRESETS.keys()))
    p = adjust_preset_for_regime(PRESETS[preset_name], regime)

    PRESET_DESCRIPTIONS = {
        "No Preset": "Wide-open filters with balanced scoring across all metrics. "
            "Surfaces well-rounded stocks with no glaring weaknesses.",
        "Value Hunting": "Finds cheap, profitable stocks the market may be underpricing. "
            "Prioritizes low P/E, low P/B, and dividend yield. Best when you want "
            "to buy quality names at a discount and wait for the market to catch up.",
        "Momentum / Growth": "Finds stocks in strong uptrends with growing revenue. "
            "Prioritizes price near 52-week highs, healthy RSI, and revenue growth. "
            "Best when the market is trending up and you want to ride winners.",
        "Dividend Income": "Finds reliable dividend payers with sustainable earnings. "
            "Prioritizes high yield backed by reasonable P/E and profit margins. "
            "Best for building a portfolio that generates steady cash flow.",
        "Oversold Bounce": "Finds beaten-down stocks showing signs of life. "
            "Prioritizes low RSI, large drawdowns from highs, and unusual volume. "
            "Best for contrarian, short-term tactical trades after sharp selloffs.",
    }
    st.info(PRESET_DESCRIPTIONS.get(preset_name, ""))

    if regime != "normal" and preset_name != "No Preset":
        st.caption(f"⚙ Filters adjusted for **{regime_info['label']}** market")

    st.divider()
    st.header("Fundamental Filters")

    sectors = sorted(df["Sector"].dropna().unique().tolist())
    default_sectors = (
        [s for s in p["sectors"] if s in sectors] if p["sectors"] else sectors
    )
    sel_sectors = st.multiselect("Sector", sectors, default=default_sectors)

    pe_max_bound = float(min(df["P/E"].dropna().max(), 200)) if df["P/E"].notna().any() else 200.0
    pe_range = st.slider(
        "P/E Ratio",
        min_value=0.0,
        max_value=pe_max_bound,
        value=(p["pe"][0], min(p["pe"][1], pe_max_bound)),
        step=1.0,
    )

    mktcap_choices = {
        "Any": (0, None),
        "Mega (>200B)": (200_000_000_000, None),
        "Large (10B–200B)": (10_000_000_000, 200_000_000_000),
        "Mid (2B–10B)": (2_000_000_000, 10_000_000_000),
        "Small (<2B)": (0, 2_000_000_000),
    }
    mktcap_keys = list(mktcap_choices.keys())
    mktcap_label = st.selectbox(
        "Market Cap",
        mktcap_keys,
        index=mktcap_keys.index(p["mktcap"]),
    )
    mktcap_lo, mktcap_hi = mktcap_choices[mktcap_label]

    div_min = st.slider("Min Dividend Yield %", 0.0, 10.0, p["div_min"], 0.1)

    st.divider()
    st.header("Technical Filters")

    rsi_range = st.slider("RSI (14)", 0.0, 100.0, (p["rsi"][0], p["rsi"][1]), 1.0)
    ma_filter = st.radio(
        "Moving Average Filter",
        ALL_MA_OPTIONS,
        index=ALL_MA_OPTIONS.index(p["ma"]),
    )
    vol_spike = st.slider("Min Volume / 20d Avg", 0.0, 5.0, p["vol_spike"], 0.1)
    pct_from_high = st.slider("Max % from 52w High", -80.0, 0.0, p["pct_high"], 1.0)

# ── Apply filters ─────────────────────────────────────────────────────────
filtered = df[df["Sector"].isin(sel_sectors)].copy()

range_filters = {
    "P/E": (pe_range[0], pe_range[1]),
    "Market Cap": (mktcap_lo, mktcap_hi),
    "Div Yield %": (div_min, None),
    "RSI (14)": (rsi_range[0], rsi_range[1]),
    "Vol vs Avg": (vol_spike, None),
    "% from 52w High": (pct_from_high, None),
}
filtered = apply_filters(filtered, range_filters)

if ma_filter == "Above 50-MA":
    filtered = filtered[filtered["Above 50-MA"] == True]
elif ma_filter == "Above 200-MA":
    filtered = filtered[filtered["Above 200-MA"] == True]
elif ma_filter == "Golden Cross (50 > 200)":
    filtered = filtered[
        (filtered["50-day MA"].notna())
        & (filtered["200-day MA"].notna())
        & (filtered["50-day MA"] > filtered["200-day MA"])
    ]

# ── Score ──────────────────────────────────────────────────────────────────
if not filtered.empty:
    filtered["Score"] = compute_score(filtered, preset_name)
    filtered = filtered.sort_values("Score", ascending=False).reset_index(drop=True)

# ── Summary metrics ───────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Stocks Passing", f"{len(filtered):,}")
c2.metric("Avg Score", f"{filtered['Score'].mean():.0f}" if not filtered.empty else "—")
c3.metric("Avg P/E", f"{filtered['P/E'].mean():.1f}" if filtered['P/E'].notna().any() else "—")
c4.metric("Avg RSI", f"{filtered['RSI (14)'].mean():.1f}" if filtered['RSI (14)'].notna().any() else "—")
c5.metric("Avg Div Yield", f"{filtered['Div Yield %'].mean():.2f}%" if filtered['Div Yield %'].notna().any() else "—")

st.divider()

# ── Sector distribution chart ─────────────────────────────────────────────
tab_table, tab_charts, tab_detail = st.tabs(["Screener Results", "Charts", "Stock Detail"])

with tab_table:
    display_cols = [
        "Score", "Ticker", "Name", "Sector", "Price", "Market Cap", "P/E", "Fwd P/E",
        "EPS", "Div Yield %", "P/B", "Revenue Growth %", "Profit Margin %",
        "RSI (14)", "50-day MA", "200-day MA", "Vol vs Avg", "% from 52w High", "Beta",
    ]
    show_cols = [c for c in display_cols if c in filtered.columns]

    display_df = filtered[show_cols].copy()
    display_df["Market Cap"] = (display_df["Market Cap"] / 1e9).round(1)
    display_df = display_df.rename(columns={"Market Cap": "Mkt Cap ($B)"})

    st.dataframe(
        display_df.reset_index(drop=True),
        use_container_width=True,
        height=600,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.0f",
                help="Composite score (0–100) based on the selected strategy preset. Higher = better fit for the strategy.",
            ),
            "Ticker": st.column_config.TextColumn(
                help="Stock ticker symbol.",
            ),
            "Mkt Cap ($B)": st.column_config.NumberColumn(
                format="%.1f",
                help="Market capitalization in billions. Total value of all outstanding shares (price × shares).",
            ),
            "Price": st.column_config.NumberColumn(
                format="$%.2f",
                help="Latest closing price.",
            ),
            "P/E": st.column_config.NumberColumn(
                help="Price-to-Earnings ratio. How much you pay per $1 of earnings. Lower = cheaper. Typical range: 10–30.",
            ),
            "Fwd P/E": st.column_config.NumberColumn(
                help="Forward P/E. Uses analyst earnings estimates for the next 12 months instead of trailing earnings.",
            ),
            "EPS": st.column_config.NumberColumn(
                help="Earnings Per Share. Net income divided by shares outstanding. Higher = more profitable per share.",
            ),
            "Div Yield %": st.column_config.NumberColumn(
                format="%.2f%%",
                help="Annual dividend as a % of stock price. Higher = more cash income. Above 4% may signal risk.",
            ),
            "P/B": st.column_config.NumberColumn(
                help="Price-to-Book ratio. Stock price vs net asset value. Below 1.0 = trading below book value (potentially cheap).",
            ),
            "Revenue Growth %": st.column_config.NumberColumn(
                format="%.1f%%",
                help="Year-over-year revenue growth. Positive = business is growing. Above 20% is strong growth.",
            ),
            "Profit Margin %": st.column_config.NumberColumn(
                format="%.1f%%",
                help="Net income as a % of revenue. Higher = more efficient at turning sales into profit. Above 20% is strong.",
            ),
            "RSI (14)": st.column_config.NumberColumn(
                help="Relative Strength Index (14-day). Measures momentum. Below 30 = oversold (potential buy). Above 70 = overbought (potential sell). 40–60 = neutral.",
            ),
            "50-day MA": st.column_config.NumberColumn(
                help="50-day moving average. Price above this = short-term uptrend. Price below = short-term downtrend.",
            ),
            "200-day MA": st.column_config.NumberColumn(
                help="200-day moving average. Price above this = long-term uptrend. When 50-MA crosses above 200-MA = 'Golden Cross' (bullish).",
            ),
            "Vol vs Avg": st.column_config.NumberColumn(
                help="Today's volume divided by the 20-day average. Above 1.5 = unusual activity. Above 2.0 = significant spike.",
            ),
            "% from 52w High": st.column_config.NumberColumn(
                format="%.1f%%",
                help="How far the stock is from its 52-week high. 0% = at the high. -20% = 20% below the high.",
            ),
            "Beta": st.column_config.NumberColumn(
                help="Volatility relative to the market. 1.0 = moves with the market. Above 1.5 = significantly more volatile. Below 0.8 = defensive.",
            ),
        },
    )
    st.caption(f"Showing {len(filtered)} of {len(df)} stocks · Ranked by {preset_name} score")

with tab_charts:
    col_l, col_r = st.columns(2)

    with col_l:
        sector_counts = filtered["Sector"].value_counts()
        fig_sector = go.Figure(
            go.Bar(x=sector_counts.index, y=sector_counts.values, marker_color="#4F8BF9")
        )
        fig_sector.update_layout(title="Stocks by Sector", xaxis_tickangle=-45, height=400, margin=dict(t=40, b=80))
        st.plotly_chart(fig_sector, use_container_width=True)

    with col_r:
        if filtered["RSI (14)"].notna().any():
            fig_rsi = go.Figure(go.Histogram(x=filtered["RSI (14)"], nbinsx=20, marker_color="#F97316"))
            fig_rsi.add_vline(x=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_rsi.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.update_layout(title="RSI Distribution", height=400, margin=dict(t=40, b=40))
            st.plotly_chart(fig_rsi, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        if filtered["P/E"].notna().any():
            fig_pe = go.Figure(go.Box(y=filtered["P/E"], name="P/E", marker_color="#10B981"))
            fig_pe.update_layout(title="P/E Distribution", height=350, margin=dict(t=40, b=20))
            st.plotly_chart(fig_pe, use_container_width=True)

    with col_r2:
        if filtered["Div Yield %"].notna().any():
            top_div = filtered.nlargest(15, "Div Yield %")
            fig_div = go.Figure(
                go.Bar(x=top_div["Ticker"], y=top_div["Div Yield %"], marker_color="#8B5CF6")
            )
            fig_div.update_layout(title="Top 15 Dividend Yields", height=350, margin=dict(t=40, b=40))
            st.plotly_chart(fig_div, use_container_width=True)

with tab_detail:
    selected_ticker = st.selectbox(
        "Pick a stock for detail view",
        filtered["Ticker"].tolist() if not filtered.empty else (selected_tickers or SP500_TICKERS)[:10],
    )

    if selected_ticker:
        tk = yf.Ticker(selected_ticker)
        hist_6m = tk.history(period="6mo")

        if not hist_6m.empty:
            fig_price = go.Figure()
            fig_price.add_trace(
                go.Candlestick(
                    x=hist_6m.index,
                    open=hist_6m["Open"],
                    high=hist_6m["High"],
                    low=hist_6m["Low"],
                    close=hist_6m["Close"],
                    name="Price",
                )
            )
            close_6m = hist_6m["Close"]
            if len(close_6m) >= 50:
                ma50 = close_6m.rolling(50).mean()
                fig_price.add_trace(go.Scatter(x=ma50.index, y=ma50, name="50-MA", line=dict(color="#F59E0B", width=1.5)))
            fig_price.update_layout(
                title=f"{selected_ticker} — 6-Month Candlestick",
                xaxis_rangeslider_visible=False,
                height=450,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_price, use_container_width=True)

        row = filtered[filtered["Ticker"] == selected_ticker]
        if not row.empty:
            r = row.iloc[0]
            mc0, mc1, mc2, mc3, mc4, mc5 = st.columns(6)
            mc0.metric("Score", f"{r['Score']:.0f} / 100" if pd.notna(r.get("Score")) else "—")
            mc1.metric("Price", f"${r['Price']:.2f}")
            mc2.metric("P/E", f"{r['P/E']:.1f}" if pd.notna(r["P/E"]) else "—")
            mc3.metric("RSI", f"{r['RSI (14)']:.0f}" if pd.notna(r["RSI (14)"]) else "—")
            mc4.metric("Div Yield", f"{r['Div Yield %']:.2f}%")
            mc5.metric("Beta", f"{r['Beta']:.2f}" if pd.notna(r["Beta"]) else "—")

            mc6, mc7, mc8, mc9, mc10 = st.columns(5)
            mc6.metric("EPS", f"${r['EPS']:.2f}" if pd.notna(r["EPS"]) else "—")
            mc7.metric("Fwd P/E", f"{r['Fwd P/E']:.1f}" if pd.notna(r["Fwd P/E"]) else "—")
            mc8.metric("P/B", f"{r['P/B']:.2f}" if pd.notna(r["P/B"]) else "—")
            mc9.metric("Rev Growth", f"{r['Revenue Growth %']:.1f}%" if pd.notna(r["Revenue Growth %"]) else "—")
            mc10.metric("Profit Margin", f"{r['Profit Margin %']:.1f}%" if pd.notna(r["Profit Margin %"]) else "—")
