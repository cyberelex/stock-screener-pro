"""Stock Screener — Streamlit app with fundamental + technical filters,
paper trading, backtesting, and a robo-trader bot."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from screener import (
    SP500_TICKERS, UNIVERSES, fetch_screening_data, apply_filters,
    compute_score, detect_regime, adjust_preset_for_regime,
)
from database import (
    get_or_create_portfolio, get_portfolio, execute_trade,
    get_holdings, get_trade_log, save_snapshot, get_snapshots,
    enrich_holdings_with_prices, reset_portfolio, list_portfolios,
    snapshot_all_portfolios, create_challenge, get_active_challenge,
    is_trading_allowed, cancel_challenge, get_challenge_history,
)
from backtest import run_backtest
from bot import bot_rebalance, get_bot_status, auto_select_strategy

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

if "snapshotted_today" not in st.session_state:
    snapshot_all_portfolios()
    st.session_state["snapshotted_today"] = True

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
            "Prioritizes low P/E, low P/B, and high dividend yield. Best when you want "
            "to buy quality names at a discount and get paid to wait.",
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

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_table, tab_charts, tab_detail, tab_paper, tab_backtest, tab_vs = st.tabs(
    ["Screener Results", "Charts", "Stock Detail",
     "Paper Trading", "Backtest Lab", "You vs Bot"]
)

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

    # ── Quick Buy from screener ───────────────────────────────────────
    if not filtered.empty:
        st.divider()
        trading_ok_scr, trading_msg_scr = is_trading_allowed()

        if not trading_ok_scr:
            st.info(f"Quick buy disabled — {trading_msg_scr}")
        else:
            st.markdown("**Quick Buy**")
            qb1, qb2, qb3, qb4 = st.columns([2, 1, 1, 1])
            with qb1:
                qb_ticker = st.selectbox("Stock", filtered["Ticker"].tolist(), key="qb_ticker")
            with qb2:
                qb_row = filtered.loc[filtered["Ticker"] == qb_ticker]
                qb_price = qb_row["Price"].iloc[0] if not qb_row.empty else 0
                st.metric("Price", f"${qb_price:.2f}")
            with qb3:
                qb_shares = st.number_input("Shares", min_value=0.01, value=10.0, step=1.0, key="qb_shares")
            with qb4:
                st.caption(f"Cost: ${qb_shares * qb_price:,.2f}")
                if st.button("Buy", type="primary", key="qb_buy"):
                    qb_pid = get_or_create_portfolio("My Portfolio", ptype="manual")
                    err = execute_trade(qb_pid, qb_ticker, "buy", qb_shares, qb_price,
                                        reason="Quick buy from screener")
                    if err:
                        st.error(err)
                    else:
                        qb_port = get_portfolio(qb_pid)
                        save_snapshot(qb_pid, qb_port["cash"])
                        st.success(f"Bought {qb_shares:.0f} shares of {qb_ticker} @ ${qb_price:.2f}")
                        st.rerun()

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

# ── Paper Trading tab ─────────────────────────────────────────────────────
with tab_paper:
    st.subheader("Paper Trading")
    st.caption("Practice trading with $100k virtual cash. Picks from your screener results.")

    trading_allowed, trading_msg = is_trading_allowed()
    challenge = get_active_challenge()

    if challenge and challenge["status"] == "locked":
        st.warning(f"Portfolios are **locked**. {trading_msg}")
    elif challenge and challenge["status"] == "active":
        st.success(f"Challenge active: **{challenge['name']}** — {trading_msg}")

    manual_pid = get_or_create_portfolio("My Portfolio", ptype="manual")
    manual_port = get_portfolio(manual_pid)

    pc1, pc2, pc3 = st.columns(3)
    holdings_raw = get_holdings(manual_pid)
    if not holdings_raw.empty:
        holdings_enriched = enrich_holdings_with_prices(holdings_raw)
        market_val = holdings_enriched["Market Value"].sum()
    else:
        holdings_enriched = holdings_raw
        market_val = 0

    total_val = manual_port["cash"] + market_val
    total_pnl = total_val - manual_port["starting_cash"]
    total_pnl_pct = (total_pnl / manual_port["starting_cash"]) * 100

    pc1.metric("Cash", f"${manual_port['cash']:,.2f}")
    pc2.metric("Portfolio Value", f"${total_val:,.2f}")
    pc3.metric("Total P&L", f"${total_pnl:,.2f} ({total_pnl_pct:+.1f}%)")

    st.divider()

    trade_col, holdings_col = st.columns([1, 2])

    with trade_col:
        st.markdown("**Place a Trade**")
        if not trading_allowed:
            st.info("Trading is locked during the hold period. Check the You vs Bot tab for standings.")
        elif filtered.empty:
            st.info("Run the screener first to populate the ticker list.")
            available_tickers = []
        else:
            available_tickers = filtered["Ticker"].tolist()

        if trading_allowed and available_tickers:
            trade_ticker = st.selectbox("Ticker", available_tickers, key="trade_ticker")
            trade_side = st.radio("Side", ["buy", "sell"], horizontal=True, key="trade_side")

            if trade_ticker:
                try:
                    live_price = yf.Ticker(trade_ticker).info.get("currentPrice") or \
                                 yf.Ticker(trade_ticker).history(period="1d")["Close"].iloc[-1]
                except Exception:
                    live_price = filtered.loc[filtered["Ticker"] == trade_ticker, "Price"].iloc[0]
                st.caption(f"Current price: **${live_price:.2f}**")

            trade_shares = st.number_input("Shares", min_value=0.01, value=10.0, step=1.0, key="trade_shares")

            if trade_side == "buy":
                cost_preview = trade_shares * live_price
                st.caption(f"Estimated cost: ${cost_preview:,.2f}")

            if st.button("Execute Trade", type="primary", key="exec_trade"):
                err = execute_trade(manual_pid, trade_ticker, trade_side,
                                    trade_shares, live_price, reason="Manual trade")
                if err:
                    st.error(err)
                else:
                    save_snapshot(manual_pid, total_val)
                    st.success(f"{'Bought' if trade_side == 'buy' else 'Sold'} "
                               f"{trade_shares:.2f} shares of {trade_ticker} @ ${live_price:.2f}")
                    st.rerun()

    with holdings_col:
        st.markdown("**Current Holdings**")
        if holdings_enriched.empty:
            st.info("No holdings yet. Buy some stocks!")
        else:
            st.dataframe(
                holdings_enriched.reset_index(drop=True),
                use_container_width=True,
                height=300,
                column_config={
                    "P&L ($)": st.column_config.NumberColumn(format="$%.2f"),
                    "P&L (%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Market Value": st.column_config.NumberColumn(format="$%.2f"),
                    "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Avg Cost": st.column_config.NumberColumn(format="$%.2f"),
                },
            )

    st.divider()
    st.markdown("**Trade Log**")
    trade_log = get_trade_log(manual_pid)
    if trade_log.empty:
        st.caption("No trades yet.")
    else:
        st.dataframe(trade_log, use_container_width=True, height=300)

    st.divider()
    if st.button("Reset Portfolio", type="secondary", key="reset_manual"):
        reset_portfolio(manual_pid)
        st.success("Portfolio reset to $100k cash. All trades cleared.")
        st.rerun()


# ── Backtest Lab tab ──────────────────────────────────────────────────────
with tab_backtest:
    st.subheader("Backtest Lab")
    st.caption("Replay your strategy against historical data to see how it would have performed.")

    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        bt_preset = st.selectbox("Strategy", list(PRESETS.keys()), key="bt_preset")
    with bc2:
        bt_topn = st.slider("Top N stocks to hold", 3, 30, 10, key="bt_topn")
    with bc3:
        bt_rebal = st.selectbox("Rebalance", ["monthly", "weekly"], key="bt_rebal")
    with bc4:
        bt_years = st.slider("Lookback (years)", 1, 5, 2, key="bt_years")

    bt_tickers = selected_tickers or SP500_TICKERS

    if st.button("Run Backtest", type="primary", key="run_bt"):
        with st.spinner("Running backtest... this may take a few minutes for large universes."):
            bar = st.progress(0, text="Starting backtest...")
            results = run_backtest(
                tickers=bt_tickers,
                preset=bt_preset,
                top_n=bt_topn,
                rebalance_freq=bt_rebal,
                lookback_years=bt_years,
                progress_callback=bar.progress,
            )
            bar.empty()

        if results["equity_curve"].empty:
            st.warning("Not enough data to run the backtest. Try fewer tickers or shorter lookback.")
        else:
            st.session_state["bt_results"] = results

    if "bt_results" in st.session_state:
        results = st.session_state["bt_results"]
        ec = results["equity_curve"]
        stats = results["stats"]

        st.markdown("### Performance Summary")
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Total Return", stats.get("Total Return", "—"))
        sc2.metric("Benchmark (SPY)", stats.get("Benchmark Return", "—"))
        sc3.metric("Max Drawdown", stats.get("Max Drawdown", "—"))
        sc4.metric("Sharpe Ratio", stats.get("Sharpe Ratio", "—"))
        sc5.metric("Final Value", stats.get("Final Value", "—"))

        sc6, sc7, sc8 = st.columns(3)
        sc6.metric("Annualized Return", stats.get("Annualized Return", "—"))
        sc7.metric("Win Rate", stats.get("Win Rate", "—"))
        sc8.metric("Total Trades", stats.get("Total Trades", "—"))

        st.markdown("### Equity Curve")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=ec["Date"], y=ec["Portfolio"], mode="lines",
            name="Your Strategy", line=dict(color="#4F8BF9", width=2),
        ))
        fig_eq.add_trace(go.Scatter(
            x=ec["Date"], y=ec["Benchmark"], mode="lines",
            name="SPY Benchmark", line=dict(color="#9CA3AF", width=1.5, dash="dash"),
        ))
        fig_eq.update_layout(
            height=450, margin=dict(t=20, b=20),
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        st.markdown("### Backtest Trades")
        bt_trades_df = pd.DataFrame(results["trades"])
        if not bt_trades_df.empty:
            st.dataframe(bt_trades_df, use_container_width=True, height=300)
        else:
            st.caption("No trades generated.")


# ── You vs Bot tab ────────────────────────────────────────────────────────
with tab_vs:
    st.subheader("You vs Bot")
    st.caption("The bot runs its own portfolio using the screener's top picks. Compare your manual trades against the algorithm.")

    challenge_vs = get_active_challenge()
    trading_ok, trading_reason = is_trading_allowed()

    # ── Challenge banner ──────────────────────────────────────────────
    if challenge_vs:
        from datetime import date as _date
        ch = challenge_vs
        if ch["status"] == "active":
            days_trade = (_date.fromisoformat(ch["trade_end"]) - _date.today()).days
            days_total = (_date.fromisoformat(ch["challenge_end"]) - _date.today()).days
            st.success(
                f"**{ch['name']}** — Trading window open. "
                f"**{days_trade} day(s)** left to make your picks, "
                f"then portfolios lock for {days_total - days_trade} more days."
            )
        elif ch["status"] == "locked":
            days_left = (_date.fromisoformat(ch["challenge_end"]) - _date.today()).days
            st.warning(
                f"**{ch['name']}** — Portfolios are **locked**. "
                f"**{days_left} day(s)** until the challenge ends. "
                "Just open the app to record daily snapshots."
            )
        elif ch["status"] == "completed":
            if ch.get("winner") == "Cancelled":
                st.info(f"**{ch['name']}** was cancelled.")
            else:
                st.balloons()
                st.success(f"**{ch['name']}** is over! Winner: **{ch.get('winner', '—')}**")

        if ch["status"] in ("active", "locked"):
            if st.button("Cancel Challenge", type="secondary", key="cancel_challenge"):
                cancel_challenge(ch["id"])
                st.success("Challenge cancelled. Free trading restored.")
                st.rerun()

    # ── Start a new challenge ─────────────────────────────────────────
    if not challenge_vs or challenge_vs["status"] == "completed":
        with st.expander("Start a New Challenge", expanded=not bool(challenge_vs)):
            st.markdown(
                "Set a **trading window** to pick your stocks, then a **hold period** "
                "where portfolios lock and the market decides the winner."
            )
            ch_col1, ch_col2, ch_col3, ch_col4 = st.columns(4)
            with ch_col1:
                ch_name = st.text_input("Challenge name", value="Week 1 Challenge", key="ch_name")
            with ch_col2:
                ch_trade_days = st.number_input("Trading window (days)", 1, 14, 5, key="ch_trade_days")
            with ch_col3:
                ch_hold_days = st.number_input("Hold period (days)", 7, 90, 21, key="ch_hold_days")
            with ch_col4:
                ch_cash = st.number_input("Starting cash ($)", 10_000, 1_000_000, 100_000,
                                          step=10_000, key="ch_cash")

            st.caption(
                f"You and the bot each get **${ch_cash:,.0f}**. "
                f"Trade for **{ch_trade_days} days**, then hold for **{ch_hold_days} days**. "
                f"Total challenge: **{ch_trade_days + ch_hold_days} days**."
            )

            if st.button("Start Challenge", type="primary", key="start_challenge"):
                create_challenge(ch_name, ch_trade_days, ch_hold_days, ch_cash)
                st.success(f"Challenge **{ch_name}** started! Both portfolios reset to ${ch_cash:,.0f}. Go make your picks!")
                st.rerun()

    st.divider()

    # ── Bot controls ──────────────────────────────────────────────────
    vs_col1, vs_col2 = st.columns(2)

    strategy_options = ["Auto (Best Fit)"] + list(PRESETS.keys())
    with vs_col1:
        bot_preset_vs = st.selectbox("Bot Strategy", strategy_options,
                                     index=0, key="bot_strategy_vs")
        bot_topn_vs = st.slider("Bot top N holdings", 3, 20, 10, key="bot_topn_vs")

    if bot_preset_vs == "Auto (Best Fit)" and not df.empty:
        auto_result = auto_select_strategy(df)
        st.info(auto_result["reasoning"])
        effective_preset = auto_result["preset"]
    else:
        effective_preset = bot_preset_vs if bot_preset_vs != "Auto (Best Fit)" else "No Preset"

    with vs_col2:
        st.markdown(" ")
        st.markdown(" ")
        if not trading_ok:
            st.info("Bot trading is locked during the hold period.")
        elif st.button("Run Bot Rebalance Now", type="primary", key="run_bot"):
            if df.empty:
                st.warning("Load screener data first.")
            else:
                with st.spinner("Bot is trading..."):
                    result = bot_rebalance(
                        screener_df=df,
                        preset=effective_preset,
                        top_n=bot_topn_vs,
                    )
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(
                        f"Bot chose **{effective_preset}** — "
                        f"{len(result['buys'])} buys, "
                        f"{len(result['sells'])} sells. "
                        f"Portfolio: ${result['portfolio_value']:,.2f}"
                    )
                    st.rerun()

    st.divider()

    # ── Head-to-head comparison ───────────────────────────────────────
    manual_pid_vs = get_or_create_portfolio("My Portfolio", ptype="manual")
    manual_port_vs = get_portfolio(manual_pid_vs)
    bot_status = get_bot_status("Robo Bot")

    manual_holdings = get_holdings(manual_pid_vs)
    if not manual_holdings.empty:
        manual_holdings = enrich_holdings_with_prices(manual_holdings)
        manual_mkt = manual_holdings["Market Value"].sum()
    else:
        manual_mkt = 0
    manual_total = manual_port_vs["cash"] + manual_mkt
    manual_pnl = manual_total - manual_port_vs["starting_cash"]
    manual_pnl_pct = (manual_pnl / manual_port_vs["starting_cash"]) * 100

    bot_total = bot_status["total_value"]
    bot_pnl = bot_status["pnl"]
    bot_pnl_pct = bot_status["pnl_pct"]

    st.markdown("### Head-to-Head")
    h1, h2 = st.columns(2)

    with h1:
        st.markdown("**Your Portfolio**")
        m1a, m1b, m1c = st.columns(3)
        m1a.metric("Value", f"${manual_total:,.0f}")
        m1b.metric("P&L", f"${manual_pnl:,.0f}")
        m1c.metric("Return", f"{manual_pnl_pct:+.1f}%")

        if not manual_holdings.empty:
            st.dataframe(
                manual_holdings[["Ticker", "Shares", "Market Value", "P&L (%)"]].reset_index(drop=True),
                use_container_width=True, height=250,
            )
        else:
            st.caption("No holdings. Use the Paper Trading tab to buy stocks.")

    with h2:
        st.markdown("**Robo Bot Portfolio**")
        m2a, m2b, m2c = st.columns(3)
        m2a.metric("Value", f"${bot_total:,.0f}")
        m2b.metric("P&L", f"${bot_pnl:,.0f}")
        m2c.metric("Return", f"{bot_pnl_pct:+.1f}%")

        bot_holdings = bot_status["holdings"]
        if not bot_holdings.empty:
            display_bot = bot_holdings[["Ticker", "Shares", "Market Value", "P&L (%)"]].reset_index(drop=True) \
                if "Market Value" in bot_holdings.columns else bot_holdings
            st.dataframe(display_bot, use_container_width=True, height=250)
        else:
            st.caption("Bot hasn't traded yet. Click 'Run Bot Rebalance Now' above.")

    st.divider()

    st.markdown("### Performance Over Time")
    manual_snaps = get_snapshots(manual_pid_vs)
    bot_pid_vs = get_or_create_portfolio("Robo Bot", ptype="bot")
    bot_snaps = get_snapshots(bot_pid_vs)

    if not manual_snaps.empty or not bot_snaps.empty:
        fig_vs = go.Figure()
        if not manual_snaps.empty:
            fig_vs.add_trace(go.Scatter(
                x=manual_snaps["Date"], y=manual_snaps["Value"],
                mode="lines+markers", name="You",
                line=dict(color="#4F8BF9", width=2),
            ))
        if not bot_snaps.empty:
            fig_vs.add_trace(go.Scatter(
                x=bot_snaps["Date"], y=bot_snaps["Value"],
                mode="lines+markers", name="Robo Bot",
                line=dict(color="#F97316", width=2),
            ))
        fig_vs.add_hline(y=100_000, line_dash="dot", line_color="gray",
                         annotation_text="Starting Cash")
        fig_vs.update_layout(
            height=400, margin=dict(t=20, b=20),
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_vs, use_container_width=True)
    else:
        st.info("Start trading and run the bot to see the performance comparison chart build up over time.")

    st.divider()
    st.markdown("### Bot Trade Log")
    bot_trades = get_trade_log(bot_pid_vs)
    if not bot_trades.empty:
        st.dataframe(bot_trades, use_container_width=True, height=250)
    else:
        st.caption("No bot trades yet.")

    # ── Challenge history ─────────────────────────────────────────────
    history = get_challenge_history()
    if history:
        st.divider()
        with st.expander("Past Challenges"):
            for ch in history:
                result_label = ch.get("winner", "—")
                st.markdown(
                    f"**{ch['name']}** — "
                    f"Trade: {ch['trade_start']} to {ch['trade_end']}, "
                    f"Hold until: {ch['challenge_end']} — "
                    f"Result: **{result_label}** ({ch['status']})"
                )

    st.divider()
    if st.button("Reset Bot Portfolio", type="secondary", key="reset_bot"):
        reset_portfolio(bot_pid_vs)
        st.success("Bot portfolio reset.")
        st.rerun()
