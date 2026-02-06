
"""
═══════════════════════════════════════════════════════════════════════════════
DXY (US DOLLAR INDEX) - COMPREHENSIVE ANALYTICS DASHBOARD
═══════════════════════════════════════════════════════════════════════════════
The Mountain Path - World of Finance
Prof. V. Ravichandran
28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from config import PAGE_CONFIG, COLORS
from styles import apply_main_styles
from components import HeroHeader, MetricsDisplay, Footer, TabsDisplay, CardDisplay

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=PAGE_CONFIG["page_title"],
    page_icon=PAGE_CONFIG["page_icon"],
    layout=PAGE_CONFIG["layout"],
    initial_sidebar_state=PAGE_CONFIG["initial_sidebar_state"],
)
apply_main_styles()

# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════

HeroHeader.render(
    title="DXY — US DOLLAR INDEX ANALYTICS",
    subtitle="Comprehensive Analysis of the World's Reserve Currency Benchmark",
    description="Real-Time Data | Exploratory Analysis | Correlation & Risk Insights",
    emoji="💵"
)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("---")
    st.markdown("### ⚙️ DATA CONFIGURATION")
    st.markdown("---")

    # Data Source Selection
    data_source = st.selectbox(
        "📡 Data Source",
        ["Yahoo Finance (DX-Y.NYB)", "FRED (Trade-Weighted USD)"],
        help="Yahoo Finance provides DXY futures; FRED provides the Trade-Weighted Broad Dollar Index"
    )

    st.markdown("---")
    st.markdown("### 📅 PERIOD SELECTION")

    # Period selection
    period_mode = st.radio(
        "Select Period Mode",
        ["Preset Period", "Custom Date Range"]
    )

    if period_mode == "Preset Period":
        period = st.selectbox(
            "Lookback Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
            index=3,
            format_func=lambda x: {
                "1mo": "1 Month", "3mo": "3 Months", "6mo": "6 Months",
                "1y": "1 Year", "2y": "2 Years", "5y": "5 Years",
                "10y": "10 Years", "max": "Maximum Available"
            }.get(x, x)
        )
        start_date = None
        end_date = None
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        period = None

    st.markdown("---")
    st.markdown("### 📊 ANALYSIS SETTINGS")

    # Moving Average windows
    ma_short = st.slider("Short MA Window (days)", 5, 50, 20)
    ma_long = st.slider("Long MA Window (days)", 50, 200, 50)

    # Correlation assets
    st.markdown("---")
    st.markdown("### 🔗 CORRELATION ASSETS")
    corr_assets = st.multiselect(
        "Select Assets for Correlation",
        ["GC=F (Gold)", "CL=F (Crude Oil)", "^TNX (US 10Y Yield)",
         "EURUSD=X (EUR/USD)", "GBPUSD=X (GBP/USD)", "JPY=X (USD/JPY)",
         "^GSPC (S&P 500)", "^VIX (Volatility Index)"],
        default=["GC=F (Gold)", "CL=F (Crude Oil)", "^TNX (US 10Y Yield)",
                 "EURUSD=X (EUR/USD)", "^GSPC (S&P 500)"]
    )

    st.markdown("---")
    run_btn = st.button("🚀 EXECUTE ANALYSIS", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; font-size:12px; opacity:0.8;">
        <p>🏔️ <b>The Mountain Path</b></p>
        <p>World of Finance</p>
        <p>Prof. V. Ravichandran</p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_data(period=None, start=None, end=None):
    """Fetch DXY data from Yahoo Finance"""
    ticker = yf.Ticker("DX-Y.NYB")
    if period:
        df = ticker.history(period=period)
    else:
        df = ticker.history(start=start, end=end)
    if df.empty:
        return pd.DataFrame()
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_data(start=None, end=None):
    """Fetch Trade-Weighted USD Index from FRED via pandas_datareader"""
    try:
        import pandas_datareader.data as web
        if start is None:
            start = datetime(2015, 1, 1)
        if end is None:
            end = datetime.now()
        df = web.DataReader("DTWEXBGS", "fred", start, end)
        df.columns = ["Close"]
        df["Open"] = df["Close"]
        df["High"] = df["Close"]
        df["Low"] = df["Close"]
        df["Volume"] = 0
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"FRED fetch failed: {e}. Falling back to Yahoo Finance.")
        return fetch_yahoo_data(period="5y")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_correlation_data(tickers, period=None, start=None, end=None):
    """Fetch multiple assets for correlation analysis"""
    data = {}
    for t in tickers:
        symbol = t.split(" ")[0]
        try:
            tk = yf.Ticker(symbol)
            if period:
                hist = tk.history(period=period)
            else:
                hist = tk.history(start=start, end=end)
            if not hist.empty:
                hist.index = hist.index.tz_localize(None) if hist.index.tz else hist.index
                data[t] = hist["Close"]
        except Exception:
            pass
    return pd.DataFrame(data)


def add_technical_indicators(df, ma_short=20, ma_long=50):
    """Add technical indicators to the dataframe"""
    df = df.copy()
    df["Returns"] = df["Close"].pct_change() * 100
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1)) * 100
    df[f"MA_{ma_short}"] = df["Close"].rolling(window=ma_short).mean()
    df[f"MA_{ma_long}"] = df["Close"].rolling(window=ma_long).mean()
    df["Volatility_20d"] = df["Returns"].rolling(window=20).std() * np.sqrt(252)
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["BB_Upper"], df["BB_Lower"] = compute_bollinger(df["Close"], 20, 2)
    df["MACD"], df["Signal_Line"] = compute_macd(df["Close"])
    df["Daily_Range"] = df["High"] - df["Low"]
    df["Daily_Range_Pct"] = (df["Daily_Range"] / df["Close"]) * 100
    return df


def compute_rsi(series, window=14):
    """Compute Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_bollinger(series, window=20, num_std=2):
    """Compute Bollinger Bands"""
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return ma + num_std * std, ma - num_std * std


def compute_macd(series, fast=12, slow=26, signal=9):
    """Compute MACD and Signal Line"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


# ═══════════════════════════════════════════════════════════════════════════════
# TAB CONTENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def tab_about():
    """Tab 1: About the Project"""
    st.markdown("## 📋 About the DXY Index Analytics Project")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        #### What is the DXY (US Dollar Index)?

        The **U.S. Dollar Index (DXY / USDX)** is a measure of the value of the United States dollar 
        relative to a basket of six major foreign currencies. It was established in **March 1973** by 
        the Intercontinental Exchange (ICE) with a base value of **100.000**.

        The DXY is the most widely watched benchmark for the overall strength of the U.S. dollar 
        in global currency markets. It is used by traders, portfolio managers, central banks, and 
        risk managers worldwide.

        #### Currency Basket Composition & Weights
        """)

        basket_data = pd.DataFrame({
            "Currency": ["Euro (EUR)", "Japanese Yen (JPY)", "British Pound (GBP)",
                         "Canadian Dollar (CAD)", "Swedish Krona (SEK)", "Swiss Franc (CHF)"],
            "Weight (%)": [57.6, 13.6, 11.9, 9.1, 4.2, 3.6],
            "Country/Region": ["Eurozone", "Japan", "United Kingdom",
                               "Canada", "Sweden", "Switzerland"]
        })
        st.dataframe(basket_data, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Key Facts")
        CardDisplay.render_card(
            title="Base Year",
            content="March 1973 = 100.000",
            icon="📅"
        )
        st.markdown("")
        CardDisplay.render_card(
            title="Exchange",
            content="ICE Futures U.S.",
            icon="🏛️"
        )
        st.markdown("")
        CardDisplay.render_card(
            title="Trading Hours",
            content="Sunday 6 PM – Friday 5 PM ET",
            icon="⏰"
        )
        st.markdown("")
        CardDisplay.render_card(
            title="Dominant Weight",
            content="EUR: 57.6% of basket",
            icon="🇪🇺",
            highlight=True
        )

    st.markdown("---")
    st.markdown("#### 🎯 Project Objectives")

    obj_col1, obj_col2, obj_col3 = st.columns(3)
    with obj_col1:
        CardDisplay.render_card(
            title="Data Acquisition",
            content="Fetch real-time & historical DXY data from Yahoo Finance and FRED with flexible date range selection.",
            icon="📡"
        )
    with obj_col2:
        CardDisplay.render_card(
            title="Exploratory Analysis",
            content="Compute descriptive statistics, distribution analysis, trend analysis with moving averages, and volatility profiling.",
            icon="🔍"
        )
    with obj_col3:
        CardDisplay.render_card(
            title="Advanced Analytics",
            content="Correlation matrix with macro assets, MACD/RSI/Bollinger overlays, rolling beta analysis, and regime detection.",
            icon="📊",
            highlight=True
        )

    st.markdown("---")
    st.markdown("""
    #### 📐 Methodology & Data Sources

    This dashboard provides two primary data sources for analysis. **Yahoo Finance** delivers 
    the ICE DXY futures contract (ticker: DX-Y.NYB) with OHLCV data suitable for technical analysis. 
    **FRED** (Federal Reserve Economic Data) provides the Trade-Weighted U.S. Dollar Index: Broad (DTWEXBGS), 
    which covers a wider basket of 26 currencies weighted by trade volumes — a more comprehensive measure 
    of USD purchasing power.

    Technical indicators computed include Simple Moving Averages (SMA), Bollinger Bands (±2σ), 
    Relative Strength Index (RSI-14), and MACD (12-26-9). Risk metrics include annualized volatility, 
    Value at Risk (parametric), maximum drawdown, and Sharpe ratio calculations.

    #### ⚠️ Disclaimer
    *This tool is for educational and analytical purposes only. It does not constitute financial advice. 
    Past performance does not guarantee future results.*
    """)


def tab_data_fetching(df, source_label):
    """Tab 2: Data Fetching & Display"""
    st.markdown("## 📡 Data Fetching & Inspection")
    st.markdown("---")

    if df is None or df.empty:
        st.warning("⚠️ No data available. Please click **EXECUTE ANALYSIS** in the sidebar.")
        return

    # Summary metrics
    MetricsDisplay.render_metrics([
        {"title": "Data Source", "value": source_label, "emoji": "📡", "description": "Active feed"},
        {"title": "Total Records", "value": f"{len(df):,}", "emoji": "📊", "description": "Trading days"},
        {"title": "Date Range", "value": f"{df.index[0].strftime('%d-%b-%Y')}", "emoji": "📅",
         "description": f"to {df.index[-1].strftime('%d-%b-%Y')}"},
        {"title": "Latest Close", "value": f"{df['Close'].iloc[-1]:.2f}", "emoji": "💵",
         "description": f"as of {df.index[-1].strftime('%d-%b-%Y')}", "highlight": True},
        {"title": "Period Span", "value": f"{(df.index[-1] - df.index[0]).days:,} days", "emoji": "📐",
         "description": "Calendar days"},
    ], columns=5, title="DATA OVERVIEW")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔝 First 10 Records")
        st.dataframe(
            df.head(10).style.format({
                "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}",
                "Close": "{:.2f}", "Volume": "{:,.0f}"
            }),
            use_container_width=True
        )

    with col2:
        st.markdown("#### 🔚 Last 10 Records")
        st.dataframe(
            df.tail(10).style.format({
                "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}",
                "Close": "{:.2f}", "Volume": "{:,.0f}"
            }),
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("#### 📋 Data Quality Check")

    quality_col1, quality_col2, quality_col3 = st.columns(3)
    with quality_col1:
        missing = df.isnull().sum()
        st.markdown("**Missing Values per Column:**")
        st.dataframe(missing.to_frame("Missing Count"), use_container_width=True)
    with quality_col2:
        st.markdown("**Data Types:**")
        dtypes = df.dtypes.astype(str).to_frame("Type")
        st.dataframe(dtypes, use_container_width=True)
    with quality_col3:
        st.markdown("**Quick Statistics:**")
        st.dataframe(
            df[["Close"]].describe().style.format("{:.2f}"),
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("#### 📥 Download Raw Data")
    csv = df.to_csv()
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name=f"DXY_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


def tab_eda(df, ma_short, ma_long):
    """Tab 3: Exploratory Data Analysis"""
    st.markdown("## 🔍 Exploratory Data Analysis")
    st.markdown("---")

    if df is None or df.empty:
        st.warning("⚠️ No data available. Please click **EXECUTE ANALYSIS** in the sidebar.")
        return

    # ── Descriptive Statistics ──
    st.markdown("### 📊 Descriptive Statistics")
    desc_stats = df[["Close", "Returns", "Log_Returns", "Daily_Range_Pct", "Volatility_20d"]].describe()
    desc_stats.loc["Skewness"] = df[["Close", "Returns", "Log_Returns", "Daily_Range_Pct", "Volatility_20d"]].skew()
    desc_stats.loc["Kurtosis"] = df[["Close", "Returns", "Log_Returns", "Daily_Range_Pct", "Volatility_20d"]].kurtosis()
    st.dataframe(desc_stats.style.format("{:.4f}"), use_container_width=True)

    st.markdown("---")

    # ── Price Chart with Moving Averages ──
    st.markdown("### 📈 DXY Price Chart with Moving Averages & Bollinger Bands")

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="DXY Close",
        line=dict(color="#003366", width=2)
    ))
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df[f"MA_{ma_short}"], name=f"MA-{ma_short}",
        line=dict(color="#FFD700", width=1.5, dash="dash")
    ))
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df[f"MA_{ma_long}"], name=f"MA-{ma_long}",
        line=dict(color="#e74c3c", width=1.5, dash="dot")
    ))
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df["BB_Upper"], name="BB Upper",
        line=dict(color="rgba(0,51,102,0.3)", width=1), showlegend=True
    ))
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df["BB_Lower"], name="BB Lower",
        line=dict(color="rgba(0,51,102,0.3)", width=1),
        fill="tonexty", fillcolor="rgba(0,51,102,0.05)", showlegend=True
    ))
    fig_price.update_layout(
        template="plotly_white",
        height=500,
        title="DXY Closing Price with Moving Averages & Bollinger Bands",
        xaxis_title="Date", yaxis_title="DXY Level",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("---")

    # ── Candlestick Chart ──
    st.markdown("### 🕯️ OHLC Candlestick Chart (Last 90 Trading Days)")
    df_candle = df.tail(90)
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df_candle.index,
        open=df_candle["Open"], high=df_candle["High"],
        low=df_candle["Low"], close=df_candle["Close"],
        increasing_line_color="#2ecc71", decreasing_line_color="#e74c3c"
    )])
    fig_candle.update_layout(
        template="plotly_white", height=450,
        title="DXY OHLC — Last 90 Trading Days",
        xaxis_rangeslider_visible=False, yaxis_title="DXY Level"
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    st.markdown("---")

    # ── Returns Distribution ──
    st.markdown("### 📉 Returns Distribution Analysis")
    ret_col1, ret_col2 = st.columns(2)

    with ret_col1:
        fig_hist = px.histogram(
            df.dropna(subset=["Returns"]), x="Returns", nbins=80,
            title="Daily Returns Distribution",
            color_discrete_sequence=["#003366"],
            marginal="box"
        )
        fig_hist.update_layout(template="plotly_white", height=400,
                               xaxis_title="Daily Return (%)", yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)

    with ret_col2:
        fig_qq = go.Figure()
        returns_clean = df["Returns"].dropna()
        theoretical_q = np.percentile(
            np.random.normal(returns_clean.mean(), returns_clean.std(), 10000),
            np.linspace(1, 99, len(returns_clean))
        )
        sample_q = np.sort(returns_clean.values)
        if len(theoretical_q) > len(sample_q):
            theoretical_q = theoretical_q[:len(sample_q)]
        elif len(sample_q) > len(theoretical_q):
            sample_q = sample_q[:len(theoretical_q)]

        fig_qq.add_trace(go.Scatter(
            x=theoretical_q, y=sample_q,
            mode="markers", name="Q-Q Plot",
            marker=dict(color="#003366", size=3)
        ))
        min_val = min(theoretical_q.min(), sample_q.min())
        max_val = max(theoretical_q.max(), sample_q.max())
        fig_qq.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Normal Line",
            line=dict(color="#e74c3c", dash="dash")
        ))
        fig_qq.update_layout(
            template="plotly_white", height=400,
            title="Q-Q Plot: Returns vs Normal Distribution",
            xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles"
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    # Normality test
    if len(returns_clean) >= 20:
        jb_stat, jb_pval = stats.jarque_bera(returns_clean)
        shapiro_stat, shapiro_pval = stats.shapiro(returns_clean[:5000]) if len(returns_clean) > 5000 else stats.shapiro(returns_clean)
        norm_col1, norm_col2, norm_col3, norm_col4 = st.columns(4)
        with norm_col1:
            st.metric("Jarque-Bera Stat", f"{jb_stat:.2f}")
        with norm_col2:
            st.metric("JB p-value", f"{jb_pval:.4e}")
        with norm_col3:
            st.metric("Shapiro-Wilk Stat", f"{shapiro_stat:.4f}")
        with norm_col4:
            st.metric("SW p-value", f"{shapiro_pval:.4e}")

        if jb_pval < 0.05:
            st.info("📌 **Jarque-Bera test rejects normality** (p < 0.05) — returns exhibit significant skewness and/or excess kurtosis, consistent with fat-tailed behaviour typical of financial time series.")

    st.markdown("---")

    # ── Volatility Analysis ──
    st.markdown("### 📊 Rolling Volatility (Annualized)")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=df.index, y=df["Volatility_20d"],
        name="20-Day Rolling Vol (Annualized)",
        fill="tozeroy", fillcolor="rgba(0,51,102,0.15)",
        line=dict(color="#003366", width=2)
    ))
    fig_vol.update_layout(
        template="plotly_white", height=400,
        title="20-Day Rolling Annualized Volatility",
        xaxis_title="Date", yaxis_title="Volatility (%)",
        yaxis_ticksuffix="%"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")

    # ── RSI Chart ──
    st.markdown("### 📉 RSI (14-Day)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI-14",
        line=dict(color="#003366", width=2)
    ))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#e74c3c",
                      annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#2ecc71",
                      annotation_text="Oversold (30)")
    fig_rsi.add_hrect(y0=30, y1=70, fillcolor="rgba(0,51,102,0.05)", line_width=0)
    fig_rsi.update_layout(
        template="plotly_white", height=350,
        title="Relative Strength Index (RSI-14)",
        xaxis_title="Date", yaxis_title="RSI",
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig_rsi, use_container_width=True)


def tab_advanced_analysis(df, corr_assets, ma_short, ma_long, period, start_date, end_date):
    """Tab 4: Advanced Analysis — Correlation, MACD, Drawdown, Regime"""
    st.markdown("## 📊 Advanced Analytics & Correlation")
    st.markdown("---")

    if df is None or df.empty:
        st.warning("⚠️ No data available. Please click **EXECUTE ANALYSIS** in the sidebar.")
        return

    # ── Correlation Matrix ──
    st.markdown("### 🔗 Correlation Matrix: DXY vs Macro Assets")

    if corr_assets:
        with st.spinner("Fetching correlation asset data..."):
            corr_df = fetch_correlation_data(
                corr_assets, period=period, start=start_date, end=end_date
            )
            if not corr_df.empty:
                corr_df["DXY"] = df["Close"]
                corr_returns = corr_df.pct_change().dropna()
                corr_matrix = corr_returns.corr()

                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title="Return Correlation Matrix"
                )
                fig_corr.update_layout(height=550, template="plotly_white")
                st.plotly_chart(fig_corr, use_container_width=True)

                # Heatmap interpretation
                dxy_corr = corr_matrix["DXY"].drop("DXY").sort_values()
                st.markdown("#### 📌 DXY Correlation Rankings (by return correlation)")
                corr_rank_df = dxy_corr.to_frame("Correlation with DXY").reset_index()
                corr_rank_df.columns = ["Asset", "Correlation"]
                corr_rank_df = corr_rank_df.sort_values("Correlation", ascending=False)
                st.dataframe(
                    corr_rank_df.style.format({"Correlation": "{:.4f}"}).background_gradient(
                        cmap="RdBu_r", subset=["Correlation"], vmin=-1, vmax=1
                    ),
                    use_container_width=True, hide_index=True
                )
            else:
                st.warning("Could not fetch correlation asset data.")
    else:
        st.info("Select assets in the sidebar to compute the correlation matrix.")

    st.markdown("---")

    # ── MACD Chart ──
    st.markdown("### 📈 MACD (12-26-9)")
    fig_macd = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.05,
        subplot_titles=("DXY Close Price", "MACD & Signal Line")
    )
    fig_macd.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="DXY Close",
        line=dict(color="#003366", width=2)
    ), row=1, col=1)

    fig_macd.add_trace(go.Scatter(
        x=df.index, y=df["MACD"], name="MACD",
        line=dict(color="#003366", width=1.5)
    ), row=2, col=1)
    fig_macd.add_trace(go.Scatter(
        x=df.index, y=df["Signal_Line"], name="Signal Line",
        line=dict(color="#e74c3c", width=1.5, dash="dash")
    ), row=2, col=1)
    histogram = df["MACD"] - df["Signal_Line"]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in histogram]
    fig_macd.add_trace(go.Bar(
        x=df.index, y=histogram, name="MACD Histogram",
        marker_color=colors, opacity=0.6
    ), row=2, col=1)

    fig_macd.update_layout(template="plotly_white", height=600, showlegend=True,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_macd, use_container_width=True)

    st.markdown("---")

    # ── Maximum Drawdown ──
    st.markdown("### 📉 Maximum Drawdown Analysis")
    cummax = df["Close"].cummax()
    drawdown = (df["Close"] - cummax) / cummax * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df.index, y=drawdown,
        fill="tozeroy", fillcolor="rgba(231,76,60,0.2)",
        line=dict(color="#e74c3c", width=1.5),
        name="Drawdown (%)"
    ))
    fig_dd.update_layout(
        template="plotly_white", height=400,
        title="Drawdown from Peak",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        yaxis_ticksuffix="%"
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    dd_col1, dd_col2, dd_col3 = st.columns(3)
    with dd_col1:
        st.metric("Max Drawdown", f"{drawdown.min():.2f}%")
    with dd_col2:
        max_dd_date = drawdown.idxmin()
        st.metric("Max Drawdown Date", max_dd_date.strftime("%d-%b-%Y") if pd.notna(max_dd_date) else "N/A")
    with dd_col3:
        current_dd = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        st.metric("Current Drawdown", f"{current_dd:.2f}%")

    st.markdown("---")

    # ── Monthly Returns Heatmap ──
    st.markdown("### 🗓️ Monthly Returns Heatmap")
    df_monthly = df["Close"].resample("ME").last().pct_change() * 100
    df_monthly = df_monthly.dropna()
    monthly_pivot = pd.DataFrame({
        "Year": df_monthly.index.year,
        "Month": df_monthly.index.month,
        "Return": df_monthly.values
    })
    monthly_heatmap = monthly_pivot.pivot_table(index="Year", columns="Month", values="Return", aggfunc="mean")
    monthly_heatmap.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig_heat = px.imshow(
        monthly_heatmap,
        text_auto=".1f",
        color_continuous_scale="RdBu_r",
        title="Monthly Returns (%)",
        labels=dict(color="Return %")
    )
    fig_heat.update_layout(height=max(300, len(monthly_heatmap) * 35 + 100), template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Risk Metrics Summary ──
    st.markdown("### 🎯 Risk Metrics Summary")
    returns_clean = df["Returns"].dropna()
    annual_return = returns_clean.mean() * 252
    annual_vol = returns_clean.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol != 0 else 0
    var_95 = np.percentile(returns_clean, 5)
    cvar_95 = returns_clean[returns_clean <= var_95].mean()
    max_dd = drawdown.min()

    MetricsDisplay.render_metrics([
        {"title": "Annualized Return", "value": f"{annual_return:.2f}%", "emoji": "📈",
         "description": "Mean daily × 252"},
        {"title": "Annualized Volatility", "value": f"{annual_vol:.2f}%", "emoji": "📊",
         "description": "Std × √252"},
        {"title": "Sharpe Ratio", "value": f"{sharpe:.3f}", "emoji": "⚡",
         "description": "Return / Volatility", "highlight": True},
        {"title": "VaR (95%)", "value": f"{var_95:.3f}%", "emoji": "⚠️",
         "description": "5th percentile daily"},
        {"title": "CVaR (95%)", "value": f"{cvar_95:.3f}%", "emoji": "🔴",
         "description": "Expected Shortfall"},
    ], columns=5, title="KEY RISK METRICS")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize session state
if "dxy_data" not in st.session_state:
    st.session_state.dxy_data = None
    st.session_state.source_label = ""

# Fetch data on button click
if run_btn:
    with st.spinner("🔄 Fetching DXY data..."):
        if "Yahoo" in data_source:
            if period:
                raw = fetch_yahoo_data(period=period)
            else:
                raw = fetch_yahoo_data(start=start_date, end=end_date)
            st.session_state.source_label = "Yahoo Finance (DX-Y.NYB)"
        else:
            if start_date and end_date:
                raw = fetch_fred_data(start=start_date, end=end_date)
            else:
                raw = fetch_fred_data()
            st.session_state.source_label = "FRED (DTWEXBGS)"

        if not raw.empty:
            st.session_state.dxy_data = add_technical_indicators(raw, ma_short, ma_long)
            st.success(f"✅ Loaded {len(st.session_state.dxy_data):,} records from {st.session_state.source_label}")
        else:
            st.error("❌ Failed to fetch data. Please check your connection and try again.")

# Render Tabs
df = st.session_state.dxy_data

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 About the Project",
    "📡 Data Fetching",
    "🔍 EDA Analysis",
    "📊 Advanced Analytics"
])

with tab1:
    tab_about()

with tab2:
    tab_data_fetching(df, st.session_state.source_label)

with tab3:
    if df is not None and not df.empty:
        tab_eda(df, ma_short, ma_long)
    else:
        st.warning("⚠️ No data loaded. Please click **🚀 EXECUTE ANALYSIS** in the sidebar.")

with tab4:
    if df is not None and not df.empty:
        tab_advanced_analysis(df, corr_assets, ma_short, ma_long, period, start_date, end_date)
    else:
        st.warning("⚠️ No data loaded. Please click **🚀 EXECUTE ANALYSIS** in the sidebar.")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

Footer.render(
    title="🏔️ THE MOUNTAIN PATH — WORLD OF FINANCE",
    description="Bridging Academic Theory with Institutional Practice",
    author="© 2026 Prof. V. Ravichandran | 28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence",
    disclaimer="For educational and analytical purposes only. Not financial advice."
)
