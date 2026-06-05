import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="Stock Recommendation Dashboard", layout="wide")

# Tickers list provided by user
TICKERS = sorted([
    "AAOI","ALAB","ALM","AMKR","ANET","AOSL","API","APLD","ASTS","BBAI","BE",
    "CIFR","CLFD","CLSK","CORZ","CRDO","CRWV","DLO","EOSE","ERII","FRSH","GFS",
    "GLW","HDSN","HOOD","IBIDY","INDI","INFQ","IONQ","IREN","KRKNF","LAES",
    "LWLG","MP","MRVL","NBIS","NOK","NVT","NVTS","OKLO","ONDS","OUST","PATH",
    "PGY","PL","PLAB","PLTR","POET","QUBT","QUIK","RDDT","RELL","RGTI","RIOT",
    "RKLB","S","SIVEF","SLS","SMR","SOFI","SOUN","SYM","TE","TEM","UEC",
    "ULBI","VIAV"
])

GOOD_PS = 5
BAD_PS = 10
GOOD_PE = 25
BAD_PE = 50

def safe_float(value):
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan

@st.cache_data(ttl=3600)
def get_fx_rate_to_usd(currency):
    if not currency or currency == "USD":
        return 1.0
    try:
        fx = yf.Ticker(f"{currency}USD=X")
        rate = fx.fast_info.get("last_price")
        if rate is None:
            hist = fx.history(period="5d")
            if not hist.empty:
                rate = hist["Close"].iloc[-1]
        return safe_float(rate)
    except Exception:
        return np.nan

def money_fmt(value):
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.0f}"

def calculate_score(row):
    score = 0
    reasons = []

    ps = row["P/S"]
    pe = row["P/E"]
    eps = row["EPS"]
    net_profit = row["USD Net Profit"]
    revenue_growth = row["Revenue Growth %"]
    gross_margin = row["Gross Margin %"]
    fcf = row["Free Cash Flow"]

    if not pd.isna(eps):
        if eps > 0:
            score += 2; reasons.append("positive EPS")
        elif eps < 0:
            score -= 2; reasons.append("negative EPS")

    if not pd.isna(net_profit):
        if net_profit > 0:
            score += 2; reasons.append("profitable")
        elif net_profit < 0:
            score -= 2; reasons.append("net loss")

    if not pd.isna(ps):
        if ps <= GOOD_PS:
            score += 2; reasons.append("attractive P/S")
        elif ps >= BAD_PS:
            score -= 2; reasons.append("expensive P/S")

    if not pd.isna(pe):
        if 0 < pe <= GOOD_PE:
            score += 2; reasons.append("reasonable P/E")
        elif pe >= BAD_PE:
            score -= 2; reasons.append("expensive P/E")

    if not pd.isna(revenue_growth):
        if revenue_growth >= 25:
            score += 2; reasons.append("strong revenue growth")
        elif revenue_growth >= 10:
            score += 1; reasons.append("solid revenue growth")
        elif revenue_growth < 0:
            score -= 1; reasons.append("declining revenue")

    if not pd.isna(gross_margin):
        if gross_margin >= 50:
            score += 1; reasons.append("strong gross margin")
        elif gross_margin < 20:
            score -= 1; reasons.append("weak gross margin")

    if not pd.isna(fcf):
        if fcf > 0:
            score += 1; reasons.append("positive free cash flow")
        elif fcf < 0:
            score -= 1; reasons.append("negative free cash flow")

    if score >= 5:
        recommendation = "Strong Buy"
    elif score >= 2:
        recommendation = "Buy"
    else:
        recommendation = "Not Recommended"

    return pd.Series([recommendation, score, ", ".join(reasons)])

def fetch_single_ticker_data(ticker):
    """Worker function for parallel processing."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        financial_currency = info.get("financialCurrency") or info.get("currency") or "USD"
        fx_rate = get_fx_rate_to_usd(financial_currency)

        current_price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        previous_close = safe_float(info.get("previousClose"))

        today_pct = np.nan
        if not pd.isna(current_price) and not pd.isna(previous_close) and previous_close != 0:
            today_pct = ((current_price - previous_close) / previous_close) * 100

        revenue_raw = safe_float(info.get("totalRevenue"))
        net_profit_raw = safe_float(info.get("netIncomeToCommon"))
        fcf_raw = safe_float(info.get("freeCashflow"))

        revenue_usd = revenue_raw * fx_rate if not pd.isna(revenue_raw) and not pd.isna(fx_rate) else np.nan
        net_profit_usd = net_profit_raw * fx_rate if not pd.isna(net_profit_raw) and not pd.isna(fx_rate) else np.nan
        fcf_usd = fcf_raw * fx_rate if not pd.isna(fcf_raw) and not pd.isna(fx_rate) else np.nan

        market_cap = safe_float(info.get("marketCap"))
        ps_ratio = safe_float(info.get("priceToSalesTrailing12Months"))
        pe_ratio = safe_float(info.get("trailingPE"))
        eps = safe_float(info.get("trailingEps"))

        revenue_growth = safe_float(info.get("revenueGrowth"))
        if not pd.isna(revenue_growth):
            revenue_growth *= 100

        gross_margin = safe_float(info.get("grossMargins"))
        if not pd.isna(gross_margin):
            gross_margin *= 100

        return {
            "Ticker": ticker, "Company": info.get("shortName", ""), "Price": current_price,
            "Today %": today_pct, "Market Cap": market_cap, "USD Revenue": revenue_usd,
            "USD Net Profit": net_profit_usd, "Revenue Growth %": revenue_growth,
            "Gross Margin %": gross_margin, "Free Cash Flow": fcf_usd,
            "P/S": ps_ratio, "P/E": pe_ratio, "EPS": eps
        }
    except Exception:
        return {
            "Ticker": ticker, "Company": "Data unavailable", "Price": np.nan, "Today %": np.nan,
            "Market Cap": np.nan, "USD Revenue": np.nan, "USD Net Profit": np.nan,
            "Revenue Growth %": np.nan, "Gross Margin %": np.nan, "Free Cash Flow": np.nan,
            "P/S": np.nan, "P/E": np.nan, "EPS": np.nan
        }

@st.cache_data(ttl=300)
def load_data():
    st.info("⚡ Fetching market data in parallel threads...")
    
    # Utilizing ThreadPoolExecutor for lightning-fast scraping
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_single_ticker_data, TICKERS))
        
    df = pd.DataFrame(results)
    
    # Calculate scores seamlessly via pandas apply
    df[["Recommendation", "Score", "Reason"]] = df.apply(calculate_score, axis=1)
    
    # Fix rows that completely failed
    df.loc[df["Company"] == "Data unavailable", "Score"] = -999
    df.loc[df["Company"] == "Data unavailable", "Recommendation"] = "Not Recommended"
    df.loc[df["Company"] == "Data unavailable", "Reason"] = "data unavailable"
    
    return df.sort_values(["Recommendation", "Score", "Ticker"], ascending=[True, False, True])

# --- STYLING FUNCTIONS ---
def style_today(val):
    if pd.isna(val): return ""
    return "color: #00cc66; font-weight:bold" if val > 0 else "color: #ff3333; font-weight:bold" if val < 0 else ""

def style_profit(val):
    if pd.isna(val): return ""
    return "color: #00cc66; font-weight:bold" if val > 0 else "color: #ff3333; font-weight:bold" if val < 0 else ""

def style_ps(val):
    if pd.isna(val): return ""
    if val <= GOOD_PS: return "color: #00cc66; font-weight:bold"
    if val >= BAD_PS: return "color: #ff3333; font-weight:bold"
    return "color: #ff9900; font-weight:bold"

def style_pe(val):
    if pd.isna(val): return ""
    if val <= 0 or val >= BAD_PE: return "color: #ff3333; font-weight:bold"
    if val <= GOOD_PE: return "color: #00cc66; font-weight:bold"
    return "color: #ff9900; font-weight:bold"

def style_recommendation(val):
    if val == "Strong Buy": return "background-color: rgba(0, 204, 102, 0.2); color: #00cc66; font-weight:bold"
    if val == "Buy": return "background-color: rgba(255, 153, 0, 0.2); color: #ff9900; font-weight:bold"
    return "background-color: rgba(255, 51, 51, 0.2); color: #ff3333; font-weight:bold"

def format_table(df_to_format):
    return (
        df_to_format.style
        .format({
            "Price": "${:.2f}", "Today %": "{:+.2f}%", "Market Cap": money_fmt,
            "USD Revenue": money_fmt, "USD Net Profit": money_fmt, "Revenue Growth %": "{:+.2f}%",
            "Gross Margin %": "{:.2f}%", "Free Cash Flow": money_fmt, "P/S": "{:.2f}",
            "P/E": "{:.2f}", "EPS": "{:.2f}", "Score": "{:.0f}"
        }, na_rep="N/A")
        .map(style_today, subset=["Today %"])
        .map(style_profit, subset=["USD Net Profit", "EPS", "Free Cash Flow"])
        .map(style_ps, subset=["P/S"])
        .map(style_pe, subset=["P/E"])
        .map(style_recommendation, subset=["Recommendation"])
    )

# --- UI APP RENDER ---
st.title("📊 Financial Fundamentals Dashboard")
st.caption("Rule-based recommendation dashboard using multi-threaded Yahoo Finance lookups.")

if st.button("🔄 Clear App Cache & Refresh Data"):
    st.cache_data.clear()
    st.rerun()

df = load_data()

strong_buy_df = df[df["Recommendation"] == "Strong Buy"].sort_values("Score", ascending=False)
buy_df = df[df["Recommendation"] == "Buy"].sort_values("Score", ascending=False)
not_recommended_df = df[df["Recommendation"] == "Not Recommended"].sort_values("Score", ascending=False)

# Metric layout
col1, col2, col3 = st.columns(3)
col1.metric("🟩 Strong Buy Picks", len(strong_buy_df))
col2.metric("🟨 Buy Picks", len(buy_df))
col3.metric("🟥 Not Recommended", len(not_recommended_df))

tab1, tab2, tab3, tab4 = st.tabs(["Strong Buy", "Buy", "Not Recommended", "All Stocks"])

with tab1:
    st.subheader("🎯 Best Opportunities")
    st.dataframe(format_table(strong_buy_df), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("📈 Solid Options")
    st.dataframe(format_table(buy_df), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("⚠️ Avoid or Short list")
    st.dataframe(format_table(not_recommended_df), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("🌐 Global List")
    st.dataframe(format_table(df.sort_values(["Score", "Ticker"], ascending=[False, True])), use_container_width=True, hide_index=True)

# Data Downloader
csv = df.to_csv(index=False)
st.download_button("📥 Export Analysis to CSV", csv, "stock_recommendations.csv", "text/csv")