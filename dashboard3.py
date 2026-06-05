import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="Stock Recommendation Dashboard", layout="wide")

# Setup a robust request session to bypass Yahoo's automated scraping blocks
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
})

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
        fx = yf.Ticker(f"{currency}USD=X", session=session)
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
    rev_growth = row["Revenue Growth %"]
    gross_margin = row["Gross Margin %"]
    fcf = row["Free Cash Flow"]
    
    # 1. Capital Efficiency Check: Quality Margin Profile
    if not pd.isna(gross_margin):
        if gross_margin >= 60: 
            score += 2; reasons.append("Elite Margins (≥60%)")
        elif gross_margin < 25: 
            score -= 2; reasons.append("Subpar Margins (<25%)")

    # 2. Modern Efficiency Metric: Rule of 40 (Growth Rate + FCF Margin)
    if not pd.isna(rev_growth) and not pd.isna(fcf) and not pd.isna(row["USD Revenue"]) and row["USD Revenue"] > 0:
        fcf_margin = (fcf / row["USD Revenue"]) * 100
        rule_of_40 = rev_growth + fcf_margin
        if rule_of_40 >= 40:
            score += 3; reasons.append(f"Rule of 40 Leader ({rule_of_40:.1f}%)")
        elif rule_of_40 < 10:
            score -= 1; reasons.append("Fails Growth-Efficiency Test")

    # 3. Dynamic Relative Value: PEG Ratio Framework
    if not pd.isna(pe) and not pd.isna(rev_growth) and pe > 0 and rev_growth > 0:
        peg = pe / rev_growth
        if peg <= 1.0:
            score += 3; reasons.append(f"Undervalued Growth (PEG: {peg:.2f})")
        elif peg >= 3.0:
            score -= 2; reasons.append(f"Overextended Growth Valuation (PEG: {peg:.2f})")
            
    # 4. Pure Cash Generator Bonus
    if not pd.isna(fcf) and fcf > 0:
        score += 1
        if not pd.isna(net_profit) and net_profit < 0:
            reasons.append("Cash-flow positive despite net accounting losses")

    if score >= 4:
        recommendation = "Strong Buy"
    elif score >= 1:
        recommendation = "Buy"
    else:
        recommendation = "Not Recommended"

    return pd.Series([recommendation, score, ", ".join(reasons)])

def fetch_single_ticker_data(ticker):
    """Worker function optimized for resiliency against scraping blocks."""
    try:
        stock = yf.Ticker(ticker, session=session)
        info = stock.info or {}

        financial_currency = info.get("financialCurrency") or info.get("currency") or "USD"
        fx_rate = get_fx_rate_to_usd(financial_currency)

        # Anti-scraping fallback for pricing arrays
        current_price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        previous_close = safe_float(info.get("previousClose"))
        
        if pd.isna(current_price):
            hist = stock.history(period="2d")
            if not hist.empty:
                current_price = safe_float(hist['Close'].iloc[-1])
                previous_close = safe_float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price

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

        # Edge case check: If we have absolutely no info dict elements, trigger data unavailable row layout
        if not info or pd.isna(market_cap) and pd.isna(revenue_usd):
            raise ValueError("Empty response payload")

        return {
            "Ticker": ticker, "Company": info.get("shortName", ticker), "Price": current_price,
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
    st.info("⚡ Processing financials... Safely executing restricted parallel API pipelines.")
    
    # max_workers limited to 3 to simulate human browsing habits and keep the IP unblocked
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(fetch_single_ticker_data, TICKERS))
        
    df = pd.DataFrame(results)
    
    # Run the advanced recommendation engine across metrics
    df[["Recommendation", "Score", "Reason"]] = df.apply(calculate_score, axis=1)
    
    # Handle hard network failure blocks cleanly
    df.loc[df["Company"] == "Data unavailable", "Score"] = -999
    df.loc[df["Company"] == "Data unavailable", "Recommendation"] = "Not Recommended"
    df.loc[df["Company"] == "Data unavailable", "Reason"] = "Rate limit hit or fundamental profile unavailable on Yahoo"
    
    return df.sort_values(["Recommendation", "Score", "Ticker"], ascending=[True, False, True])

# --- STYLING LOGIC ---
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
    if val == "Strong Buy": return "background-color: rgba(0, 204, 102, 0.15); color: #00cc66; font-weight:bold"
    if val == "Buy": return "background-color: rgba(255, 153, 0, 0.15); color: #ff9900; font-weight:bold"
    return "background-color: rgba(255, 51, 51, 0.15); color: #ff3333; font-weight:bold"

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

# --- USER INTERFACE APP RENDER ---
st.title("📊 Financial Fundamentals + GARP Engine")
st.caption("Rule-based dashboard using structured multi-threaded lookups and a custom browser handshake.")

if st.button("🔄 Clear App Cache & Refresh Data"):
    st.cache_data.clear()
    st.rerun()

df = load_data()

strong_buy_df = df[df["Recommendation"] == "Strong Buy"].sort_values("Score", ascending=False)
buy_df = df[df["Recommendation"] == "Buy"].sort_values("Score", ascending=False)
not_recommended_df = df[df["Recommendation"] == "Not Recommended"].sort_values("Score", ascending=False)

# Metric layout blocks
col1, col2, col3 = st.columns(3)
col1.metric("🟩 Strong Buy Picks", len(strong_buy_df))
col2.metric("🟨 Buy Picks", len(buy_df))
col3.metric("🟥 Not Recommended", len(not_recommended_df))

tab1, tab2, tab3, tab4 = st.tabs(["Strong Buy", "Buy", "Not Recommended", "All Stocks"])

with tab1:
    st.subheader("🎯 High Conviction Alpha Picks")
    st.dataframe(format_table(strong_buy_df), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("📈 Growth & Value Considerations")
    st.dataframe(format_table(buy_df), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("⚠️ Higher Risk / Stretched Multiples")
    st.dataframe(format_table(not_recommended_df), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("🌐 Global Tracking Matrix")
    st.dataframe(format_table(df.sort_values(["Score", "Ticker"], ascending=[False, True])), use_container_width=True, hide_index=True)

# CSV Engine
csv = df.to_csv(index=False)
st.markdown("---")
st.download_button("📥 Export Analysis Matrix to CSV", csv, "garp_recommendation_matrix.csv", "text/csv")