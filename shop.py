from logging import exception
import streamlit as st
import pandas as pd
import requests
import urllib.parse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import plotly.express as px
import matplotlib.pyplot as plt
import requests, urllib, time
from requests.adapters import HTTPAdapter, Retry


# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@st.cache_data(ttl=300)
def get_index_details(category):
    """
    Function that returns constituents and price change / mcap data for indices
    :param category: Index
    :return: Dataframe containing Price Change and Market Cap data for all index constituents
    """

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
        'Upgrade-Insecure-Requests': "1",
        "DNT": "1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*,q=0.8",
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    }
    category = category.upper().replace('&', '%26').replace(' ', '%20')

    try:
        ref_url = "https://www.nseindia.com/market-data/live-equity-market?symbol={category}"
        ref = requests.get(ref_url, headers=headers)
        url = f"https://www.nseindia.com/api/equity-stockIndices?index={category}"
        data = requests.get(url, headers=headers, cookies=ref.cookies.get_dict()).json()
        df = pd.DataFrame(data['data'])
        if not df.empty:
            df = df.drop(["meta"], axis=1)
            df = df.set_index("symbol", drop=True)
            df['ffmc'] = round(df['ffmc']/10000000, 0)
            df = df.iloc[1:].reset_index(drop=False)
        return df
    except Exception as e:
        print("Error Fetching Index Data from NSE. Aborting....")
        return pd.DataFrame()

# Include any additional NSE indices to list below
index_list = ['NIFTY 50', 'NIFTY NEXT 50', 'NIFTY MIDCAP 50', 'NIFTY MIDCAP 100', 'NIFTY MIDCAP 150',
                      'NIFTY SMALLCAP 50',
                      'NIFTY SMALLCAP 100', 'NIFTY SMALLCAP 250', 'NIFTY MIDSMALLCAP 400', 'NIFTY 100', 'NIFTY 200',
                      'NIFTY AUTO',
                      'NIFTY BANK', 'NIFTY ENERGY', 'NIFTY FINANCIAL SERVICES', 'NIFTY FINANCIAL SERVICES 25/50',
                      'NIFTY FMCG',
                      'NIFTY IT', 'NIFTY MEDIA', 'NIFTY METAL', 'NIFTY PHARMA', 'NIFTY PSU BANK', 'NIFTY REALTY',
                      'NIFTY PRIVATE BANK', 'Securities in F&O', 'Permitted to Trade',
                      'NIFTY DIVIDEND OPPORTUNITIES 50',
                      'NIFTY50 VALUE 20', 'NIFTY100 QUALITY 30', 'NIFTY50 EQUAL WEIGHT', 'NIFTY100 EQUAL WEIGHT',
                      'NIFTY100 LOW VOLATILITY 30', 'NIFTY ALPHA 50', 'NIFTY200 QUALITY 30',
                      'NIFTY ALPHA LOW-VOLATILITY 30',
                      'NIFTY200 MOMENTUM 30', 'NIFTY COMMODITIES', 'NIFTY INDIA CONSUMPTION', 'NIFTY CPSE',
                      'NIFTY INFRASTRUCTURE',
                      'NIFTY MNC', 'NIFTY GROWTH SECTORS 15', 'NIFTY PSE', 'NIFTY SERVICES SECTOR',
                      'NIFTY100 LIQUID 15',
                      'NIFTY MIDCAP LIQUID 15']

pd.set_option("display.max_rows", None, "display.max_columns", None)

# Apply fixed screen width for app (1440px)
st.markdown(
    f"""
    <style>
      .stAppViewContainer .stMain .stMainBlockContainer{{ max-width: 1440px; }}
    </style>    
  """,
    unsafe_allow_html=True,
)

# Streamlit App

header1, header2 = st.columns([3,1])
with header1:
# with st.container():
    st.subheader("NSE Indices Heatmap - Visualizer")
    col1, col2, _ = st.columns([2,1,1])
    index_filter = col1.selectbox("Choose Index", index_list, index=0)
    slice_by = col2.selectbox("Slice By", ["Market Cap","Gainers","Losers"], index=0)

with header2:
    df = get_index_details(index_filter)
    advances = df[df['pChange'] > 0].shape[0]
    declines = df[df['pChange'] < 0].shape[0]
    no_change = df[df['pChange'] == 0].shape[0]
    total_count = advances + declines + no_change

    # Plot pie chart

    fig = px.pie(names=['Advances','Declines','No Change'],
                 values=[advances, declines, no_change],
                 color=['Advances','Declines','No Change'],
                 # color_discrete_sequence=['#2ecc71', '#e74c3c', '#95a5a6'])
                 color_discrete_sequence=['#3AA864', '#F38039', '#F2F2F2'])
    fig.update_traces(hole=0.7)
    fig.update_traces(textinfo='none')
    fig.update_layout(
        width=200,  # width in pixels
        height=200,  # height in pixels
        showlegend=False,
        annotations=[dict(
            text=f'{advances}<br>Advances<br>{declines}<br>Declines',  # Line break for style
            x=0.5, y=0.5, font_size=14, showarrow=False
        )]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0)  # left, right, top, bottom
    )
    st.plotly_chart(fig)


if not df.empty:

    if slice_by == 'Market Cap':
        slice_factor = 'ffmc'
        color_scale = ['#ff7a3a', 'white', 'green']
    elif slice_by == 'Gainers':
        slice_factor = 'pChange'
        color_scale = ['white', '#a5eb79']
    elif slice_by == 'Losers':
        df = df[df["pChange"] < 0]
        df['Abs'] = df['pChange'].abs()
        slice_factor = 'Abs'
        color_scale = ['#ff7a3a', 'white']

    st.divider()
    fig = px.treemap(
        df,
        path=['symbol'],
        values=slice_factor,
        color='pChange',
        color_continuous_scale=color_scale,
        custom_data=['pChange', 'lastPrice']
    )
    fig.update_layout(
        margin=dict(t=30, l=0, r=0, b=0),
        width=500, height=1000,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)"
    )
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>LTP: %{customdata[1]:.2f}<br>pChange: %{customdata[0]:.2f}%',
        texttemplate='%{label}<br>%{customdata[1]:.2f}<br>%{customdata[0]:.2f}%',
        textposition='middle center',
        textinfo="label+value"
    )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Failed to fetch data.")

# ==============================
# NSE Shareholdings & SMA 20 Analysis
# ==============================



# -------------------------------
# Assuming df already contains NSE index symbols
# -------------------------------
tickers = df["symbol"].tolist()

# API headers
headers = {
    "Appidkey": os.environ.get("APPIDKEY"),
    "Source": "EDEL",
}

import streamlit as st
import pandas as pd
import requests
import urllib.parse
import time
from requests.adapters import HTTPAdapter, Retry

# -------------------------------
# Helper: API request with retry
# -------------------------------
def fetch_with_retry(url, headers, retries=3, backoff_factor=1):
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    try:
        response = session.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        st.warning(f"Failed to fetch {url}: {e}")
        return None

# -------------------------------
# Initialize Session State
# -------------------------------
for key in ["shareholding_data_fetched", "share_df", "sma_df"]:
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame() if "df" in key else False

# -------------------------------
# Fetch Shareholding Data
# -------------------------------
def fetch_shareholdings(tickers, headers):
    shareholding_qoq_list = []

    for sym in tickers:
        try:
            url = f"https://nw.nuvamawealth.com/app-marketstats/quote-equity-shareholdings?exchangeSymbol={urllib.parse.quote(sym)}"
            response = fetch_with_retry(url, headers)
            if response:
                sh_data = response.json()["data"]["shareholdingsObj"]
                quarters = sh_data["quarters"]
                holdings = sh_data["holdings"]

                promoter = [float(holdings[q].get("promoter", 0)) for q in quarters]
                mf = [float(holdings[q].get("mf", 0)) for q in quarters]
                fii = [float(holdings[q].get("fii", 0)) for q in quarters]

                shareholding_qoq_list.append({
                    "Symbol": sym,
                    **{f"Promoter_{q}": v for q, v in zip(['Q1','Q2','Q3','Q4'], promoter[::-1])},
                    **{f"MF_{q}": v for q, v in zip(['Q1','Q2','Q3','Q4'], mf[::-1])},
                    **{f"FII_{q}": v for q, v in zip(['Q1','Q2','Q3','Q4'], fii[::-1])},
                })

            time.sleep(0.2)
        except Exception as e:
            st.error(f"Error fetching shareholding for {sym}: {e}")

    share_df = pd.DataFrame(shareholding_qoq_list)
    return share_df

# -------------------------------
# Generate Strong / Danger / Neutral Stocks
# -------------------------------
def generate_strength_status(df):
    df = df.copy()
    df["Promoter_Change"] = df["Promoter_Q4"] - df["Promoter_Q1"]
    df["MF_Change"] = df["MF_Q4"] - df["MF_Q1"]
    df["FII_Change"] = df["FII_Q4"] - df["FII_Q1"]
    df["Total_Change"] = df["Promoter_Change"] + df["MF_Change"] + df["FII_Change"]
    df["Strength_Status"] = df["Total_Change"].apply(lambda x: "Strong" if x>0 else ("Danger" if x<0 else "Neutral"))
    return df[["Symbol", "Strength_Status"]]

# -------------------------------
# Fetch SMA 20 Data with Strength Status
# -------------------------------
def fetch_sma_data(tickers, headers):
    all_symbols = []
    share_df = st.session_state.share_df.set_index("Symbol")

    for sym in tickers:
        try:
            url = f"https://nw.nuvamawealth.com/app-marketstats/quote-technical?symbol={urllib.parse.quote(sym)}&typ=undefined"
            response = fetch_with_retry(url, headers)
            if response:
                data = response.json()["data"]
                ltp = data["technicalMovingAverage"]["ltp"]

                sma_20 = next(
                    ((float(item["value"]),
                      (1 if item["colour"]=="positive" else -1)*float(item["indicator"].replace("%","")))
                     for item in data["technicalMovingAverage"]["smaData"] if item["period"]=="20"), (None,None)
                )

                # Strength status
                if sym in share_df.index:
                    promoter_change = share_df.at[sym, "Promoter_Q4"] - share_df.at[sym, "Promoter_Q1"]
                    mf_change = share_df.at[sym, "MF_Q4"] - share_df.at[sym, "MF_Q1"]
                    fii_change = share_df.at[sym, "FII_Q4"] - share_df.at[sym, "FII_Q1"]
                    total_change = promoter_change + mf_change + fii_change
                else:
                    total_change = 0

                status = "Strong" if total_change>0 else ("Danger" if total_change<0 else "Neutral")

                row = {
                    "Symbol": sym,
                    "LTP": ltp,
                    "SMA_20_Value": sma_20[0],
                    "SMA_20_Indicator": sma_20[1],
                    "Strength_Status": status,
                    "SMA_Bullish_Score": data["technicalMovingAverage"]["smaBullishScore"],
                    "SMA_Bearish_Score": data["technicalMovingAverage"]["smaBearishScore"],
                    "Bullish_Score": data["technicalIndicators"]["bullishScore"],
                    "Bearish_Score": data["technicalIndicators"]["bearishScore"],
                    "Neutral_Score": data["technicalIndicators"]["neutralScore"],
                    "Pivot": data["pivotValues"]["pivotPoint"],
                    "R1": data["pivotValues"]["firstResistanceR1"],
                    "S1": data["pivotValues"]["firstSupportS1"],
                    "R2": data["pivotValues"]["secondResistanceR2"],
                    "S2": data["pivotValues"]["secondSupportS2"],
                    "R3": data["pivotValues"]["thirdResistanceR3"],
                    "S3": data["pivotValues"]["thirdSupportS3"]
                }

                all_symbols.append(row)

        except Exception as e:
            st.error(f"Error fetching SMA 20 for {sym}: {e}")

    return pd.DataFrame(all_symbols)

# -------------------------------
# Generate Strong / Danger / Neutral Stocks (Symbols Only)
# -------------------------------
def generate_strength_danger_stocks():
    df = st.session_state.share_df.copy()
    
    # Calculate Q4 - Q1 changes
    df["Promoter_Change"] = df["Promoter_Q4"] - df["Promoter_Q1"]
    df["MF_Change"] = df["MF_Q4"] - df["MF_Q1"]
    df["FII_Change"] = df["FII_Q4"] - df["FII_Q1"]
    
    # Combined total change
    df["Total_Change"] = df["Promoter_Change"] + df["MF_Change"] + df["FII_Change"]
    
    # Classify stocks
    df["Strength_Status"] = df["Total_Change"].apply(lambda x: "Strong" if x > 0 else ("Danger" if x < 0 else "Neutral"))
    
    # Separate DataFrames (only Symbol column)
    strong_df = df[df["Strength_Status"] == "Strong"][["Symbol"]]
    danger_df = df[df["Strength_Status"] == "Danger"][["Symbol"]]
    neutral_df = df[df["Strength_Status"] == "Neutral"][["Symbol"]]
    
    return strong_df, danger_df, neutral_df

# -------------------------------
# Display in 4-column Layout
# -------------------------------
if st.session_state.shareholding_data_fetched:
    strong_df, danger_df, neutral_df = generate_strength_danger_stocks()
    
    # Pie Chart
    import matplotlib.pyplot as plt
    counts = [len(strong_df), len(danger_df), len(neutral_df)]
    labels = ["Strong", "Danger", "Neutral"]
    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie(counts, labels=labels, autopct='%1.0f%%', colors=['green','red','gray'])
    ax.axis('equal')
    
    # Split into 4 columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("Strong Stocks")
        st.dataframe(strong_df, use_container_width=True)
    
    with col2:
        st.subheader("Danger Stocks")
        st.dataframe(danger_df, use_container_width=True)
    
    with col3:
        st.subheader("Neutral Stocks")
        st.dataframe(neutral_df, use_container_width=True)
    
    with col4:
        st.subheader("Stock Strength Pie Chart")
        st.pyplot(fig)
        # Totals
        st.write(f"**Totals:** Strong: {len(strong_df)}, Danger: {len(danger_df)}, Neutral: {len(neutral_df)}")

# -------------------------------
# Fetch SMA 20 Data with Strength Status (Stored in Session)
# -------------------------------
def fetch_sma_data(tickers, headers):
    all_symbols = []
    
    share_df = st.session_state.share_df.set_index("Symbol")
    
    for sym in tickers:
        try:
            url = f"https://nw.nuvamawealth.com/app-marketstats/quote-technical?symbol={urllib.parse.quote(sym)}&typ=undefined"
            response = requests.get(url, headers=headers, verify=False)
            if response.status_code == 200:
                data = response.json()["data"]
                ltp = data["technicalMovingAverage"]["ltp"]

                sma_20 = next(
                    ((float(item["value"]),
                      (1 if item["colour"]=="positive" else -1)*float(item["indicator"].replace("%","")) )
                     for item in data["technicalMovingAverage"]["smaData"] if item["period"]=="20"), (None,None)
                )

                # Calculate Strength_Status from Q4-Q1 changes
                if sym in share_df.index:
                    promoter_change = share_df.at[sym, "Promoter_Q4"] - share_df.at[sym, "Promoter_Q1"]
                    mf_change = share_df.at[sym, "MF_Q4"] - share_df.at[sym, "MF_Q1"]
                    fii_change = share_df.at[sym, "FII_Q4"] - share_df.at[sym, "FII_Q1"]
                    total_change = promoter_change + mf_change + fii_change
                else:
                    total_change = 0

                if total_change > 0:
                    status = "Strong"
                elif total_change < 0:
                    status = "Danger"
                else:
                    status = "Neutral"

                row = {
                    "Symbol": sym,
                    "LTP": ltp,
                    "SMA_20_Value": sma_20[0],
                    "SMA_20_Indicator": sma_20[1],
                    "Strength_Status": status,
                    "SMA_Bullish_Score": data["technicalMovingAverage"]["smaBullishScore"],
                    "SMA_Bearish_Score": data["technicalMovingAverage"]["smaBearishScore"],
                    "Bullish_Score": data["technicalIndicators"]["bullishScore"],
                    "Bearish_Score": data["technicalIndicators"]["bearishScore"],
                    "Neutral_Score": data["technicalIndicators"]["neutralScore"],
                    "Pivot": data["pivotValues"]["pivotPoint"],
                    "R1": data["pivotValues"]["firstResistanceR1"],
                    "S1": data["pivotValues"]["firstSupportS1"],
                    "R2": data["pivotValues"]["secondResistanceR2"],
                    "S2": data["pivotValues"]["secondSupportS2"],
                    "R3": data["pivotValues"]["thirdResistanceR3"],
                    "S3": data["pivotValues"]["thirdSupportS3"]
                }

                all_symbols.append(row)

        except Exception as e:
            st.error(f"Error fetching SMA 20 for {sym}: {e}")

    # Save in session state so it persists until next fetch
    st.session_state.sma_df = pd.DataFrame(all_symbols)
    return st.session_state.sma_df

# -------------------------------
# Streamlit Buttons
# -------------------------------
if st.button("Fetch Shareholdings", key="fetch_shareholdings"):
    st.session_state.share_df = fetch_shareholdings(tickers, headers)
    st.session_state.shareholding_data_fetched = True

if st.session_state.shareholding_data_fetched:
    st.subheader("All Symbols - Quarterly Shareholding")
    st.dataframe(st.session_state.share_df)

if st.button("Fetch SMA 20 Technical Data", key="fetch_sma"):
    st.session_state.sma_df = fetch_sma_data(tickers, headers)
    st.subheader("SMA 20 Technical Table with Strength Status")
    st.dataframe(st.session_state.sma_df)
