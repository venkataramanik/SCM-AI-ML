import os
import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="BTS Freight Trends", layout="wide")

st.title("BTS Freight Trends (FAF5)")
st.caption("Pulls live data from Socrata (data.bts.gov) and plots freight by mode.")

# --- Settings ---
DOMAIN = "data.bts.gov"
DATASET_ID = "dv3p-7ye6"   # FAF5 Flow by Mode
DEFAULT_LIMIT = 20000
PAGE_SIZE = 5000
RETRIES = 3
SLEEP_SEC = 0.5
APP_TOKEN = st.secrets.get("SOCRATA_APP_TOKEN", os.getenv("SOCRATA_APP_TOKEN"))

col1, col2, col3 = st.columns(3)
with col1:
    limit = st.number_input("Max rows", value=DEFAULT_LIMIT, min_value=1000, step=5000)
with col2:
    year_min = st.number_input("Min year", value=2015, min_value=1997, max_value=2050)
with col3:
    year_max = st.number_input("Max year", value=2023, min_value=1997, max_value=2050)

where = f"year >= {year_min} AND year <= {year_max}"
select = "year,mode,tons,value_2017_dollars"

@st.cache_data(show_spinner=True, ttl=3600)
def fetch_socrata(domain, dataset_id, limit, select=None, where=None, page_size=PAGE_SIZE, app_token=None):
    base = f"https://{domain}/resource/{dataset_id}.json"
    headers = {"User-Agent": "Streamlit-Socrata-Demo/1.0"}
    if app_token:
        headers["X-App-Token"] = app_token

    dfs = []
    remaining = limit
    offset = 0

    while remaining > 0:
        size = min(page_size, remaining)
        params = {"$limit": size, "$offset": offset}
        if select: params["$select"] = select
        if where:  params["$where"]  = where
        params["$order"] = "year ASC, mode ASC"

        last_exc = None
        for attempt in range(1, RETRIES + 1):
            try:
                r = requests.get(base, headers=headers, params=params, timeout=60)
                if r.status_code == 200:
                    chunk = r.json()
                    if not chunk:
                        remaining = 0
                        break
                    dfs.append(pd.DataFrame(chunk))
                    got = len(chunk)
                    offset += got
                    remaining -= got
                    time.sleep(SLEEP_SEC)
                    break
                else:
                    raise RuntimeError(
                        f"HTTP {r.status_code} from Socrata\nURL: {r.url}\nBody: {r.text[:600]}"
                    )
            except Exception as e:
                last_exc = e
                if attempt < RETRIES:
                    time.sleep(1.5 * attempt)
                else:
                    raise last_exc

    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)

    for c in ["year", "tons", "value_2017_dollars"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

try:
    with st.spinner("Downloading from BTS via Socrata..."):
        df = fetch_socrata(
            DOMAIN, DATASET_ID, limit=limit,
            select=select, where=where, app_token=APP_TOKEN
        )
except Exception as e:
    st.error("Download failed.")
    st.code(str(e))
    st.stop()

if df.empty:
    st.warning("No data returned. Try increasing the year range or limit.")
    st.stop()

st.success(f"Loaded {len(df):,} rows from {int(df['year'].min())}–{int(df['year'].max())}")

st.subheader("Raw sample")
st.dataframe(df.head(20), use_container_width=True)

req_cols = {"year", "mode", "tons", "value_2017_dollars"}
missing = req_cols - set(df.columns)
if missing:
    st.error(f"Dataset missing expected columns: {missing}")
    st.stop()

agg = df.groupby(["year", "mode"], as_index=False).agg(
    tons=("tons", "sum"),
    value_2017_dollars=("value_2017_dollars", "sum")
)

st.subheader("Tonnage by Mode Over Time")
fig1 = px.line(
    agg, x="year", y="tons", color="mode",
    markers=True, title="U.S. Freight Volume by Mode (FAF5)"
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Value (2017 USD) by Mode Over Time")
fig2 = px.line(
    agg, x="year", y="value_2017_dollars", color="mode",
    markers=True, title="U.S. Freight Value by Mode (2017 dollars)"
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Modal Share of Tonnage (Stacked Area)")
share = agg.copy()
totals = share.groupby("year")["tons"].transform("sum")
share["share_pct"] = (share["tons"] / totals) * 100
fig3 = px.area(
    share, x="year", y="share_pct", color="mode",
    groupnorm="percent", title="Modal Share of U.S. Freight (by Tonnage)"
)
st.plotly_chart(fig3, use_container_width=True)

latest = agg["year"].max()
latest_year = agg[agg["year"] == latest].sort_values("tons", ascending=False)
top_row = latest_year.iloc[0]
st.write(f"Top freight mode in {int(latest)}: {top_row['mode']} with {top_row['tons']:.0f} tons.")

st.download_button(
    "Download aggregated CSV",
    agg.to_csv(index=False).encode("utf-8"),
    file_name=f"bts_faf5_{int(agg['year'].min())}_{int(agg['year'].max())}.csv",
    mime="text/csv"
)

st.caption("Source: Bureau of Transportation Statistics (FAF5) via Socrata")
