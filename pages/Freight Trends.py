# pages/bts_border_trends.py
import os
import time
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="U.S. Land Border Freight — BTS (Socrata)", layout="wide")

# ── What / Where / How / Why (blurb) ────────────────────────────────────────────
st.title("U.S. Land Border Freight — BTS (Socrata)")
st.markdown(
    """
**What:** Monthly counts of trucks, rail containers, buses, passengers, and pedestrians entering the U.S. through land ports.  
**Where the data comes from:** Bureau of Transportation Statistics (BTS) “Border Crossing/Entry Data,” published on the U.S. DOT open-data platform (Socrata).  
**How we access it:** Direct calls to the Socrata Open Data API (SoQL), no manual downloads.  
**Why it matters:** Fast signal on cross-border freight activity (capacity, seasonality, shocks). Great for supply-chain analytics and LinkedIn-friendly charts.
"""
)

# ── Config ──────────────────────────────────────────────────────────────────────
DOMAIN     = "data.bts.gov"
DATASET_ID = "keg4-3bc2"        # Border Crossing/Entry Data
PAGE_SIZE  = 5000
RETRIES    = 3
SLEEP_SEC  = 0.4
APP_TOKEN  = st.secrets.get("SOCRATA_APP_TOKEN", os.getenv("SOCRATA_APP_TOKEN"))

with st.sidebar:
    st.header("Filters")
    year_min = st.number_input("Min year", min_value=1994, max_value=2050, value=2015, step=1)
    year_max = st.number_input("Max year", min_value=1994, max_value=2050, value=2024, step=1)
    default_measures = [
        "Trucks",
        "Truck Containers Full",
        "Truck Containers Empty",
        "Rail Containers Full",
        "Rail Containers Empty",
    ]
    measures = st.multiselect("Measures", default_measures, default=["Trucks"])
    max_rows = st.number_input("Max rows to fetch", min_value=10000, max_value=200000, value=50000, step=5000)
    st.caption("Tip: add SOCRATA_APP_TOKEN to Streamlit secrets for higher limits.")

# ── Data fetch ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True, ttl=3600)
def fetch_bts_border(domain: str,
                     dataset_id: str,
                     year_min: int,
                     year_max: int,
                     measures: list[str],
                     limit: int,
                     app_token: str | None) -> pd.DataFrame:
    base = f"https://{domain}/resource/{dataset_id}.json"
    headers = {"User-Agent": "Streamlit-Socrata/1.0"}
    if app_token:
        headers["X-App-Token"] = app_token

    # Build a safe $where without tricky escaping
    where_clauses = [
        f"date between '{year_min}-01-01T00:00:00.000' and '{year_max}-12-31T23:59:59.999'"
    ]
    if measures:
        # measure values do not contain quotes; single-quote wrap is safe
        in_list = ", ".join([f"'{m}'" for m in measures])
        where_clauses.append(f"measure in ({in_list})")
    where = " AND ".join(where_clauses)

    dfs = []
    remaining = int(limit)
    offset = 0

    while remaining > 0:
        size = min(PAGE_SIZE, remaining)
        params = {
            "$select": "port_name,state,port_code,border,date,measure,value,latitude,longitude",
            "$where": where,
            "$order": "date ASC, port_name ASC, measure ASC",
            "$limit": size,
            "$offset": offset,
        }

        last_err = None
        for attempt in range(1, RETRIES + 1):
            try:
                r = requests.get(base, headers=headers, params=params, timeout=60)
                if r.status_code == 200:
                    js = r.json()
                    if not js:
                        remaining = 0
                        break
                    dfs.append(pd.DataFrame(js))
                    got = len(js)
                    offset += got
                    remaining -= got
                    time.sleep(SLEEP_SEC)
                    break
                else:
                    raise RuntimeError(f"HTTP {r.status_code}\nURL: {r.url}\nBody: {r.text[:500]}")
            except Exception as e:
                last_err = e
                if attempt < RETRIES:
                    time.sleep(1.0 * attempt)
                else:
                    raise last_err

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ── Run fetch and type columns ─────────────────────────────────────────────────
try:
    with st.spinner("Loading BTS Border Crossing/Entry data..."):
        df = fetch_bts_border(DOMAIN, DATASET_ID, year_min, year_max, measures, max_rows, APP_TOKEN)
except Exception as e:
    st.error("Download failed.")
    st.code(str(e))
    st.stop()

if df.empty:
    st.warning("No data returned. Widen the year range, choose more measures, or increase Max rows.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["date", "value"])
df["year"] = df["date"].dt.year

st.success(f"Loaded {len(df):,} rows | Years {int(df['year'].min())}–{int(df['year'].max())}")

# Optional selectors
colA, colB = st.columns(2)
with colA:
    states_all = sorted([s for s in df["state"].dropna().unique()])
    states_sel = st.multiselect("States (optional)", states_all)
with colB:
    ports_all = sorted([p for p in df["port_name"].dropna().unique()])
    ports_sel = st.multiselect("Ports (optional)", ports_all)

mask = pd.Series(True, index=df.index)
if states_sel:
    mask &= df["state"].isin(states_sel)
if ports_sel:
    mask &= df["port_name"].isin(ports_sel)
dfv = df.loc[mask].copy()

# ── Charts ─────────────────────────────────────────────────────────────────────
st.subheader("Time Series — Selected Measures (sum of selected geographies)")
ts = dfv.groupby(["date", "measure"], as_index=False)["value"].sum()
fig = px.line(ts, x="date", y="value", color="measure")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top Ports — Latest Year")
latest = int(dfv["year"].max())
tops = (
    dfv[dfv["year"] == latest]
    .groupby(["port_name", "measure"], as_index=False)["value"].sum()
    .sort_values("value", ascending=False)
    .head(20)
)
fig2 = px.bar(tops, x="value", y="port_name", color="measure", orientation="h")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("State Totals by Measure")
state_tot = dfv.groupby(["year", "state", "measure"], as_index=False)["value"].sum()
fig3 = px.line(state_tot, x="year", y="value", color="state", facet_row="measure", markers=True)
st.plotly_chart(fig3, use_container_width=True)

# ── Export ─────────────────────────────────────────────────────────────────────
st.download_button(
    "Download filtered data (CSV)",
    dfv.to_csv(index=False).encode("utf-8"),
    file_name=f"bts_border_{year_min}_{year_max}.csv",
    mime="text/csv",
)

st.caption("Source: BTS Border Crossing/Entry Data via Socrata Open Data API (data.bts.gov)")
