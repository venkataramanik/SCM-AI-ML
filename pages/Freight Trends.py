import os, time, requests, pandas as pd, streamlit as st
import plotly.express as px

st.set_page_config(page_title="U.S. Land Border Freight — BTS (Socrata)", layout="wide")

# ---- Top blurb (what/why/how) ----
st.title("U.S. Land Border Freight — BTS (Socrata)")
st.markdown("""
**What this is:** Monthly counts of trucks, trains, containers, buses, passengers, and pedestrians entering the U.S. through land ports.

**Where the data comes from:** Bureau of Transportation Statistics (BTS) *Border Crossing/Entry Data*, published on the U.S. DOT Socrata platform.

**How we access it:** Directly via the Socrata Open Data API (SoQL) — no manual downloads.

**Why it matters:** It’s a real-time window into cross-border freight activity (capacity, demand, seasonality, shocks). Great for supply-chain trend tracking and storytelling.
""")

# ---- Config ----
DOMAIN      = "data.bts.gov"
DATASET_ID  = "keg4-3bc2"   # Border Crossing/Entry Data
PAGE_SIZE   = 5000
RETRIES     = 3
SLEEP_SEC   = 0.4
APP_TOKEN   = st.secrets.get("SOCRATA_APP_TOKEN", os.getenv("SOCRATA_APP_TOKEN"))

# ---- Controls ----
with st.sidebar:
    st.header("Filters")
    year_min = st.number_input("Min year", 2015, 1994, 2015)
    year_max = st.number_input("Max year", 2050, 1994, 2024)
    measures_default = ["Trucks","Truck Containers Full","Truck Containers Empty",
                        "Rail Containers Full","Rail Containers Empty"]
    measures = st.multiselect("Measures", measures_default, default=["Trucks"])
    max_rows = st.number_input("Max rows", 10000, 200000, 50000, step=5000)
    st.caption("Tip: add SOCRATA_APP_TOKEN in secrets for higher limits.")

# ---- Fetch ----
@st.cache_data(show_spinner=True, ttl=3600)
def fetch_bts(domain, dataset_id, year_min, year_max, measures, limit, app_token=None):
    base = f"https://{domain}/resource/{dataset_id}.json"
    headers = {"User-Agent": "Streamlit-Socrata/1.0"}
    if app_token:
        headers["X-App-Token"] = app_token

    where_parts = [f"date between '{year_min}-01-01T00:00:00.000' and '{year_max}-12-31T23:59:59.999'"]
    if measures:
        # quote single quotes inside measure names if any
        m = ",".join([f"'{m.replace(\"'\",\"\\'\")}'" for m in measures])
        where_parts.append(f"measure in ({m})")
    where = " AND ".join(where_parts)

    dfs, remaining, offset = [], limit, 0
    while remaining > 0:
        size = min(PAGE_SIZE, remaining)
        params = {
            "$select": "port_name,state,port_code,border,date,measure,value,latitude,longitude",
            "$where": where,
            "$order": "date ASC, port_name ASC, measure ASC",
            "$limit": size,
            "$offset": offset,
        }
        last_exc = None
        for attempt in range(1, RETRIES+1):
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
                    raise RuntimeError(f"HTTP {r.status_code}\nURL: {r.url}\nBody: {r.text[:600]}")
            except Exception as e:
                last_exc = e
                if attempt < RETRIES: time.sleep(1.2*attempt)
                else: raise last_exc
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

try:
    with st.spinner("Loading BTS Border Crossing/Entry data..."):
        df = fetch_bts(DOMAIN, DATASET_ID, year_min, year_max, measures, max_rows, APP_TOKEN)
except Exception as e:
    st.error("Download failed.")
    st.code(str(e))
    st.stop()

if df.empty:
    st.warning("No data returned. Try widening years/measures or increasing max rows.")
    st.stop()

# ---- Typing & basics ----
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["date","value"])
df["year"] = df["date"].dt.year

st.success(f"Loaded {len(df):,} rows  |  Years {int(df['year'].min())}–{int(df['year'].max())}")

# Quick selectors for headline chart
colA, colB = st.columns(2)
with colA:
    states = sorted([s for s in df["state"].dropna().unique()])
    state_sel = st.multiselect("States (optional)", states)
with colB:
    ports = sorted([p for p in df["port_name"].dropna().unique()])
    port_sel = st.multiselect("Ports (optional)", ports)

mask = pd.Series(True, index=df.index)
if state_sel:
    mask &= df["state"].isin(state_sel)
if port_sel:
    mask &= df["port_name"].isin(port_sel)
dfv = df[mask].copy()

# ---- Charts ----
st.subheader("Time Series — Selected Measures")
ts = (dfv.groupby(["date","measure"], as_index=False)["value"].sum())
fig = px.line(ts, x="date", y="value", color="measure",
              title="Border crossings over time (sum of selected geographies)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top Ports — Latest Year")
latest = int(dfv["year"].max())
tops = (dfv[dfv["year"]==latest]
        .groupby(["port_name","measure"], as_index=False)["value"].sum())
tops = tops.sort_values("value", ascending=False).head(20)
fig2 = px.bar(tops, x="value", y="port_name", color="measure", orientation="h",
              title=f"Top ports by crossings — {latest}")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("State Totals by Measure")
state_tot = (dfv.groupby(["year","state","measure"], as_index=False)["value"].sum())
fig3 = px.line(state_tot, x="year", y="value", color="state", facet_row="measure",
               title="State-level totals by measure", markers=True)
st.plotly_chart(fig3, use_container_width=True)

# ---- Data export ----
st.download_button("Download filtered data (CSV)", dfv.to_csv(index=False).encode("utf-8"),
                   file_name=f"bts_border_{year_min}_{year_max}.csv", mime="text/csv")

st.caption("Source: BTS Border Crossing/Entry Data via Socrata Open Data API (data.bts.gov).")
