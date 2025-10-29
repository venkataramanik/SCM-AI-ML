# pages/supply_chain_bts.py
import os, time, requests, pandas as pd, plotly.express as px, streamlit as st

st.set_page_config(page_title="Supply Chain Signals — BTS (Socrata)", layout="wide")

# ── Big, bold blurb ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.blurb p{margin:0.25rem 0;}
.blurb strong{font-weight:700;}
.blurb{font-size:1.15rem; line-height:1.6;}
h1, h2, h3 { font-weight: 700; }
.caption-small{font-size:0.9rem; color:#6b7280;}
</style>
""", unsafe_allow_html=True)

st.title("Supply Chain Signals — BTS (Socrata)")
st.markdown("""
<div class="blurb">
<p><strong>What:</strong> Monthly counts of <strong>trucks, rail containers, buses, passengers, and pedestrians</strong> entering the U.S. through land ports, plus a national <strong>Transportation Services Index (TSI)</strong> for freight & passenger activity.</p>
<p><strong>Where the data comes from:</strong> Bureau of Transportation Statistics (BTS) open data on the U.S. DOT <strong>Socrata</strong> platform.</p>
<p><strong>How we access it:</strong> Direct <strong>Socrata Open Data API (SoQL)</strong> calls — no manual downloads.</p>
<p><strong>Why it matters:</strong> A fast signal for <strong>capacity, seasonality, and shocks</strong> in U.S. logistics — perfect for quick analytics.</p>
</div>
""", unsafe_allow_html=True)

APP_TOKEN = st.secrets.get("SOCRATA_APP_TOKEN", os.getenv("SOCRATA_APP_TOKEN"))
HDRS = {"User-Agent": "Streamlit-Socrata/1.0", **({"X-App-Token": APP_TOKEN} if APP_TOKEN else {})}
PAGE_SIZE, RETRIES, SLEEP_SEC = 5000, 3, 0.4

@st.cache_data(show_spinner=False, ttl=3600)
def socrata_fetch(domain, dataset, select=None, where=None, order=None, limit=50000):
    base = f"https://{domain}/resource/{dataset}.json"
    dfs, remaining, offset = [], int(limit), 0
    while remaining > 0:
        size = min(PAGE_SIZE, remaining)
        params = {"$limit": size, "$offset": offset}
        if select: params["$select"] = select
        if where:  params["$where"]  = where
        if order:  params["$order"]  = order
        last = None
        for attempt in range(1, RETRIES+1):
            try:
                r = requests.get(base, headers=HDRS, params=params, timeout=60)
                if r.status_code == 200:
                    js = r.json()
                    if not js:
                        remaining = 0; break
                    dfs.append(pd.DataFrame(js))
                    got = len(js); offset += got; remaining -= got
                    time.sleep(SLEEP_SEC); break
                else:
                    raise RuntimeError(f"HTTP {r.status_code}\nURL: {r.url}\nBody: {r.text[:500]}")
            except Exception as e:
                last = e
                if attempt < RETRIES: time.sleep(1.2*attempt)
                else: raise last
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

tab1, tab2 = st.tabs(["Border Crossings (Ports & States)", "Transportation Services Index (TSI)"])

# ── TAB 1: Border Crossings ────────────────────────────────────────────────────
with tab1:
    st.header("U.S. Land Border Crossings — Trucks & Rail Containers (Monthly)")
    st.markdown('<div class="caption-small">Dataset: BTS “Border Crossing/Entry Data” (data.bts.gov, id: <code>keg4-3bc2</code>)</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        y_min = st.number_input("Min year", 1994, 2050, 2015, step=1)
    with c2:
        y_max = st.number_input("Max year", 1994, 2050, 2024, step=1)
    with c3:
        max_rows = st.number_input("Max rows", 10000, 200000, 60000, step=5000)

    default_measures = ["Trucks","Truck Containers Full","Truck Containers Empty","Rail Containers Full","Rail Containers Empty"]
    measures = st.multiselect("Measures", default_measures, default=["Trucks"])

    where = f"date between '{y_min}-01-01T00:00:00.000' and '{y_max}-12-31T23:59:59.999'"
    if measures:
        in_list = ", ".join([f"'{m}'" for m in measures])
        where += f" AND measure in ({in_list})"

    df = socrata_fetch(
        "data.bts.gov", "keg4-3bc2",
        select="port_name,state,port_code,border,date,measure,value,latitude,longitude",
        where=where, order="date ASC, port_name ASC, measure ASC", limit=max_rows
    )
    if df.empty:
        st.warning("No data returned. Widen the year range, choose more measures, or increase Max rows.")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date","value"])
        df["year"] = df["date"].dt.year

        colA, colB = st.columns(2)
        with colA:
            states = sorted([s for s in df["state"].dropna().unique()])
            sel_states = st.multiselect("Filter: States (optional)", states)
        with colB:
            ports = sorted([p for p in df["port_name"].dropna().unique()])
            sel_ports  = st.multiselect("Filter: Ports (optional)", ports)

        mask = pd.Series(True, index=df.index)
        if sel_states: mask &= df["state"].isin(sel_states)
        if sel_ports:  mask &= df["port_name"].isin(sel_ports)
        dff = df.loc[mask].copy()

        st.subheader("Trend of Crossings by Measure")
        ts = dff.groupby(["date","measure"], as_index=False)["value"].sum()
        fig = px.line(
            ts, x="date", y="value", color="measure",
            title="Border Crossings Over Time — Monthly Total (Selected Measures)",
            labels={"value":"Monthly count", "date":"Month"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Top Ports by Crossings — {int(dff['year'].max())}")
        tops = (dff[dff["year"]==dff["year"].max()]
                .groupby(["port_name","measure"], as_index=False)["value"].sum()
                .sort_values("value", ascending=False).head(20))
        fig2 = px.bar(
            tops, x="value", y="port_name", color="measure", orientation="h",
            title=f"Top 20 Ports by Crossings — {int(dff['year'].max())}",
            labels={"value":"Annual crossings"}
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button(
            "Download filtered (CSV)",
            dff.to_csv(index=False).encode("utf-8"),
            file_name=f"bts_border_{y_min}_{y_max}.csv",
            mime="text/csv"
        )

# ── TAB 2: Transportation Services Index (TSI) ─────────────────────────────────
with tab2:
    st.header("Transportation Services Index (TSI) — National Activity (2000–Present)")
    st.markdown('<div class="caption-small">Dataset: BTS “Transportation Services Index 2000–Present” (data.bts.gov, id: <code>r8cc-5x95</code>)</div>', unsafe_allow_html=True)

    # Common TSI columns on BTS: date + combined/freight/passenger measures (names can vary across revisions).
    # We’ll fetch everything, coerce numerics, and auto-detect likely TSI columns.
    tsi = socrata_fetch(
        "data.bts.gov", "r8cc-5x95",
        order="date ASC", limit=50000
    )
    if tsi.empty:
        st.warning("No TSI data returned.")
    else:
        # Parse & coerce
        for c in tsi.columns:
            if c.lower() == "date":
                tsi["date"] = pd.to_datetime(tsi[c], errors="coerce")
            elif c != "date":
                tsi[c] = pd.to_numeric(tsi[c], errors="ignore")

        # Try to find typical TSI fields
        lower = {c.lower(): c for c in tsi.columns}
        fields = {
            "combined": lower.get("tsi") or lower.get("combined_tsi") or lower.get("transportation_services_index"),
            "freight":  lower.get("freight_tsi") or lower.get("freightindex") or lower.get("freight"),
            "passenger":lower.get("passenger_tsi") or lower.get("passengerindex") or lower.get("passenger"),
        }
        # Build a long-form table with whatever exists
        series = []
        for name, col in fields.items():
            if col in tsi.columns:
                tmp = tsi.loc[tsi["date"].notna(), ["date", col]].copy()
                tmp.columns = ["date", "value"]
                tmp["series"] = name.capitalize()
                series.append(tmp)
        if not series:
            st.error("Could not detect TSI value columns automatically. Available columns:")
            st.write(list(tsi.columns))
        else:
            tsil = pd.concat(series, ignore_index=True)
            fig = px.line(
                tsil, x="date", y="value", color="series",
                title="Transportation Services Index — Freight vs Passenger vs Combined",
                labels={"value":"Index (2000=100)", "date":"Month", "series":"Series"}
            )
            st.plotly_chart(fig, use_container_width=True)

            # MoM / YoY quick KPIs (for LinkedIn captions)
            st.subheader("Change Metrics")
            latest = tsil["date"].max()
            base  = tsil[tsil["date"]==latest]
            prev  = tsil[tsil["date"]== (latest - pd.offsets.MonthBegin(1))]
            yearago = tsil[tsil["date"]== (latest - pd.offsets.DateOffset(years=1))]
            def pct(a,b): 
                return ((a - b)/b*100.0) if (b is not None and b != 0) else None
            cols = st.columns(len(base))
            for i,(series_name, row) in enumerate(base.groupby("series")):
                v = row["value"].values[0]
                vm = prev[prev["series"]==series_name]["value"]
                vy = yearago[yearago["series"]==series_name]["value"]
                mom = pct(v, vm.values[0]) if len(vm) else None
                yoy = pct(v, vy.values[0]) if len(vy) else None
                with cols[i]:
                    st.metric(f"{series_name} — latest", f"{v:,.1f}", 
                              delta=(f"{mom:+.2f}% MoM" if mom is not None else None))
                    if yoy is not None:
                        st.caption(f"{yoy:+.2f}% YoY")

        st.download_button(
            "Download TSI (CSV)",
            tsi.to_csv(index=False).encode("utf-8"),
            file_name="bts_tsi.csv",
            mime="text/csv"
        )

st.caption("Sources: BTS Border Crossing/Entry Data (keg4-3bc2) and Transportation Services Index (r8cc-5x95) via Socrata Open Data API.")
