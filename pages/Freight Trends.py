import os, time, requests, pandas as pd, plotly.express as px, streamlit as st

# -------------------- Page / Theme --------------------
st.set_page_config(page_title="Supply Chain Signals — BTS (Socrata)", layout="wide")

st.markdown("""
<style>
/* Power BI vibe: bold headings, tighter cards, bigger intro */
h1, h2, h3 { font-weight: 700; color:#0f172a; }
.intro { font-size:1.18rem; line-height:1.6; }
.kpi { background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px; }
.kpi h3 { font-size:0.95rem; margin:0 0 6px 0; color:#374151; font-weight:600; }
.kpi .v { font-size:1.4rem; font-weight:700; color:#111827; }
.findings { font-size:0.98rem; color:#374151; background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:10px 12px; }
.caption { color:#6b7280; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("Supply Chain Signals — BTS (Socrata)")

# Big, bold blurb
st.markdown("""
<div class="intro">
<p><strong>What:</strong> Monthly counts of <strong>trucks, rail containers, buses, passengers, and pedestrians</strong> at land ports, plus <strong>container TEUs (loaded vs empty)</strong> and the national <strong>Transportation Services Index (TSI)</strong>.</p>
<p><strong>Where the data comes from:</strong> Bureau of Transportation Statistics (BTS) on the U.S. DOT <strong>Socrata</strong> platform.</p>
<p><strong>How we access it:</strong> Direct <strong>Socrata Open Data API (SoQL)</strong> calls — no manual downloads.</p>
<p><strong>Why it matters:</strong> Actionable signals for <strong>capacity, seasonality, and shocks</strong> in U.S. logistics — perfect for fast analysis and LinkedIn-ready visuals.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Socrata helper --------------------
APP_TOKEN = st.secrets.get("SOCRATA_APP_TOKEN", os.getenv("SOCRATA_APP_TOKEN"))
HDRS = {"User-Agent":"Streamlit-Socrata/1.0"}
if APP_TOKEN: HDRS["X-App-Token"] = APP_TOKEN
PAGE_SIZE, RETRIES, SLEEP = 5000, 3, 0.35

@st.cache_data(show_spinner=False, ttl=3600)
def socrata_fetch(domain, dataset, select=None, where=None, order=None, limit=60000):
    base = f"https://{domain}/resource/{dataset}.json"
    dfs, remaining, offset = [], int(limit), 0
    while remaining > 0:
        size = min(PAGE_SIZE, remaining)
        params = {"$limit": size, "$offset": offset}
        if select: params["$select"] = select
        if where:  params["$where"]  = where
        if order:  params["$order"]  = order
        err = None
        for attempt in range(1, RETRIES+1):
            try:
                r = requests.get(base, headers=HDRS, params=params, timeout=60)
                if r.status_code == 200:
                    js = r.json()
                    if not js: remaining = 0; break
                    dfs.append(pd.DataFrame(js))
                    got = len(js); offset += got; remaining -= got
                    time.sleep(SLEEP); break
                else:
                    raise RuntimeError(f"HTTP {r.status_code}\nURL: {r.url}\nBody: {r.text[:400]}")
            except Exception as e:
                err = e
                if attempt < RETRIES: time.sleep(1.2*attempt)
                else: raise err
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# -------------------- Tabs --------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Border Crossings",
    "Port TEUs (Loaded vs Empty)",
    "Top 25 Ports (TEU Ranking)",
    "Transportation Services Index (TSI)"
])

# === TAB 1: Border Crossings (BTS keg4-3bc2) ===
with tab1:
    st.header("Border Crossings — Trucks & Rail Containers (Monthly, by Port/State)")
    st.caption("Dataset: BTS “Border Crossing/Entry Data” (data.bts.gov id: keg4-3bc2)")

    c1,c2,c3 = st.columns(3)
    with c1: y_min = st.number_input("Min year", 1994, 2050, 2015, step=1)
    with c2: y_max = st.number_input("Max year", 1994, 2050, 2024, step=1)
    with c3: limit1 = st.number_input("Max rows", 10000, 200000, 60000, step=5000)

    measures_all = ["Trucks","Truck Containers Full","Truck Containers Empty","Rail Containers Full","Rail Containers Empty"]
    measures = st.multiselect("Measures", measures_all, default=["Trucks","Rail Containers Full"])

    where = f"date between '{y_min}-01-01T00:00:00.000' and '{y_max}-12-31T23:59:59.999'"
    if measures:
        where += " AND measure in (" + ", ".join([f"'{m}'" for m in measures]) + ")"

    df = socrata_fetch(
        "data.bts.gov","keg4-3bc2",
        select="port_name,state,port_code,border,date,measure,value,latitude,longitude",
        where=where, order="date ASC, port_name ASC, measure ASC", limit=limit1
    )

    if df.empty:
        st.warning("No data returned. Expand years/measures or raise Max rows.")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date","value"]); df["year"] = df["date"].dt.year

        colA,colB = st.columns(2)
        with colA:
            states = sorted([s for s in df["state"].dropna().unique()])
            sel_states = st.multiselect("Filter: States (optional)", states)
        with colB:
            ports = sorted([p for p in df["port_name"].dropna().unique()])
            sel_ports = st.multiselect("Filter: Ports (optional)", ports)

        m = pd.Series(True, index=df.index)
        if sel_states: m &= df["state"].isin(sel_states)
        if sel_ports:  m &= df["port_name"].isin(sel_ports)
        dff = df[m].copy()

        # KPI cards
        latest_year = int(dff["year"].max())
        curr = dff[dff["year"]==latest_year].groupby("measure")["value"].sum()
        prev = dff[dff["year"]==latest_year-1].groupby("measure")["value"].sum()
        k1,k2,k3 = st.columns(3)
        def fmt_delta(v): 
            return (f"{v:+.1f}%" if pd.notna(v) else "n/a")
        with k1:
            delta = (curr.get("Trucks",pd.NA)/prev.get("Trucks",pd.NA)-1)*100 if "Trucks" in curr and "Trucks" in prev else pd.NA
            st.markdown('<div class="kpi"><h3>Trucks — latest year</h3><div class="v">{:,}</div><div class="caption">YoY: {}</div></div>'
                        .format(int(curr.get("Trucks",0)), fmt_delta(delta)), unsafe_allow_html=True)
        with k2:
            delta = (curr.get("Rail Containers Full",pd.NA)/prev.get("Rail Containers Full",pd.NA)-1)*100 if "Rail Containers Full" in curr and "Rail Containers Full" in prev else pd.NA
            st.markdown('<div class="kpi"><h3>Rail Containers (Full)</h3><div class="v">{:,}</div><div class="caption">YoY: {}</div></div>'
                        .format(int(curr.get("Rail Containers Full",0)), fmt_delta(delta)), unsafe_allow_html=True)
        with k3:
            delta = (curr.get("Truck Containers Empty",pd.NA)/prev.get("Truck Containers Empty",pd.NA)-1)*100 if "Truck Containers Empty" in curr and "Truck Containers Empty" in prev else pd.NA
            st.markdown('<div class="kpi"><h3>Truck Containers (Empty)</h3><div class="v">{:,}</div><div class="caption">YoY: {}</div></div>'
                        .format(int(curr.get("Truck Containers Empty",0)), fmt_delta(delta)), unsafe_allow_html=True)

        # Chart 1 — Time series
        st.subheader("Border Crossings Over Time — Monthly Total")
        ts = dff.groupby(["date","measure"], as_index=False)["value"].sum()
        fig1 = px.line(ts, x="date", y="value", color="measure",
                       title="Crossings by Measure (Monthly, summed across selected states/ports)",
                       labels={"value":"Monthly count","date":"Month","measure":"Measure"})
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('<div class="findings"><strong>Analysis & Findings:</strong> Truck volumes show seasonality and cycle with macro conditions. Empty container flows help infer repositioning pressure; sustained high empties indicate equipment imbalance or softened export demand.</div>', unsafe_allow_html=True)

        # Chart 2 — Top ports latest year
        st.subheader(f"Top Ports by Crossings — {latest_year}")
        top = (dff[dff["year"]==latest_year].groupby(["port_name","measure"], as_index=False)["value"].sum()
               .sort_values("value", ascending=False).head(20))
        fig2 = px.bar(top, x="value", y="port_name", color="measure", orientation="h",
                      title=f"Top 20 Ports by Annual Crossings — {latest_year}",
                      labels={"value":"Annual crossings","port_name":"Port"})
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="findings"><strong>Analysis & Findings:</strong> Rank order highlights where capacity and staffing matter most. Shifts in the top 5 often precede carrier routing changes or policy impacts at specific crossings.</div>', unsafe_allow_html=True)

# === TAB 2: Port TEUs (Monthly, Loaded vs Empty) — BTS rd72-aq8r ===
with tab2:
    st.header("Port Containers (TEUs) — Monthly, Loaded vs Empty")
    st.caption("Dataset: BTS “Monthly TEU Data” (data.bts.gov id: rd72-aq8r)")
    c1,c2,c3 = st.columns(3)
    with c1: y_min2 = st.number_input("Min year (TEU)", 2009, 2050, 2018, step=1)
    with c2: y_max2 = st.number_input("Max year (TEU)", 2009, 2050, 2024, step=1)
    with c3: limit2 = st.number_input("Max rows (TEU)", 10000, 200000, 80000, step=5000)

    # Pull everything, then filter; column names vary slightly across revisions
    teu = socrata_fetch("data.bts.gov","rd72-aq8r", order="date ASC", limit=limit2)
    if teu.empty:
        st.warning("No TEU data returned.")
    else:
        # Normalize columns
        cols = {c.lower(): c for c in teu.columns}
        dcol = cols.get("date") or [c for c in teu.columns if "date" in c.lower()][0]
        teu["date"] = pd.to_datetime(teu[dcol], errors="coerce")
        teu["year"] = teu["date"].dt.year
        # try typical fields
        possible_port = [c for c in teu.columns if "port" in c.lower() and "name" in c.lower()]
        port_col = possible_port[0] if possible_port else None
        # value fields (loaded, empty, total)
        value_map = {}
        for c in teu.columns:
            lc = c.lower()
            if "loaded" in lc and "teu" in lc: value_map["loaded_teu"] = c
            if "empty"  in lc and "teu" in lc: value_map["empty_teu"]  = c
            if lc in ("teu","total_teu","total"): value_map["total_teu"] = c
        for k in value_map:
            teu[k] = pd.to_numeric(teu[value_map[k]], errors="coerce")

        # Filter year range
        teuf = teu[(teu["year"]>=y_min2) & (teu["year"]<=y_max2)].copy()
        # Port selector
        ports2 = sorted(teuf[port_col].dropna().unique()) if port_col else []
        sel_ports2 = st.multiselect("Ports (optional)", ports2)
        if sel_ports2:
            teuf = teuf[teuf[port_col].isin(sel_ports2)]

        # KPIs
        k1,k2,k3 = st.columns(3)
        latest2 = teuf["date"].max()
        row_latest = teuf[teuf["date"]==latest2].sum(numeric_only=True)
        prev2 = teuf[teuf["date"]== (latest2 - pd.offsets.MonthBegin(1))].sum(numeric_only=True)
        def kpi_card(title, val, prev):
            delta = None
            if pd.notna(prev) and prev != 0: delta = (val/prev-1)*100
            st.markdown(f'<div class="kpi"><h3>{title}</h3><div class="v">{val:,.0f}</div><div class="caption">MoM: {("" if delta is None else f"{delta:+.1f}%")}</div></div>', unsafe_allow_html=True)
        with k1: kpi_card("Loaded TEUs (latest month)", row_latest.get("loaded_teu",0), prev2.get("loaded_teu",0))
        with k2: kpi_card("Empty TEUs (latest month)",  row_latest.get("empty_teu",0),  prev2.get("empty_teu",0))
        with k3: kpi_card("Total TEUs (latest month)",   row_latest.get("total_teu",0),  prev2.get("total_teu",0))

        # Chart — Loaded vs Empty trend
        st.subheader("Loaded vs Empty TEUs — Trend (Monthly)")
        long = []
        if "loaded_teu" in teuf: 
            t = teuf[["date","loaded_teu"]].copy(); t["type"]="Loaded"; long.append(t.rename(columns={"loaded_teu":"value"}))
        if "empty_teu" in teuf:
            t = teuf[["date","empty_teu"]].copy();  t["type"]="Empty";  long.append(t.rename(columns={"empty_teu":"value"}))
        if long:
            long = pd.concat(long, ignore_index=True)
            fig = px.line(long, x="date", y="value", color="type",
                          title="Monthly TEUs — Loaded vs Empty (selected ports or all)",
                          labels={"value":"TEUs","date":"Month","type":"Container status"})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="findings"><strong>Analysis & Findings:</strong> Rising empties relative to loaded TEUs indicate equipment repositioning and soft export demand; a falling empty share suggests tighter capacity or stronger outbound loads.</div>', unsafe_allow_html=True)
        else:
            st.info("Could not auto-detect loaded/empty columns in this revision of the dataset.")

# === TAB 3: Top 25 Ports by TEU (Annual ranking) — BTS sn74-xpkp ===
with tab3:
    st.header("Top 25 U.S. Container Ports by TEU — Annual Ranking")
    st.caption("Dataset: BTS “Top 25 Container Ports by TEU” (data.bts.gov id: sn74-xpkp)")
    rank = socrata_fetch("data.bts.gov","sn74-xpkp", limit=2000)
    if rank.empty:
        st.warning("No ranking data returned.")
    else:
        # Try to find columns
        cols = {c.lower(): c for c in rank.columns}
        yearc = cols.get("year") or [c for c in rank.columns if "year" in c.lower()][0]
        portc = [c for c in rank.columns if "port" in c.lower() and "name" in c.lower()][0]
        teuc  = [c for c in rank.columns if "teu" in c.lower()][0]
        rank["year"] = pd.to_numeric(rank[yearc], errors="coerce")
        rank["teu"]  = pd.to_numeric(rank[teuc], errors="coerce")
        years = sorted(rank["year"].dropna().unique().astype(int))
        ysel = st.select_slider("Year", options=years, value=years[-1])
        r = rank[rank["year"]==ysel].sort_values("teu", ascending=False).head(25)
        fig = px.bar(r, x="teu", y=portc, orientation="h",
                     title=f"Top 25 Ports by TEU — {int(ysel)}",
                     labels={"teu":"TEUs","y": "Port"})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="findings"><strong>Analysis & Findings:</strong> Rank changes among LA/LB, NY/NJ, Savannah, Houston, and Seattle/Tacoma reveal coastal shifts and carrier network strategies. Pair with TEU empties for imbalance context.</div>', unsafe_allow_html=True)

# === TAB 4: TSI — r8cc-5x95 ===
with tab4:
    st.header("Transportation Services Index (TSI) — National Activity (2000=100)")
    st.caption("Dataset: BTS “Transportation Services Index 2000–Present” (data.bts.gov id: r8cc-5x95)")
    tsi = socrata_fetch("data.bts.gov","r8cc-5x95", order="date ASC", limit=50000)
    if tsi.empty:
        st.warning("No TSI data returned.")
    else:
        # Parse date + detect series
        for c in tsi.columns:
            if c.lower()=="date": tsi["date"]=pd.to_datetime(tsi[c], errors="coerce")
        lower = {c.lower(): c for c in tsi.columns}
        fields = {
            "Combined": lower.get("tsi") or lower.get("combined_tsi"),
            "Freight":  lower.get("freight_tsi") or lower.get("freight"),
            "Passenger":lower.get("passenger_tsi") or lower.get("passenger"),
        }
        series=[]
        for name,col in fields.items():
            if col in tsi.columns:
                t = tsi[["date", col]].copy(); t.columns=["date","value"]; t["series"]=name; series.append(t)
        if series:
            tsil = pd.concat(series, ignore_index=True).dropna()
            fig = px.line(tsil, x="date", y="value", color="series",
                          title="Transportation Services Index — Freight vs Passenger vs Combined",
                          labels={"value":"Index (2000=100)","date":"Month"})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="findings"><strong>Analysis & Findings:</strong> Freight TSI leads goods movement cycles; divergences with Passenger TSI often coincide with energy price swings and consumer demand shifts.</div>', unsafe_allow_html=True)

st.caption("Sources: BTS Border Crossing/Entry Data (keg4-3bc2), Monthly TEU Data (rd72-aq8r), Top 25 Ports by TEU (sn74-xpkp), Transportation Services Index (r8cc-5x95) — via Socrata Open Data API.")
