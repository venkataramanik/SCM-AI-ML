# Port Dashboard with resilient scraping + graceful fallbacks
# Order of attempts:
# 1) Port of Los Angeles container stats (with headers)
# 2) Port of Long Beach stats
# 3) User CSV upload (schema-agnostic, will detect common columns)
import streamlit as st
import pandas as pd, numpy as np, requests, io, re
from datetime import datetime

st.set_page_config(page_title="Port Throughput Dashboard", layout="wide")
st.title("Port Throughput Dashboard — Monthly TEUs")

POLA_URL = "https://www.portoflosangeles.org/business/statistics/container-statistics/historical-teu-statistics-2025"
POLB_URL = "https://polb.com/business/port-statistics/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

@st.cache_data(ttl=6*3600, show_spinner=True)
def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text

def read_widest_table_from_html(html: str) -> pd.DataFrame:
    # pandas.read_html can take raw HTML bytes/str
    tables = pd.read_html(io.StringIO(html))
    return max(tables, key=lambda t: t.shape[1]).copy()

def normalize_pola(df: pd.DataFrame) -> pd.DataFrame:
    # POLA tables commonly have columns like Month, Loaded Imports, Loaded Exports, Empty Containers, Total TEUs
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]
    colmap = {}
    def find(name_opts):
        for n in name_opts:
            for c in df.columns:
                if n.lower() == str(c).strip().lower():
                    return c
        return None

    month = find(["Month"])
    imports = find(["Loaded Imports","Imports (Loaded)","Loaded inbound"])
    exports = find(["Loaded Exports","Exports (Loaded)","Loaded outbound"])
    empties = find(["Empty Containers","Empties"])
    total = find(["Total TEUs","Total"])

    keep = [c for c in [month, imports, exports, empties, total] if c]
    df = df[keep].copy()
    ren = {}
    if month: ren[month] = "month"
    if imports: ren[imports] = "imports_loaded"
    if exports: ren[exports] = "exports_loaded"
    if empties: ren[empties] = "empties"
    if total: ren[total] = "total_teus"
    df = df.rename(columns=ren)
    for c in ["imports_loaded","exports_loaded","empties","total_teus"]:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                        .str.replace(",", "", regex=False)
                        .str.extract(r"(-?\d+\.?\d*)", expand=False))
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # parse Month text into a date (handles "Jan 2025" / "January 2025")
    def parse_m(s):
        s = str(s).strip()
        for fmt in ["%b %Y","%B %Y"]:
            try: return pd.to_datetime(s, format=fmt)
            except: pass
        # handle bare month: attach year by forward/back-fill later
        for fmt in ["%b","%B"]:
            try: return pd.to_datetime(s, format=fmt)
            except: pass
        return pd.NaT
    df["month_dt"] = df["month"].apply(parse_m)
    if df["month_dt"].isna().all():
        # fallback: make monthly sequence
        n = len(df)
        seq = pd.date_range(end=pd.Timestamp.utcnow().normalize().replace(day=1), periods=n, freq="MS")
        df["month_dt"] = seq
    df = df.dropna(subset=["month_dt"]).drop_duplicates("month_dt").sort_values("month_dt").reset_index(drop=True)
    if "total_teus" not in df.columns:
        df["total_teus"] = df[["imports_loaded","exports_loaded","empties"]].sum(axis=1, min_count=1)
    return df

def normalize_polb(df: pd.DataFrame) -> pd.DataFrame:
    # POLB "Latest Statistics" page has a tall table with monthly totals; columns vary
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]
    # Try to locate common labels
    # Look for a month column (often "Month" or first column)
    month_guess = "Month" if "Month" in df.columns else df.columns[0]
    ren = {month_guess: "month"}
    # Find total and components by partial matches
    def first_like(pattern):
        for c in df.columns:
            if re.search(pattern, c, flags=re.I):
                return c
        return None
    total = first_like(r"Total(?!.*YTD)") or first_like(r"TEUs$")
    imports = first_like(r"Loaded Inbound|Imports")
    exports = first_like(r"Loaded Outbound|Exports")
    empties_in = first_like(r"Empties Inbound")
    empties_out = first_like(r"Empties Outbound")
    keep = [x for x in [month_guess, imports, exports, empties_in, empties_out, total] if x]
    df = df[keep].copy().rename(columns=ren)
    # Build numeric columns
    numeric_cols = [c for c in df.columns if c != "month"]
    for c in numeric_cols:
        df[c] = (df[c].astype(str)
                    .str.replace(",", "", regex=False)
                    .str.extract(r"(-?\d+\.?\d*)", expand=False))
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # total_teus
    if total and total in df.columns:
        df["total_teus"] = df[total]
    else:
        parts = [c for c in ["Loaded Inbound","Imports","Loaded Outbound","Exports","Empties Inbound","Empties Outbound"] if c in df.columns]
        df["total_teus"] = df[parts].sum(axis=1, min_count=1)
    # parse month
    def parse_m(s):
        s = str(s).strip()
        for fmt in ["%b %Y","%B %Y","%Y-%m","%m/%Y"]:
            try: return pd.to_datetime(s, format=fmt)
            except: pass
        try: return pd.to_datetime(s)  # last resort
        except: return pd.NaT
    df["month_dt"] = df["month"].apply(parse_m)
    df = df.dropna(subset=["month_dt"]).drop_duplicates("month_dt").sort_values("month_dt").reset_index(drop=True)
    return df

def derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mom_total_pct"] = df["total_teus"].pct_change() * 100
    df["yoy_total_pct"] = (df["total_teus"] / df["total_teus"].shift(12) - 1) * 100
    for part in ["imports_loaded","exports_loaded","empties"]:
        if part in df.columns:
            df[f"{part}_share_pct"] = (df[part] / df["total_teus"]) * 100
    return df

# ----------------- Data Source panel -----------------
with st.expander("Data Source", expanded=True):
    st.markdown(f"""
- **Primary target (attempted first):** Port of Los Angeles – Container Statistics (monthly TEUs).  
  - Index page: https://www.portoflosangeles.org/business/statistics/container-statistics  
  - 2025 page used by this app: {POLA_URL}
- **Fallback:** Port of Long Beach – Latest Statistics (monthly TEUs): {POLB_URL}  
- **Why fallback?** Some sites block headless scrapers. We fetch with a browser-like header; if still blocked, we use POLB, which is public and stable.  
- **Last fetch (UTC):** {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}
""")

# ----------------- Methodology panel -----------------
with st.expander("What We Do With The Data (Methodology)", expanded=True):
    st.markdown("""
1) **Fetch HTML with browser headers** → parse the **widest table** on the page.  
2) **Normalize** columns → coerce numerics; parse `"Month"` into a date.  
3) **Derive**: MoM & YoY % change, component mix (imports/exports/empties) when available.  
4) **Visualize**: KPIs, trend lines, YoY bars, mix/volatility.  
5) **Graceful fallback**: if both sources fail, upload a CSV and continue.
""")

# ----------------- Try POLA → POLB → Upload -----------------
source_used = None
df = None
err_msgs = []

try:
    html = fetch_html(POLA_URL)
    df_raw = read_widest_table_from_html(html)
    df = normalize_pola(df_raw)
    source_used = "Port of Los Angeles"
except Exception as e:
    err_msgs.append(f"POLA fetch failed: {e}")
    try:
        html = fetch_html(POLB_URL)
        df_raw = read_widest_table_from_html(html)
        df = normalize_polb(df_raw)
        source_used = "Port of Long Beach"
    except Exception as e2:
        err_msgs.append(f"POLB fetch failed: {e2}")

if df is None or df.empty:
    st.warning("Live fetch failed. You can upload a CSV to proceed (columns like "
               "`Month`, `Total TEUs`, and optionally `Loaded Imports`, `Loaded Exports`, `Empties`).")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        updf = pd.read_csv(up)
        # attempt schema alignment
        cols = {c.lower().strip(): c for c in updf.columns}
        def pick(*names):
            for n in names:
                if n in cols: return cols[n]
            return None
        month_col = pick("month","date")
        total_col = pick("total teus","teus","total")
        df = pd.DataFrame()
        if month_col and total_col:
            df["month"] = updf[month_col]
            df["total_teus"] = updf[total_col]
            for p_guess, std in [("loaded imports","imports_loaded"),
                                 ("imports","imports_loaded"),
                                 ("loaded exports","exports_loaded"),
                                 ("exports","exports_loaded"),
                                 ("empties","empties")]:
                c = pick(p_guess)
                if c: df[std] = updf[c]
            # parse dates
            df["month_dt"] = pd.to_datetime(df["month"], errors="coerce")
            df = df.dropna(subset=["month_dt"]).sort_values("month_dt").reset_index(drop=True)
            source_used = "User CSV"
        else:
            st.error("CSV must include at least Month and Total TEUs.")
            st.stop()
    else:
        st.error("No data available.\n\n" + "\n".join(err_msgs))
        st.stop()

# Derive metrics
df = derive_metrics(df)

st.info(f"**Source in use:** {source_used}")

# ----------------- Dashboards -----------------
tab_overview, tab_trends, tab_yoy, tab_mix, tab_table = st.tabs(
    ["Overview", "Trends", "YoY Change", "Mix & Volatility", "Data Table"]
)

with tab_overview:
    st.subheader("Key Metrics (Latest Month)")
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Month", latest["month_dt"].strftime("%b %Y"))
    c2.metric("Total TEUs", f"{int(latest['total_teus']):,}",
              delta=f"{(latest['mom_total_pct']):.1f}% MoM" if pd.notna(latest["mom_total_pct"]) else None)
    if "imports_loaded" in df.columns:
        c3.metric("Loaded Imports", f"{int(latest['imports_loaded']):,}")
    if "exports_loaded" in df.columns:
        c4.metric("Loaded Exports", f"{int(latest['exports_loaded']):,}")

with tab_trends:
    st.subheader("Total TEUs — Trend")
    st.line_chart(df.set_index("month_dt")[["total_teus"]], height=300)
    comps = [c for c in ["imports_loaded","exports_loaded","empties"] if c in df.columns]
    if comps:
        st.subheader("Components — Imports / Exports / Empties")
        st.line_chart(df.set_index("month_dt")[comps], height=320)

with tab_yoy:
    st.subheader("Year-over-Year % Change — Total TEUs")
    yoy = df.dropna(subset=["yoy_total_pct"]).copy()
    if not yoy.empty:
        st.bar_chart(yoy.set_index("month_dt")[["yoy_total_pct"]], height=320)
    else:
        st.caption("Need ≥ 13 months of data to compute YoY.")

with tab_mix:
    shares = [c for c in ["imports_loaded_share_pct","exports_loaded_share_pct","empties_share_pct"] if c in df.columns]
    if shares:
        mix = df.set_index("month_dt")[shares].copy()
        st.subheader("Mix of TEUs (%)")
        st.area_chart(mix, height=320)
        st.caption("Mix shifts can signal chassis repositioning, yard space, and empties management.")
    st.subheader("3-month Rolling Volatility (Total TEUs)")
    tmp = df[["month_dt","total_teus"]].copy()
    tmp["ret"] = tmp["total_teus"].pct_change()
    tmp["vol_3m"] = tmp["ret"].rolling(3).std() * np.sqrt(12) * 100
    st.line_chart(tmp.set_index("month_dt")[["vol_3m"]], height=280)

with tab_table:
    st.subheader("Data (Raw + Derived)")
    show = df.copy().rename(columns={
        "month_dt":"Month","total_teus":"Total TEUs",
        "mom_total_pct":"MoM Total (%)","yoy_total_pct":"YoY Total (%)",
        "imports_loaded":"Loaded Imports","exports_loaded":"Loaded Exports","empties":"Empties",
        "imports_loaded_share_pct":"Imports Share (%)",
        "exports_loaded_share_pct":"Exports Share (%)",
        "empties_share_pct":"Empties Share (%)",
    })
    fmt = {c:"{:,.0f}" for c in ["Total TEUs","Loaded Imports","Loaded Exports","Empties"] if c in show.columns}
    fmt.update({c:"{:.1f}%" for c in ["MoM Total (%)","YoY Total (%)","Imports Share (%)","Exports Share (%)","Empties Share (%)"] if c in show.columns})
    st.dataframe(show.set_index("Month").style.format(fmt), use_container_width=True)

# ----------------- Analysis & Implications -----------------
st.markdown("---")
st.header("Analysis & Implications")

latest_m = df.iloc[-1]
mom = latest_m["mom_total_pct"]
yoy = latest_m["yoy_total_pct"]
bullets = []

if pd.notna(mom):
    if mom >= 5:
        bullets.append(f"Throughput **up {mom:.1f}% MoM** → plan for near-term increases in drayage moves, yard turns, and overtime shifts.")
    elif mom <= -5:
        bullets.append(f"Throughput **down {mom:.1f}% MoM** → expect temporary easing in yard utilization and chassis demand.")
    else:
        bullets.append(f"Throughput changed **{mom:.1f}% MoM**, within normal variation.")

if pd.notna(yoy):
    if yoy > 0:
        bullets.append(f"YoY **+{yoy:.1f}%** points to sustained demand; capacity and chassis pools may face pressure.")
    else:
        bullets.append(f"YoY **{yoy:.1f}%** suggests softer flow vs. last year; window to reposition empties and schedule maintenance.")

shares_present = all(c in df.columns for c in ["imports_loaded_share_pct","exports_loaded_share_pct","empties_share_pct"])
if shares_present:
    dom = max(
        [("Imports", latest_m["imports_loaded_share_pct"]),
         ("Exports", latest_m["exports_loaded_share_pct"]),
         ("Empties", latest_m["empties_share_pct"])],
        key=lambda x: (x[1] if pd.notna(x[1]) else -1)
    )
    if dom[0] == "Empties" and (dom[1] or 0) > 30:
        bullets.append("High **empties share** → coordinate sweeps and inland repositioning to avoid yard congestion.")
    if dom[0] == "Imports" and (dom[1] or 0) > 60:
        bullets.append("Imports dominate the mix → allocate more inbound drays and cross-dock labor next month.")

bullets.append("Contextualize with **diesel prices** (cost driver) and **rail traffic** (inland flow) for a fuller picture.")
st.markdown("\n".join([f"- {b}" for b in bullets]))
