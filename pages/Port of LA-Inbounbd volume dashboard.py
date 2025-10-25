# Port of Los Angeles – Container Statistics (official, no token)
# Sections: Data Source • What We Do • Dashboards • Analysis & Implications
import pandas as pd, numpy as np
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Port of LA – Container Stats", layout="wide")
TITLE = "Port of Los Angeles — Container Statistics Dashboard"
SOURCE_URL = "https://www.portoflosangeles.org/business/statistics/container-statistics"

st.title(TITLE)
st.caption("Official monthly TEUs from the Port of LA (public HTML tables). No API keys required.")

@st.cache_data(ttl=6*3600, show_spinner=True)
def load_polala_stats():
    # Read all tables on the stats page; pick the widest/monthly one
    tables = pd.read_html(SOURCE_URL, flavor="lxml")
    df = max(tables, key=lambda t: t.shape[1]).copy()

    # Try to normalize common structures seen on this page
    # Often columns look like: ["Month", "Loaded Imports", "Loaded Exports", "Empty Containers", "Total TEUs", "YTD", ...]
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]
    # Rename with flexible matching
    def find(colnames, options):
        for o in options:
            if o in colnames: return o
        return None

    cols = df.columns.tolist()
    month_col = find(cols, ["Month", "MONTH"])
    imports_col = find(cols, ["Loaded Imports","Imports (Loaded)","Loaded inbound"])
    exports_col = find(cols, ["Loaded Exports","Exports (Loaded)","Loaded outbound"])
    empties_col = find(cols, ["Empty Containers","Empties"])
    total_col = find(cols, ["Total TEUs","TOTAL (TEUs)","TEUs Total","Total"])

    # Keep only relevant columns available
    keep = [c for c in [month_col, imports_col, exports_col, empties_col, total_col] if c]
    df = df[keep].copy()

    # Rename
    ren = {}
    if month_col: ren[month_col] = "month"
    if imports_col: ren[imports_col] = "imports_loaded"
    if exports_col: ren[exports_col] = "exports_loaded"
    if empties_col: ren[empties_col] = "empties"
    if total_col: ren[total_col] = "total_teus"
    df = df.rename(columns=ren)

    # Coerce numerics (strip commas & notes)
    for c in ["imports_loaded","exports_loaded","empties","total_teus"]:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                    .str.replace(",", "", regex=False)
                    .str.extract(r"(-?\d+\.?\d*)", expand=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build a proper date: "Month YYYY" sometimes appears; otherwise infer.
    # Try to parse month text plus year from context: the page is "latest month" table; we’ll attempt month names.
    # If there's a Month column with names (Jan, February ...), attach a year series by forward/back fill.
    # Heuristic: when month names repeat, it's a multi-year table; detect a Year column if any.
    # If not present, we’ll try to infer with an increasing index within the current year.
    # 1) Try direct parse:
    def parse_month(mtxt):
        try:
            return pd.to_datetime(mtxt, format="%B %Y")  # "January 2025"
        except:
            try:
                return pd.to_datetime(mtxt, format="%b %Y")  # "Jan 2025"
            except:
                try:
                    # if just "January" etc., we’ll attach year later
                    return pd.to_datetime(mtxt, format="%B")
                except:
                    try:
                        return pd.to_datetime(mtxt, format="%b")
                    except:
                        return pd.NaT

    df["month_dt"] = df["month"].apply(parse_month)

    # If yearless, assign sequential months ending at the latest known release around the 15th
    if df["month_dt"].isna().all():
        # Assume rows are chronological; attach a synthetic year by rolling back from now.
        # We’ll just enumerate 12 months ending this month as a fallback.
        today = pd.Timestamp.utcnow().tz_localize(None)
        n = len(df)
        seq = pd.date_range(end=today.replace(day=1), periods=n, freq="MS")
        df["month_dt"] = seq

    # Clean duplicates, drop blank rows
    df = df.dropna(subset=["month_dt"]).drop_duplicates(subset=["month_dt"]).sort_values("month_dt")
    df = df.reset_index(drop=True)

    # Derive metrics
    if "total_teus" not in df.columns:
        # reconstruct if not provided
        df["total_teus"] = df[["imports_loaded","exports_loaded","empties"]].sum(axis=1, min_count=1)

    df["yoy_total_pct"] = (df["total_teus"] / df["total_teus"].shift(12) - 1.0) * 100.0
    df["mom_total_pct"] = (df["total_teus"] / df["total_teus"].shift(1) - 1.0) * 100.0
    for part in ["imports_loaded","exports_loaded","empties"]:
        if part in df.columns:
            df[f"{part}_share_pct"] = (df[part] / df["total_teus"]) * 100.0

    return df

# ---------- Data Source section ----------
with st.expander("Data Source", expanded=True):
    st.markdown(f"""
- **Source:** [Port of Los Angeles — Container Statistics]({SOURCE_URL})  
- **What it is:** Official **monthly** TEU counts by category (imports, exports, empties) with latest month posted around the **15th** of each month.  
- **Access:** Public HTML tables parsed by `pandas.read_html()` (no API key).  
- **Last fetched (UTC):** {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}
""")

# ---------- Methodology section ----------
with st.expander("What We Do With The Data (Methodology)", expanded=True):
    st.markdown("""
1) **Extract** HTML tables with `pandas.read_html()` → select the **widest** table.  
2) **Normalize** headers and coerce numeric fields (strip commas/notes).  
3) **Build time index** (`month_dt`) and **derive metrics**:  
   - **MoM** and **YoY** % change of **Total TEUs**  
   - **Mix shares**: Imports / Exports / Empties as % of total  
4) **Visualize**:  
   - KPIs for latest month  
   - Trend lines for Total + components  
   - YoY bar chart  
   - Mix over time & volatility  
5) **Implications**: Auto-generated bullets to translate signals into **operations & cost** decisions.
""")

# ---------- Load data ----------
try:
    df = load_polala_stats()
except Exception as e:
    st.error(f"Could not load Container Statistics. Error: {e}")
    st.stop()

if df.empty:
    st.warning("Container Statistics loaded but appears empty.")
    st.stop()

# ---------- Dashboards ----------
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
              delta=f"{((latest['total_teus']/prev['total_teus']-1)*100):.1f}% MoM" if prev["total_teus"]>0 else None)
    if "imports_loaded" in df.columns:
        c3.metric("Loaded Imports", f"{int(latest['imports_loaded']):,}")
    if "exports_loaded" in df.columns:
        c4.metric("Loaded Exports", f"{int(latest['exports_loaded']):,}")

with tab_trends:
    st.subheader("Total TEUs — Trend")
    st.line_chart(df.set_index("month_dt")[["total_teus"]], height=300)
    st.caption("Longer-term flow through LA; useful for planning rail/drayage and warehouse labor.")

    comps = [c for c in ["imports_loaded","exports_loaded","empties"] if c in df.columns]
    if comps:
        st.subheader("Components — Imports / Exports / Empties")
        st.line_chart(df.set_index("month_dt")[comps], height=320)

with tab_yoy:
    st.subheader("Year-over-Year % Change — Total TEUs")
    yoy = df.dropna(subset=["yoy_total_pct"]).copy()
    st.bar_chart(yoy.set_index("month_dt")[["yoy_total_pct"]], height=320)
    st.caption("YoY helps separate seasonality from real demand shifts.")

with tab_mix:
    if all(c in df.columns for c in ["imports_loaded_share_pct","exports_loaded_share_pct","empties_share_pct"]):
        mix = df.set_index("month_dt")[["imports_loaded_share_pct","exports_loaded_share_pct","empties_share_pct"]].copy()
        st.subheader("Mix of TEUs (%)")
        st.area_chart(mix, height=320)
        st.caption("Mix shifts can signal chassis repositioning needs, yard space mix, and empties management.")
    # Simple volatility measure on total TEUs
    st.subheader("3-month Rolling Volatility (Total TEUs)")
    tmp = df[["month_dt","total_teus"]].copy()
    tmp["returns"] = tmp["total_teus"].pct_change()
    tmp["vol_3m"] = tmp["returns"].rolling(3).std() * np.sqrt(12) * 100  # annualized-ish %
    st.line_chart(tmp.set_index("month_dt")[["vol_3m"]], height=280)

with tab_table:
    st.subheader("Raw & Derived Data")
    show = df.copy()
    # prettier names
    rename = {
        "month_dt":"Month",
        "imports_loaded":"Loaded Imports",
        "exports_loaded":"Loaded Exports",
        "empties":"Empties",
        "total_teus":"Total TEUs",
        "mom_total_pct":"MoM Total (%)",
        "yoy_total_pct":"YoY Total (%)",
        "imports_loaded_share_pct":"Imports Share (%)",
        "exports_loaded_share_pct":"Exports Share (%)",
        "empties_share_pct":"Empties Share (%)",
    }
    for k,v in rename.items():
        if k in show.columns: show = show.rename(columns={k:v})
    fmt = {c:"{:,.0f}" for c in ["Loaded Imports","Loaded Exports","Empties","Total TEUs"] if c in show.columns}
    fmt.update({c:"{:.1f}%" for c in ["MoM Total (%)","YoY Total (%)","Imports Share (%)","Exports Share (%)","Empties Share (%)"] if c in show.columns})
    st.dataframe(show.set_index("Month").style.format(fmt), use_container_width=True)

# ---------- Analysis & Implications ----------
st.markdown("---")
st.header("Analysis & Implications")

latest_m = df.iloc[-1]
prev_m = df.iloc[-2] if len(df)>=2 else latest_m
mom = (latest_m["total_teus"]/prev_m["total_teus"]-1)*100 if prev_m["total_teus"] else np.nan
yoy = latest_m["yoy_total_pct"]
bullets = []

# MoM movement
if pd.notna(mom):
    if mom >= 5:
        bullets.append(f"Total throughput **up {mom:.1f}% MoM** — expect **near-term increases** in drayage moves and yard turns.")
    elif mom <= -5:
        bullets.append(f"Total throughput **down {mom:.1f}% MoM** — look for **temporary easing** in warehouse labor and chassis demand.")
    else:
        bullets.append(f"Throughput changed **{mom:.1f}% MoM**, within a normal band.")

# YoY context
if pd.notna(yoy):
    if yoy > 0:
        bullets.append(f"YoY growth **{yoy:.1f}%** indicates **stronger demand** vs last year; prepare for sustained capacity pressure.")
    else:
        bullets.append(f"YoY decline **{yoy:.1f}%** suggests **softer flow**; opportunity to catch up on maintenance and reposition empties.")

# Mix implications
parts = []
for p, label in [("imports_loaded_share_pct","imports"),("exports_loaded_share_pct","exports"),("empties_share_pct","empties")]:
    if p in df.columns: parts.append((label, latest_m[p]))
if parts:
    dom = max(parts, key=lambda x: x[1])
    if dom[0] == "empties" and dom[1] > 30:
        bullets.append("High **empties share** → coordinate return sweeps and inland repositioning to avoid yard congestion.")
    if dom[0] == "imports" and dom[1] > 60:
        bullets.append("Imports dominate the mix → plan for **more inbound drays** and **cross-dock labor** next month.")

bullets.append("Pair this with **diesel prices** (cost signal) and **rail traffic** (inland flow) for a fuller demand picture.")

st.markdown("\n".join([f"- {b}" for b in bullets]))
st.caption("Source: Port of Los Angeles — Container Statistics (official). Latest month posted ~15th of each month.")
