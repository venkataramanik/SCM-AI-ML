# streamlit_app.py
# Port of Los Angeles "The Signal" dashboard (no tokens, no AI icon).
# Sections:
# 1) Data Source  2) What We Do With The Data  3) Analysis & Implications  4) Dashboards

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Port of LA – The Signal Dashboard", layout="wide")

TITLE = "Port of Los Angeles – Inbound Volume Dashboard"
SOURCE_URL = "https://signal.portoptimizer.com/"

# -------------------------------
# Section: Header
# -------------------------------
st.title(TITLE)
st.caption("A practical, business-focused view of inbound container flow using the public 'The Signal' page.")

# -------------------------------
# Helpers
# -------------------------------
@st.cache_data(ttl=6*3600, show_spinner=True)
def load_signal_table():
    """
    Attempt to read the main table from Port of LA 'The Signal' page via pandas.read_html.
    Returns a tidy DataFrame with normalized column names and types.
    """
    tables = pd.read_html(SOURCE_URL)
    # The page usually has a primary table with weekly forecast/actual TEUs.
    # If structure changes, pick the widest table.
    df = max(tables, key=lambda t: t.shape[1]).copy()

    # Normalize column names
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]

    # Try common name patterns; fallback gracefully
    # Expected columns (examples): "Week of", "Forecast TEUs", "Actual TEUs", "% Difference"
    colmap_candidates = {
        "week": ["Week of", "Week", "Week Beginning", "Week beginning", "Week Start"],
        "forecast": ["Forecast TEUs", "Forecast", "Forecast TEU", "Forecast (TEUs)"],
        "actual": ["Actual TEUs", "Actual", "Actual TEU", "Actual (TEUs)"],
        "pct_diff": ["% Difference", "Difference %", "Forecast vs Actual (%)", "% diff"],
    }

    def best_match(name_list):
        for name in name_list:
            if name in df.columns:
                return name
        return None

    week_col = best_match(colmap_candidates["week"])
    forecast_col = best_match(colmap_candidates["forecast"])
    actual_col = best_match(colmap_candidates["actual"])
    pct_col = best_match(colmap_candidates["pct_diff"])

    # Basic validation
    needed = [week_col, forecast_col, actual_col]
    if any(x is None for x in needed):
        raise ValueError("Expected columns not found. The page layout may have changed.")

    out = df[[week_col, forecast_col, actual_col] + ([pct_col] if pct_col else [])].copy()
    out = out.rename(columns={
        week_col: "week",
        forecast_col: "forecast_teus",
        actual_col: "actual_teus",
        (pct_col if pct_col else "pct_diff"): "pct_diff"
    })

    # Coerce types
    out["week"] = pd.to_datetime(out["week"], errors="coerce")
    for c in ["forecast_teus", "actual_teus"]:
        out[c] = (
            out[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"([0-9]+)", expand=False)
            .astype(float)
        )

    if "pct_diff" in out.columns:
        out["pct_diff"] = (
            out["pct_diff"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.extract(r"(-?\d+\.?\d*)", expand=False)
            .astype(float)
        )

    # Sort ascending by week
    out = out.dropna(subset=["week"]).sort_values("week").reset_index(drop=True)

    # Derive fields
    out["wow_actual_change_pct"] = out["actual_teus"].pct_change() * 100.0
    out["rolling_4w_actual"] = out["actual_teus"].rolling(4, min_periods=1).mean()
    out["yoy_actual_change_pct"] = (
        (out["actual_teus"] / out["actual_teus"].shift(52) - 1.0) * 100.0
        if len(out) >= 60 else np.nan
    )

    # Create a 'year-week' label for heatmaps
    out["year"] = out["week"].dt.isocalendar().year.astype(int)
    out["iso_week"] = out["week"].dt.isocalendar().week.astype(int)

    return out

def format_number(x):
    if pd.isna(x):
        return "—"
    return f"{int(x):,}"

def format_pct(x, decimals=1):
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}%"

# -------------------------------
# Section: Data Source
# -------------------------------
with st.expander("Data Source", expanded=True):
    st.markdown(f"""
- **Primary Source:** [Port of Los Angeles – The Signal]({SOURCE_URL})  
- **Data Type:** Weekly inbound container volumes (TEUs), forecasts vs. actuals  
- **Access Method:** Public HTML table parsed via `pandas.read_html()`  
- **Refresh in App:** Every 6 hours (cached)  
- **Downloaded at:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
""")

# -------------------------------
# Section: What We Do With The Data
# -------------------------------
with st.expander("What We Do With The Data (Methodology)", expanded=True):
    st.markdown("""
1. **Extract** the weekly table using `pandas.read_html()` and normalize headers.  
2. **Clean & Typecast**: parse dates, convert TEU strings to numeric, strip percent signs.  
3. **Engineer Features**:  
   - Week-over-week change in actual TEUs (`wow_actual_change_pct`)  
   - 4-week rolling average of actual TEUs (`rolling_4w_actual`)  
   - Year-over-year change in actual TEUs (`yoy_actual_change_pct`, when ≥ 52 weeks available)  
4. **Visualize**: trend lines (forecast vs actual), rolling averages, accuracy, and a WoW heatmap.  
5. **Flag Risks**: simple rules on spikes/drops to drive operational discussion.
""")

# -------------------------------
# Load Data + Guardrails
# -------------------------------
try:
    df = load_signal_table()
except Exception as e:
    st.error(f"Could not load the data from The Signal. Error: {e}")
    st.stop()

if df.empty:
    st.warning("Data loaded but appears empty.")
    st.stop()

# -------------------------------
# Section: Dashboards (Tabs)
# -------------------------------
tab_overview, tab_trends, tab_accuracy, tab_heatmap, tab_table = st.tabs(
    ["Overview", "Trends", "Forecast vs Actual", "WoW Heatmap", "Data Table"]
)

# ---------- Overview ----------
with tab_overview:
    st.subheader("Key Metrics")
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Latest Week", latest["week"].date().isoformat())
    kpi_cols[1].metric("Actual TEUs", format_number(latest["actual_teus"]),
                       delta=format_number(latest["actual_teus"] - prev["actual_teus"]) if len(df) >= 2 else None)
    kpi_cols[2].metric("Forecast TEUs", format_number(latest["forecast_teus"]))
    kpi_cols[3].metric("WoW Change (Actual)", format_pct(latest["wow_actual_change_pct"]),
                       delta=f"{format_pct(latest['wow_actual_change_pct'] - prev['wow_actual_change_pct'])}" if len(df) >= 3 and pd.notna(prev["wow_actual_change_pct"]) else None)

    st.markdown("---")

    # Simple risk flags
    flags = []
    if pd.notna(latest["wow_actual_change_pct"]):
        if latest["wow_actual_change_pct"] >= 8:
            flags.append("Significant week-over-week increase in inbound TEUs (≥ 8%). Consider drayage and cross-dock capacity adjustments.")
        if latest["wow_actual_change_pct"] <= -8:
            flags.append("Significant week-over-week drop in inbound TEUs (≤ -8%). Expect softer near-term warehouse labor demand.")
    if len(df) >= 5:
        if latest["actual_teus"] > df["rolling_4w_actual"].iloc[-2] * 1.10:
            flags.append("Actual TEUs exceeded 4-week average by >10% — potential short-term congestion risk.")
        if latest["actual_teus"] < df["rolling_4w_actual"].iloc[-2] * 0.90:
            flags.append("Actual TEUs fell >10% below 4-week average — potential easing of congestion.")

    if flags:
        st.subheader("Risk/Opportunity Signals")
        for f in flags:
            st.write(f"- {f}")
    else:
        st.info("No notable risk flags based on current simple rules.")

# ---------- Trends ----------
with tab_trends:
    st.subheader("Rolling View of Inbound Volumes")
    lookback_weeks = st.slider("Show last N weeks", min_value=12, max_value=min(104, len(df)), value=min(52, len(df)))
    dff = df.tail(lookback_weeks).copy()

    st.line_chart(
        dff.set_index("week")[["actual_teus", "rolling_4w_actual"]],
        height=320
    )
    st.caption("Lines: Actual TEUs (weekly) and 4-week rolling average (smoother trend).")

# ---------- Forecast vs Actual ----------
with tab_accuracy:
    st.subheader("Forecast vs Actual (Weekly)")
    dfa = df.copy()
    dfa["forecast_error_teus"] = dfa["actual_teus"] - dfa["forecast_teus"]
    dfa["forecast_error_pct"] = (dfa["forecast_error_teus"] / dfa["forecast_teus"]) * 100.0

    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(
            dfa.set_index("week")[["forecast_teus", "actual_teus"]],
            height=320
        )
        st.caption("Weekly forecast vs actual TEUs.")

    with c2:
        st.bar_chart(
            dfa.set_index("week")[["forecast_error_pct"]],
            height=320
        )
        st.caption("Forecast error (%): positive = actual above forecast, negative = actual below forecast.")

    # Summary stats
    st.markdown("**Forecast Accuracy Summary**")
    summary = pd.DataFrame({
        "MAPE (%)": [dfa["forecast_error_pct"].abs().mean()],
        "Bias (Mean Error, TEUs)": [dfa["forecast_error_teus"].mean()],
        "RMSE (TEUs)": [np.sqrt((dfa["forecast_error_teus"]**2).mean())]
    })
    st.dataframe(summary.style.format({"MAPE (%)": "{:.1f}", "Bias (Mean Error, TEUs)": "{:,.0f}", "RMSE (TEUs)": "{:,.0f}"}), use_container_width=True)

# ---------- WoW Heatmap ----------
with tab_heatmap:
    st.subheader("Week-over-Week Change Heatmap")
    heat = df[["year", "iso_week", "wow_actual_change_pct"]].dropna().copy()

    # Pivot to year x week for a matrix-like view
    pivot = heat.pivot(index="year", columns="iso_week", values="wow_actual_change_pct").sort_index(ascending=False)
    st.dataframe(
        pivot.style.background_gradient(cmap="RdYlGn").format("{:.0f}%").set_precision(0),
        use_container_width=True
    )
    st.caption("Cells show % change in actual TEUs vs prior week (Red = drop, Green = increase).")

# ---------- Data Table ----------
with tab_table:
    st.subheader("Raw & Derived Data")
    show_cols = ["week", "forecast_teus", "actual_teus", "pct_diff", "wow_actual_change_pct", "rolling_4w_actual", "yoy_actual_change_pct"]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(
        df[show_cols].rename(columns={
            "week": "Week",
            "forecast_teus": "Forecast TEUs",
            "actual_teus": "Actual TEUs",
            "pct_diff": "% Difference (Site)",
            "wow_actual_change_pct": "WoW Change (%)",
            "rolling_4w_actual": "4W Rolling Avg (TEUs)",
            "yoy_actual_change_pct": "YoY Change (%)"
        }).style.format({
            "Forecast TEUs": "{:,.0f}",
            "Actual TEUs": "{:,.0f}",
            "% Difference (Site)": "{:.1f}",
            "WoW Change (%)": "{:.1f}",
            "4W Rolling Avg (TEUs)": "{:,.0f}",
            "YoY Change (%)": "{:.1f}",
        }),
        use_container_width=True
    )

# -------------------------------
# Section: Analysis & Implications
# -------------------------------
st.markdown("---")
st.header("Analysis & Implications")

# Generate concise narrative
latest_week = df["week"].iloc[-1].date().isoformat()
latest_actual = df["actual_teus"].iloc[-1]
prev_actual = df["actual_teus"].iloc[-2] if len(df) >= 2 else latest_actual
wow = (latest_actual - prev_actual) / prev_actual * 100 if prev_actual else np.nan
rolling = df["rolling_4w_actual"].iloc[-1]

bullets = []
if not pd.isna(wow):
    if wow >= 5:
        bullets.append(f"Inbound volume rose {wow:.1f}% week-over-week (week of {latest_week}). Prepare for near-term increases in drayage moves and yard turns.")
    elif wow <= -5:
        bullets.append(f"Inbound volume fell {wow:.1f}% week-over-week (week of {latest_week}). Expect short-term easing in yard utilization and chassis demand.")
    else:
        bullets.append(f"Inbound volume moved {wow:.1f}% WoW (week of {latest_week}), within a normal range.")

if latest_actual > rolling * 1.10:
    bullets.append("Actual TEUs are >10% above the 4-week average — heightened risk of berth/yard congestion and overtime labor.")
elif latest_actual < rolling * 0.90:
    bullets.append("Actual TEUs are >10% below the 4-week average — opportunity to catch up on maintenance and reposition empties.")

bullets.append("Use rolling averages, not single weeks, to plan labor shifts and drayage capacity. Pair with inland rail and diesel price signals for a fuller picture.")

st.markdown("\n".join([f"- {b}" for b in bullets]))
