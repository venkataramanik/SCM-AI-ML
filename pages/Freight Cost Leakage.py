```python
# app.py
# 🚢✈️ Freight Cost Leakage RCA — Ocean + Air (Leadership Demo)
# Streamlit app that:
#  1) Loads invoice-line data (CSV upload) or generates synthetic demo data
#  2) Computes Expected vs Billed variance (leakage)
#  3) Assigns explainable Root Cause (RCA) reason codes
#  4) Shows an executive-friendly "tree" graphic (Plotly Treemap) + Pareto + Top offenders
#
# Run:
#   pip install streamlit pandas numpy matplotlib plotly
#   streamlit run app.py

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Freight Cost Leakage RCA — Ocean + Air",
    page_icon="🚢✈️",
    layout="wide",
)

st.title("🚢✈️ Freight Cost Leakage RCA — Ocean + Air")
st.caption(
    "Leadership-ready demo: quantify leakage ($), explain the 'why' (RCA), and show where to act (tree + offenders)."
)

np.random.seed(7)


# -----------------------------
# Thresholds / config
# -----------------------------
@dataclass
class Thresholds:
    abs_usd: float = 50.0
    pct: float = 0.06
    duplicate_days_window: int = 7


# -----------------------------
# Data schema guidance
# -----------------------------
REQUIRED_COLS = [
    "shipment_id",
    "invoice_id",
    "mode",            # AIR or OCEAN
    "lane",            # e.g., CN->US
    "carrier",
    "ship_date",
    "charge_category", # BASE/FUEL/THC/DND/etc.
    "expected_amount",
    "billed_amount",
]

OPTIONAL_COLS = [
    "gross_kg", "vol_m3", "pieces",
    "dwell_days", "free_time_days",
    "contract_missing",  # bool
]


# -----------------------------
# Synthetic demo data generator
# -----------------------------
def make_synthetic_data(n: int = 1400) -> pd.DataFrame:
    modes = np.random.choice(["OCEAN", "AIR"], size=n, p=[0.55, 0.45])
    lanes = np.random.choice(["CN->US", "DE->US", "SG->US", "US->EU", "IN->US"], size=n)
    carriers = np.random.choice(["CarrierA", "CarrierB", "CarrierC", "ForwarderX", "ForwarderY"], size=n)

    charge_category = np.random.choice(
        ["BASE", "FUEL", "SECURITY", "SCREENING", "THC", "DOC", "DND", "HANDLING"],
        size=n,
        p=[0.35, 0.20, 0.06, 0.05, 0.10, 0.07, 0.10, 0.07],
    )

    ship_date = pd.to_datetime("2025-09-01") + pd.to_timedelta(np.random.randint(0, 120, size=n), unit="D")

    gross_kg = np.clip(np.random.normal(900, 450, size=n), 20, 4000)
    vol_m3 = np.clip(np.random.normal(6.5, 3.0, size=n), 0.1, 25)
    pieces = np.random.randint(1, 40, size=n)

    dwell_days = np.clip(np.random.poisson(lam=3.2, size=n), 0, 20)
    free_time_days = np.where(lanes == "CN->US", 4, 5)

    contract_missing = np.random.rand(n) < 0.06

    expected_base = np.where(
        modes == "OCEAN",
        np.where(lanes == "CN->US", 3500, np.where(lanes == "SG->US", 3100, 2800)),
        np.where(lanes == "DE->US", 4200, np.where(lanes == "CN->US", 5100, 3900)),
    ).astype(float)

    expected_amount = np.zeros(n, dtype=float)
    for i, cat in enumerate(charge_category):
        if cat == "BASE":
            expected_amount[i] = expected_base[i]
        elif cat == "FUEL":
            expected_amount[i] = (0.18 if modes[i] == "AIR" else 0.12) * expected_base[i]
        elif cat in ["SECURITY", "SCREENING"]:
            expected_amount[i] = (80 + 0.02 * expected_base[i]) if modes[i] == "AIR" else 0.0
        elif cat == "THC":
            expected_amount[i] = 250 if modes[i] == "OCEAN" else 0.0
        elif cat == "DOC":
            expected_amount[i] = 65 if modes[i] == "OCEAN" else 45
        elif cat == "HANDLING":
            expected_amount[i] = 55 if modes[i] == "AIR" else 0.0
        elif cat == "DND":
            expected_amount[i] = max(0, int(dwell_days[i] - free_time_days[i])) * 150 if modes[i] == "OCEAN" else 0.0

    billed_amount = expected_amount.copy()

    # Inject realistic leakage patterns
    contract_mismatch = np.random.rand(n) < 0.05
    air_weight_issue = (modes == "AIR") & (np.random.rand(n) < 0.06)
    dnd_not_eligible = (
        (charge_category == "DND")
        & (modes == "OCEAN")
        & (dwell_days <= free_time_days)
        & (np.random.rand(n) < 0.35)
    )
    wrong_service = (modes == "AIR") & (charge_category == "BASE") & (np.random.rand(n) < 0.04)
    duplicate_flag = np.random.rand(n) < 0.03

    billed_amount += np.where(contract_mismatch, 0.10 * np.maximum(expected_amount, 200), 0.0)
    billed_amount += np.where(air_weight_issue & np.isin(charge_category, ["BASE", "FUEL"]),
                              0.15 * np.maximum(expected_amount, 300), 0.0)
    billed_amount += np.where(dnd_not_eligible, 300 + np.random.randint(50, 400, size=n), 0.0)
    billed_amount += np.where(wrong_service, 0.20 * np.maximum(expected_amount, 1000), 0.0)

    billed_amount += np.random.normal(0, 8, size=n)
    billed_amount = np.clip(billed_amount, 0, None)

    df = pd.DataFrame({
        "shipment_id": ["S" + str(100000 + i) for i in range(n)],
        "invoice_id": ["I" + str(50000 + (i // 3)) for i in range(n)],
        "mode": modes,
        "lane": lanes,
        "carrier": carriers,
        "ship_date": ship_date,
        "charge_category": charge_category,
        "gross_kg": gross_kg.round(1),
        "vol_m3": vol_m3.round(2),
        "pieces": pieces,
        "dwell_days": dwell_days,
        "free_time_days": free_time_days,
        "contract_missing": contract_missing,
        "expected_amount": expected_amount.round(2),
        "billed_amount": billed_amount.round(2),
        "synthetic_dup_hint": duplicate_flag,
    })

    # Add some duplicate-ish lines for demo
    dup_rows = df[df["synthetic_dup_hint"]].head(20).copy()
    if len(dup_rows) > 0:
        dup_rows["ship_date"] = dup_rows["ship_date"] + pd.to_timedelta(
            np.random.randint(0, 5, size=len(dup_rows)), unit="D"
        )
        df = pd.concat([df, dup_rows], ignore_index=True)

    return df


# -----------------------------
# Cleaning / validation
# -----------------------------
def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        warnings.append(f"Missing required columns: {missing}")

    if "ship_date" in df.columns:
        df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")
        if df["ship_date"].isna().any():
            warnings.append("Some ship_date values could not be parsed; they became NaT.")

    for col in ["expected_amount", "billed_amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                warnings.append(f"Some {col} values are non-numeric; they became NaN.")

    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str).str.upper().str.strip()
    if "charge_category" in df.columns:
        df["charge_category"] = df["charge_category"].astype(str).str.upper().str.strip()

    if "contract_missing" in df.columns:
        df["contract_missing"] = df["contract_missing"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    else:
        df["contract_missing"] = False

    # Add optional columns if missing (keeps UI stable)
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df["expected_amount"] = df["expected_amount"].fillna(0.0)
    df["billed_amount"] = df["billed_amount"].fillna(0.0)

    return df, warnings


# -----------------------------
# Leakage detection helpers
# -----------------------------
def detect_duplicates(df: pd.DataFrame, t: Thresholds) -> pd.Series:
    """
    Leadership-friendly duplicate heuristic:
    Same shipment_id + charge_category + billed_amount within a short time window.
    """
    tmp = df[["shipment_id", "charge_category", "billed_amount", "ship_date"]].copy()
    tmp = tmp.sort_values(["shipment_id", "charge_category", "billed_amount", "ship_date"])

    dup = np.zeros(len(tmp), dtype=bool)
    same_key = (
        (tmp["shipment_id"].values[1:] == tmp["shipment_id"].values[:-1]) &
        (tmp["charge_category"].values[1:] == tmp["charge_category"].values[:-1]) &
        (np.isclose(tmp["billed_amount"].values[1:], tmp["billed_amount"].values[:-1], atol=0.01))
    )
    date_diff = (tmp["ship_date"].values[1:] - tmp["ship_date"].values[:-1]) / np.timedelta64(1, "D")
    within = (date_diff >= 0) & (date_diff <= t.duplicate_days_window)
    dup[1:] = same_key & within

    return pd.Series(dup, index=tmp.index).reindex(df.index).fillna(False).astype(bool)


def compute_leakage(df: pd.DataFrame, t: Thresholds) -> pd.DataFrame:
    df = df.copy()
    df["variance"] = (df["billed_amount"] - df["expected_amount"]).round(2)
    df["variance_pct"] = np.where(df["expected_amount"] > 0, df["variance"] / df["expected_amount"], np.nan)

    df["is_duplicate"] = detect_duplicates(df, t)

    df["is_leakage"] = (
        (df["variance"] > t.abs_usd) &
        (
            (df["variance_pct"].fillna(0) > t.pct) |
            (df["expected_amount"] == 0)
        )
    )

    return df


# -----------------------------
# RCA attribution (explainable)
# -----------------------------
def rca_reason_codes(row: pd.Series) -> Tuple[str, str]:
    mode = str(row.get("mode", "")).upper()
    cat = str(row.get("charge_category", "")).upper()
    variance = float(row.get("variance", 0.0))
    expected = float(row.get("expected_amount", 0.0))
    billed = float(row.get("billed_amount", 0.0))

    contract_missing = bool(row.get("contract_missing", False))
    is_dup = bool(row.get("is_duplicate", False))

    if variance <= 0:
        return ("NO LEAKAGE", "No material positive variance.")

    # 1) Cleanest to explain
    if is_dup:
        return ("DUPLICATE BILLING", "Same shipment + charge + amount appears more than once within the duplicate window.")

    # 2) Contract linkage is a top real-world driver
    if contract_missing:
        return ("CONTRACT / RATE CARD NOT APPLIED", "Contract linkage missing; billed may reflect tariff/spot or default rating.")

    # 3) Ocean: D&D eligibility and clock issues
    if mode == "OCEAN" and cat == "DND":
        dwell = row.get("dwell_days", None)
        free = row.get("free_time_days", None)

        if expected == 0 and billed > 0:
            if dwell is not None and free is not None and not (pd.isna(dwell) or pd.isna(free)):
                return (
                    "D&D NOT ELIGIBLE / FREE TIME MISAPPLIED",
                    f"Expected D&D = $0 (dwell {int(dwell)}d ≤ free time {int(free)}d), but D&D was billed."
                )
            return ("D&D NOT ELIGIBLE / FREE TIME MISAPPLIED", "Expected D&D = $0, but D&D was billed.")

        if expected > 0 and variance > max(200, 0.5 * expected):
            return (
                "D&D CLOCK / DAY-COUNT MISMATCH",
                "Variance on D&D is unusually high; likely clock start/stop, day-count, or day-rate mismatch vs terms."
            )

    # 4) Air: chargeable weight & rating/service mismatch
    if mode == "AIR" and cat in ["BASE", "FUEL"]:
        gross = row.get("gross_kg", None)
        vol_m3 = row.get("vol_m3", None)
        if gross is not None and vol_m3 is not None and not (pd.isna(gross) or pd.isna(vol_m3)):
            vol_kg = float(vol_m3) * 167.0  # rule-of-thumb conversion
            if vol_kg > float(gross) * 1.15:
                return (
                    "CHARGEABLE WEIGHT / DIMENSIONAL RATING ISSUE",
                    f"Volumetric weight likely dominates (vol≈{vol_kg:.0f}kg > gross≈{gross:.0f}kg). Recheck dims/divisor/rounding."
                )
        return ("SERVICE LEVEL / RATING MISMATCH", "Air variance often comes from service-level, rating logic, or chargeable weight rules.")

    # 5) Accessorial / surcharge drift
    if cat in ["FUEL", "SECURITY", "SCREENING", "THC", "DOC", "HANDLING"]:
        return ("SURCHARGE / ACCESSORIAL MISAPPLIED", "Charge exists but exceeds expected; validate eligibility, formula, effective dates, and currency.")

    return ("OTHER BILLING VARIANCE", "Variance exists; drill into contract terms, event evidence, and invoice detail.")


def apply_rca(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rc = []
    ex = []
    for _, r in df.iterrows():
        a, b = rca_reason_codes(r)
        rc.append(a)
        ex.append(b)
    df["root_cause"] = rc
    df["rca_explanation"] = ex
    return df


# -----------------------------
# Leadership tree graphic (Plotly Treemap)
# -----------------------------
def build_tree_figure(tree_summary: pd.DataFrame) -> go.Figure:
    total = float(tree_summary["leakage_amount"].sum())

    labels = ["TOTAL LEAKAGE"]
    parents = [""]
    values = [total]

    for mode in ["OCEAN", "AIR"]:
        sub = tree_summary[tree_summary["mode"] == mode].copy()
        mode_total = float(sub["leakage_amount"].sum())

        labels.append(f"{mode}")
        parents.append("TOTAL LEAKAGE")
        values.append(mode_total)

        sub = sub.sort_values("leakage_amount", ascending=False).head(8)
        for _, r in sub.iterrows():
            labels.append(r["root_cause"])
            parents.append(mode)
            values.append(float(r["leakage_amount"]))

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>Leakage: $%{value:,.0f}<extra></extra>",
    ))
    fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
    return fig


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Demo Controls")

data_mode = st.sidebar.radio(
    "Data source",
    ["Use synthetic demo data", "Upload CSV (invoice lines)"],
    index=0
)

t = Thresholds(
    abs_usd=st.sidebar.slider("Flag leakage if variance ≥ ($)", 0, 500, 50, 10),
    pct=st.sidebar.slider("AND variance ≥ (%)", 0.00, 0.50, 0.06, 0.01),
    duplicate_days_window=st.sidebar.slider("Duplicate window (days)", 1, 30, 7, 1),
)

st.sidebar.markdown("---")
show_notes = st.sidebar.checkbox("Show 'How it works' notes", value=True)
show_detail = st.sidebar.checkbox("Show detailed flagged lines table", value=True)


# -----------------------------
# Load data
# -----------------------------
if data_mode == "Use synthetic demo data":
    df = make_synthetic_data()
    df, warns = validate_and_clean(df)
else:
    uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV, or switch back to synthetic demo data.")
        st.stop()
    df = pd.read_csv(uploaded)
    df, warns = validate_and_clean(df)

if warns:
    with st.expander("⚠️ Data warnings"):
        for w in warns:
            st.write(f"- {w}")

missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_required:
    st.error(f"Dataset missing required columns: {missing_required}")
    st.stop()


# -----------------------------
# Compute leakage + RCA
# -----------------------------
df2 = compute_leakage(df, t)
df2 = apply_rca(df2)

flagged = df2[df2["is_leakage"]].copy()
flagged["leakage_amount"] = flagged["variance"].clip(lower=0)


# -----------------------------
# Executive KPIs
# -----------------------------
total_spend = float(df2["billed_amount"].sum())
total_leak = float(flagged["leakage_amount"].sum())
leak_rate = (total_leak / total_spend) if total_spend > 0 else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total billed spend", f"${total_spend:,.0f}")
k2.metric("Leakage identified", f"${total_leak:,.0f}")
k3.metric("Leakage rate", f"{leak_rate*100:.2f}%")
k4.metric("Flagged line items", f"{len(flagged):,}")


# -----------------------------
# Leadership views
# -----------------------------
c1, c2 = st.columns([1.1, 0.9])

with c1:
    st.subheader("📌 Where the money leaks (Mode → Root Cause)")
    if len(flagged) == 0:
        st.warning("No leakage flagged with current thresholds. Try lowering thresholds.")
    else:
        summary = (
            flagged.groupby(["mode", "root_cause"], as_index=False)["leakage_amount"]
            .sum()
            .sort_values("leakage_amount", ascending=False)
        )
        st.dataframe(summary, use_container_width=True, height=320)

with c2:
    st.subheader("📊 Top drivers (Pareto)")
    if len(flagged) == 0:
        st.info("No flagged lines to chart.")
    else:
        pareto = (
            flagged.groupby("root_cause", as_index=False)["leakage_amount"]
            .sum()
            .sort_values("leakage_amount", ascending=False)
            .head(10)
        )

        fig = plt.figure()
        plt.bar(pareto["root_cause"], pareto["leakage_amount"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Leakage ($)")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)


# -----------------------------
# Tree view
# -----------------------------
st.subheader("🌳 Executive RCA Tree (click to drill)")
if len(flagged) == 0:
    st.info("No leakage flagged — adjust thresholds to populate the tree.")
else:
    tree_summary = (
        flagged.groupby(["mode", "root_cause"], as_index=False)["leakage_amount"].sum()
    )
    st.plotly_chart(build_tree_figure(tree_summary), use_container_width=True)


# -----------------------------
# Offenders view (who to act on)
# -----------------------------
st.subheader("🏷️ Top offenders to act on (Carrier/Forwarder, Lane)")
if len(flagged) == 0:
    st.info("No flagged lines to summarize.")
else:
    o1, o2 = st.columns(2)

    with o1:
        st.markdown("**By carrier/forwarder**")
        offenders_carrier = (
            flagged.groupby("carrier", as_index=False)["leakage_amount"].sum()
            .sort_values("leakage_amount", ascending=False)
            .head(10)
        )
        st.dataframe(offenders_carrier, use_container_width=True, height=300)

    with o2:
        st.markdown("**By lane**")
        offenders_lane = (
            flagged.groupby("lane", as_index=False)["leakage_amount"].sum()
            .sort_values("leakage_amount", ascending=False)
            .head(10)
        )
        st.dataframe(offenders_lane, use_container_width=True, height=300)


# -----------------------------
# Drill-down detail (what to dispute)
# -----------------------------
st.subheader("🔎 Drill-down: Top leakage invoice lines (ready for dispute)")
if len(flagged) == 0:
    st.info("No drill-down available without flagged lines.")
else:
    topn = st.slider("Lines to show", 10, 200, 30, 10)
    drill = flagged.sort_values("leakage_amount", ascending=False).head(topn).copy()

    show_cols = [
        "shipment_id", "invoice_id", "mode", "lane", "carrier", "ship_date",
        "charge_category", "expected_amount", "billed_amount", "variance",
        "root_cause", "rca_explanation",
        "gross_kg", "vol_m3", "dwell_days", "free_time_days", "contract_missing",
        "is_duplicate",
    ]
    show_cols = [c for c in show_cols if c in drill.columns]

    if show_detail:
        st.dataframe(drill[show_cols], use_container_width=True, height=360)

    st.markdown("**Example dispute packet (copy/paste):**")
    pick = drill.iloc[0]
    dispute_text = f"""
    Shipment {pick.get('shipment_id')} / Invoice {pick.get('invoice_id')}
    Mode: {pick.get('mode')} | Lane: {pick.get('lane')} | Carrier/Forwarder: {pick.get('carrier')}
    Charge: {pick.get('charge_category')}
    Expected: ${pick.get('expected_amount'):,.2f} | Billed: ${pick.get('billed_amount'):,.2f} | Variance: ${pick.get('variance'):,.2f}

    Root Cause: {pick.get('root_cause')}
    Evidence/Rationale: {pick.get('rca_explanation')}
    """
    st.code(textwrap.dedent(dispute_text).strip())


# -----------------------------
# Notes / explanation (leadership friendly)
# -----------------------------
if show_notes:
    st.subheader("📝 How this works (simple, explainable, scalable)")

    st.markdown("""
### What this demo proves
- We can compute **Expected vs Billed** at the **invoice-line** level (where leakage hides).
- We can assign a **plain-English root cause** for each exception.
- We can produce an **executive view**: tree → top drivers → top offenders → dispute-ready lines.

### What the “RCA model” is doing
This is an **explainable rules + evidence** approach (audit-friendly):
1) **Detect leakage** using thresholds (min $ and %).
2) **Attribute root cause** using evidence:
   - **Duplicate billing**: same shipment + charge + amount repeated within a short window
   - **Contract not applied**: missing contract linkage is a common driver of misrating
   - **Ocean D&D**:
     - billed when **not eligible** (dwell ≤ free time)
     - or billed with **clock/day-count mismatch**
   - **Air rating**:
     - likely **chargeable weight/dim** issue when volumetric dominates gross
     - otherwise **service/rating mismatch**
   - **Surcharge/accessorial drift** for fuel/security/THC/docs/handling

### What you need in real life (to go from demo → pilot)
To make this production-grade for a shipper like 3M, you need 3 layers of data:

**1) Invoice line items (must-have)**
- invoice_id, shipment_id, charge code/category, amount, currency, billed party
- carrier/forwarder, lane, service level/product
- invoice dates and references

**2) Shipment facts + event timestamps (to validate eligibility)**
- Ocean: availability, pickup, return empty, terminal/port, free-time terms
- Air: AWB, pieces, dims, gross vs chargeable weight rules, tender/uplift/arrival/delivery

**3) Contract & surcharge logic (to compute Expected)**
- contracted rates by lane/service/equipment + effective dates
- fuel/security formulas, currency rules, accessorial eligibility, free time clauses

### What leadership should expect as outcomes
- **Leakage $** and leakage rate by mode, lane, carrier/forwarder, and charge type
- **Top root causes** (Pareto) to drive prevention
- **Top offenders** to manage carriers/forwarders and processes
- **Dispute packets** that accelerate recovery and reduce cycle time

### Next upgrades to make it “enterprise”
- Replace expected_amount with a real **expected-cost engine**
- Add **preventable vs operational** split (recovery vs prevention)
- Add a “claims/credits outcome loop” to train a supervised model later
""")

    st.markdown("#### Minimum CSV columns")
    st.code("\n".join(REQUIRED_COLS))
    st.markdown("#### Helpful optional columns (improves RCA quality)")
    st.code("\n".join(OPTIONAL_COLS))

st.caption("Tip: start with synthetic data for leadership. Then swap in invoice lines + contract terms + event milestones for a 4–6 week pilot.")
```
