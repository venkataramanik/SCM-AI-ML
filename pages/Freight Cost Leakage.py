# app.py
# Freight Cost Leakage RCA (Ocean + Air) — Streamlit leadership demo
# Run:
#   pip install streamlit pandas numpy matplotlib graphviz
#   streamlit run app.py
#
# Optional: upload your invoice-line CSV with the expected schema (see "Real-life data needs" section in UI)

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from graphviz import Digraph


# -----------------------------
# 0) Page setup
# -----------------------------
st.set_page_config(
    page_title="Freight Cost Leakage RCA — Ocean + Air",
    page_icon="🚢✈️",
    layout="wide",
)

st.title("🚢✈️ Freight Cost Leakage RCA — Ocean + Air (Shipper View)")
st.caption(
    "Leadership demo: invoice-line level leakage detection + explainable root-cause attribution + tree visualization."
)

np.random.seed(7)


# -----------------------------
# 1) Configuration / thresholds
# -----------------------------
@dataclass
class Thresholds:
    abs_usd: float = 50.0            # minimum variance ($) to consider leakage
    pct: float = 0.06                # minimum variance (%) to consider leakage
    air_dim_divisor: float = 6000.0  # typical divisor for cm-based volumetric weight (demo)
    ocean_dnd_day_rate: float = 150.0
    duplicate_days_window: int = 7


# -----------------------------
# 2) Synthetic demo data generator
# -----------------------------
def make_synthetic_data(n: int = 1200) -> pd.DataFrame:
    modes = np.random.choice(["OCEAN", "AIR"], size=n, p=[0.55, 0.45])
    lanes = np.random.choice(["CN->US", "DE->US", "SG->US", "US->EU", "IN->US"], size=n)
    carriers = np.random.choice(["CarrierA", "CarrierB", "CarrierC", "ForwarderX", "ForwarderY"], size=n)

    # Charge categories: keep simple for leadership
    charge_category = np.random.choice(
        ["BASE", "FUEL", "SECURITY", "SCREENING", "THC", "DOC", "DND", "HANDLING"],
        size=n,
        p=[0.35, 0.20, 0.06, 0.05, 0.10, 0.07, 0.10, 0.07],
    )

    ship_date = pd.to_datetime("2025-09-01") + pd.to_timedelta(np.random.randint(0, 120, size=n), unit="D")

    # Shipment facts (simplified)
    gross_kg = np.clip(np.random.normal(900, 450, size=n), 20, 4000)
    vol_m3 = np.clip(np.random.normal(6.5, 3.0, size=n), 0.1, 25)
    pieces = np.random.randint(1, 40, size=n)

    # Ocean event proxy for D&D
    dwell_days = np.clip(np.random.poisson(lam=3.2, size=n), 0, 20)
    free_time_days = np.where(lanes == "CN->US", 4, 5)

    # Contract linkage (toy)
    contract_missing = np.random.rand(n) < 0.06

    # Expected base amounts by mode/lane (toy but plausible)
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

    # Leakage patterns (inject)
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
        "invoice_id": ["I" + str(50000 + (i // 3)) for i in range(n)],  # just to create grouping
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

    # create some "near-duplicates" by copying a few lines (demo)
    dup_rows = df[df["synthetic_dup_hint"]].head(20).copy()
    if len(dup_rows) > 0:
        dup_rows["invoice_id"] = dup_rows["invoice_id"]  # same invoice
        dup_rows["ship_date"] = dup_rows["ship_date"] + pd.to_timedelta(np.random.randint(0, 5, size=len(dup_rows)), unit="D")
        df = pd.concat([df, dup_rows], ignore_index=True)

    return df


# -----------------------------
# 3) Real-life CSV schema guidance
# -----------------------------
REQUIRED_COLS = [
    "shipment_id",
    "invoice_id",
    "mode",              # AIR or OCEAN
    "lane",              # e.g., CN->US
    "carrier",
    "ship_date",
    "charge_category",   # BASE/FUEL/... at least consistent categories
    "expected_amount",   # can be computed upstream; demo assumes provided or derived
    "billed_amount",
]

OPTIONAL_COLS = [
    "gross_kg", "vol_m3", "pieces",
    "dwell_days", "free_time_days",
    "contract_missing"
]


def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings = []
    df = df.copy()

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        warnings.append(f"Missing required columns: {missing}")

    # Basic types
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

    # Optional booleans
    if "contract_missing" in df.columns:
        df["contract_missing"] = df["contract_missing"].astype(str).str.lower().isin(["true", "1", "yes", "y"])

    # Fill missing expected with 0 (demo friendliness)
    if "expected_amount" in df.columns:
        df["expected_amount"] = df["expected_amount"].fillna(0.0)

    return df, warnings


# -----------------------------
# 4) Leakage detection + RCA (explainable)
# -----------------------------
def detect_duplicates(df: pd.DataFrame, t: Thresholds) -> pd.Series:
    """
    Simple duplicate heuristic:
    Same shipment_id + charge_category + billed_amount within a time window.
    """
    if not all(c in df.columns for c in ["shipment_id", "charge_category", "billed_amount", "ship_date"]):
        return pd.Series(False, index=df.index)

    tmp = df[["shipment_id", "charge_category", "billed_amount", "ship_date"]].copy()
    tmp["ship_date"] = pd.to_datetime(tmp["ship_date"], errors="coerce")
    tmp = tmp.sort_values(["shipment_id", "charge_category", "billed_amount", "ship_date"])

    dup = np.zeros(len(tmp), dtype=bool)
    # Compare with previous row in group
    same_key = (
        (tmp["shipment_id"].values[1:] == tmp["shipment_id"].values[:-1]) &
        (tmp["charge_category"].values[1:] == tmp["charge_category"].values[:-1]) &
        (np.isclose(tmp["billed_amount"].values[1:], tmp["billed_amount"].values[:-1], atol=0.01))
    )
    date_diff = (tmp["ship_date"].values[1:] - tmp["ship_date"].values[:-1]) / np.timedelta64(1, "D")
    within = (date_diff >= 0) & (date_diff <= t.duplicate_days_window)
    dup[1:] = same_key & within

    # Map back to original index
    dup_series = pd.Series(dup, index=tmp.index).reindex(df.index).fillna(False).astype(bool)
    return dup_series


def compute_leakage_flags(df: pd.DataFrame, t: Thresholds) -> pd.DataFrame:
    df = df.copy()
    df["variance"] = (df["billed_amount"] - df["expected_amount"]).round(2)
    df["variance_pct"] = np.where(df["expected_amount"] > 0, df["variance"] / df["expected_amount"], np.nan)

    df["is_leakage"] = (
        (df["variance"] > t.abs_usd) &
        (
            (df["variance_pct"].fillna(0) > t.pct) |
            (df["expected_amount"] == 0)  # if expected is zero but billed isn't, treat as suspect
        )
    )

    df["is_duplicate"] = detect_duplicates(df, t)
    return df


def rca_reason_codes(row: pd.Series, t: Thresholds) -> Tuple[str, str]:
    """
    Root cause attribution (explainable):
    Returns (root_cause, explanation).
    """
    mode = str(row.get("mode", "")).upper()
    cat = str(row.get("charge_category", "")).upper()
    variance = float(row.get("variance", 0.0))
    expected = float(row.get("expected_amount", 0.0))
    billed = float(row.get("billed_amount", 0.0))
    contract_missing = bool(row.get("contract_missing", False))
    is_dup = bool(row.get("is_duplicate", False))

    # 1) duplicates are the cleanest RCA
    if is_dup and variance > 0:
        return ("DUPLICATE BILLING", "Same shipment + charge + amount appears multiple times within a short window.")

    # 2) missing contract linkage often drives misrating / surcharge drift
    if contract_missing and variance > 0:
        return ("CONTRACT / RATE CARD NOT APPLIED", "Contract linkage missing; billed may be tariff/spot instead of contracted.")

    # 3) ocean D&D billed even when expected is zero is classic leakage
    if mode == "OCEAN" and cat == "DND":
        dwell = row.get("dwell_days", None)
        free = row.get("free_time_days", None)
        if expected == 0 and billed > 0:
            # If we have dwell/free time, add a stronger explanation
            if dwell is not None and free is not None and not (pd.isna(dwell) or pd.isna(free)):
                return ("D&D NOT ELIGIBLE / FREE TIME MISAPPLIED",
                        f"Expected D&D = $0 (dwell {int(dwell)}d ≤ free time {int(free)}d), but billed D&D exists.")
            return ("D&D NOT ELIGIBLE / FREE TIME MISAPPLIED", "Expected D&D = $0, but billed D&D exists.")

        # If expected > 0 but variance huge, often clock-start/stop or rate/day mismatch
        if expected > 0 and variance > max(200, 0.5 * expected):
            return ("D&D DAYS / CLOCK OR RATE MISMATCH",
                    "Large variance on D&D suggests incorrect day-count (clock) or day-rate mismatch vs policy/terms.")

    # 4) air base/fuel variance often ties to chargeable weight / service level
    if mode == "AIR" and cat in ["BASE", "FUEL"] and variance > 0:
        # If we have gross_kg and vol_m3, we can hint weight/dim issue
        gross = row.get("gross_kg", None)
        vol_m3 = row.get("vol_m3", None)
        if gross is not None and vol_m3 is not None and not (pd.isna(gross) or pd.isna(vol_m3)):
            # Quick, explainable check: if volumetric likely dominates, call it dim weight issue
            # Convert m3 to volumetric kg using a common simplification: 1 m3 ~ 167 kg (airfreight rule of thumb)
            vol_kg = float(vol_m3) * 167.0
            if vol_kg > float(gross) * 1.15:
                return ("CHARGEABLE WEIGHT / DIMENSIONAL RATING ISSUE",
                        f"Volumetric weight likely dominates (vol≈{vol_kg:.0f}kg > gross={gross:.0f}kg). Recheck dims/divisor/rounding.")

        return ("SERVICE LEVEL / RATING MISMATCH", "Air BASE/FUEL variance commonly comes from service level or rating/weight rules.")

    # 5) surcharge / accessorial misapplication
    if cat in ["FUEL", "SECURITY", "SCREENING", "THC", "DOC", "HANDLING"] and variance > 0:
        return ("SURCHARGE / ACCESSORIAL MISAPPLIED", "Charge exists but exceeds expected; validate code eligibility, formula, effective date, and currency.")

    # 6) fallback
    if variance > 0:
        return ("OTHER BILLING VARIANCE", "Variance exists; needs drill-down to contract terms, events, and invoice detail.")

    return ("NO LEAKAGE", "No material variance detected.")


def apply_rca(df: pd.DataFrame, t: Thresholds) -> pd.DataFrame:
    df = df.copy()
    root_causes = []
    explanations = []
    for _, r in df.iterrows():
        rc, ex = rca_reason_codes(r, t)
        root_causes.append(rc)
        explanations.append(ex)
    df["root_cause"] = root_causes
    df["rca_explanation"] = explanations
    return df


# -----------------------------
# 5) Build a "beautiful" RCA tree graphic
# -----------------------------
def build_rca_tree_graph(summary: pd.DataFrame, title: str) -> Digraph:
    """
    Build a tree:
        Total Leakage ($)
          -> Ocean Leakage
              -> root causes...
          -> Air Leakage
              -> root causes...
    summary: columns = [mode, root_cause, leakage_amount]
    """
    g = Digraph(comment=title)
    g.attr(rankdir="LR", bgcolor="white")
    g.attr("node", shape="box", style="rounded,filled", fillcolor="white", color="#444444", fontname="Helvetica")

    total = summary["leakage_amount"].sum()
    g.node("TOTAL", f"TOTAL LEAKAGE\n${total:,.0f}", fillcolor="#f7f7f7", penwidth="2")

    for mode in ["OCEAN", "AIR"]:
        sub = summary[summary["mode"] == mode].copy()
        mode_total = sub["leakage_amount"].sum()

        mode_id = f"MODE_{mode}"
        g.node(mode_id, f"{mode} LEAKAGE\n${mode_total:,.0f}", fillcolor="#f0f6ff", penwidth="2")
        g.edge("TOTAL", mode_id, penwidth="2")

        # Top causes for that mode
        sub = sub.sort_values("leakage_amount", ascending=False).head(6)
        for i, row in enumerate(sub.itertuples(index=False), start=1):
            cause = row.root_cause
            amt = row.leakage_amount
            cause_id = f"{mode}_{i}"
            label = f"{cause}\n${amt:,.0f}"
            g.node(cause_id, label, fillcolor="#fff7f0")
            g.edge(mode_id, cause_id)

    return g


# -----------------------------
# 6) Sidebar: Data source + thresholds
# -----------------------------
st.sidebar.header("⚙️ Controls")

data_mode = st.sidebar.radio("Data source", ["Use synthetic demo data", "Upload CSV (invoice lines)"], index=0)

t = Thresholds(
    abs_usd=st.sidebar.slider("Min variance ($) to flag leakage", 0, 500, 50, 10),
    pct=st.sidebar.slider("Min variance (%) to flag leakage", 0.00, 0.50, 0.06, 0.01),
    duplicate_days_window=st.sidebar.slider("Duplicate detection window (days)", 1, 30, 7, 1),
)

st.sidebar.markdown("---")
show_notes = st.sidebar.checkbox("Show Notes / Real-life requirements", value=True)
show_raw = st.sidebar.checkbox("Show raw flagged lines table", value=True)


# -----------------------------
# 7) Load data
# -----------------------------
if data_mode == "Use synthetic demo data":
    df = make_synthetic_data(n=1300)
    df, warns = validate_and_clean(df)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to use real data, or switch back to synthetic demo data.")
        st.stop()
    df = pd.read_csv(uploaded)
    df, warns = validate_and_clean(df)

if warns:
    with st.expander("⚠️ Data warnings"):
        for w in warns:
            st.write(f"- {w}")

# Ensure required columns exist (for demo robustness)
missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_required:
    st.error(f"Your dataset is missing required columns: {missing_required}")
    st.stop()

# Fill optional columns if absent (to keep UI stable)
for col in OPTIONAL_COLS:
    if col not in df.columns:
        df[col] = np.nan
if "contract_missing" not in df.columns:
    df["contract_missing"] = False


# -----------------------------
# 8) Compute leakage + RCA
# -----------------------------
df2 = compute_leakage_flags(df, t)
df2 = apply_rca(df2, t)

flagged = df2[df2["is_leakage"]].copy()
flagged["leakage_amount"] = flagged["variance"].clip(lower=0)

# KPI summary
total_spend = df2["billed_amount"].sum()
total_leak = flagged["leakage_amount"].sum()
leak_rate = (total_leak / total_spend) if total_spend > 0 else 0.0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total billed spend", f"${total_spend:,.0f}")
kpi2.metric("Leakage identified", f"${total_leak:,.0f}")
kpi3.metric("Leakage rate", f"{leak_rate*100:.2f}%")
kpi4.metric("Flagged line items", f"{len(flagged):,}")


# -----------------------------
# 9) Views: Summary + Charts
# -----------------------------
left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("📌 Leakage by mode and root cause")
    if len(flagged) == 0:
        st.warning("No leakage flagged with current thresholds. Try lowering thresholds.")
    else:
        summary = (
            flagged.groupby(["mode", "root_cause"], as_index=False)["leakage_amount"]
            .sum()
            .sort_values("leakage_amount", ascending=False)
        )
        st.dataframe(summary, use_container_width=True, height=320)

with right:
    st.subheader("📊 Pareto of root causes (all modes)")
    if len(flagged) > 0:
        pareto = flagged.groupby("root_cause", as_index=False)["leakage_amount"].sum()
        pareto = pareto.sort_values("leakage_amount", ascending=False).head(10)

        fig = plt.figure()
        plt.bar(pareto["root_cause"], pareto["leakage_amount"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Leakage ($)")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("No flagged lines to chart.")


# -----------------------------
# 10) Tree structure graphic
# -----------------------------
st.subheader("🌳 RCA tree view (leadership-friendly)")

if len(flagged) > 0:
    tree_summary = (
        flagged.groupby(["mode", "root_cause"], as_index=False)["leakage_amount"].sum()
    )
    graph = build_rca_tree_graph(tree_summary, title="Freight Leakage RCA Tree")
    st.graphviz_chart(graph, use_container_width=True)
else:
    st.info("No leakage flagged — adjust thresholds to populate the RCA tree.")


# -----------------------------
# 11) Drill-down: Top leakage lines + explanation
# -----------------------------
st.subheader("🔎 Drill-down: top leakage lines with explainable RCA")

if len(flagged) > 0:
    topn = st.slider("How many lines to show", 10, 200, 30, 10)
    drill = flagged.sort_values("leakage_amount", ascending=False).head(topn).copy()

    # Display-friendly selection
    cols = [
        "shipment_id", "invoice_id", "mode", "lane", "carrier", "ship_date",
        "charge_category", "expected_amount", "billed_amount", "variance",
        "root_cause", "rca_explanation",
        "gross_kg", "vol_m3", "dwell_days", "free_time_days", "contract_missing",
        "is_duplicate",
    ]
    cols = [c for c in cols if c in drill.columns]

    st.dataframe(drill[cols], use_container_width=True, height=360)

    st.markdown("**Example dispute packet (one line):**")
    pick = drill.iloc[0]
    dispute_text = f"""
    Shipment {pick.get('shipment_id')} / Invoice {pick.get('invoice_id')}
    Mode: {pick.get('mode')} | Lane: {pick.get('lane')} | Carrier/Forwarder: {pick.get('carrier')}
    Charge: {pick.get('charge_category')}
    Expected: ${pick.get('expected_amount'):,.2f} | Billed: ${pick.get('billed_amount'):,.2f} | Variance: ${pick.get('variance'):,.2f}

    Root Cause: {pick.get('root_cause')}
    Rationale: {pick.get('rca_explanation')}
    """
    st.code(textwrap.dedent(dispute_text).strip())
else:
    st.info("No drill-down available without flagged lines.")


# -----------------------------
# 12) Notes: How RCA works + real-life needs
# -----------------------------
if show_notes:
    st.subheader("📝 Notes: How this RCA works (and what you need in real life)")

    st.markdown("""
### How the RCA works in this demo
This demo operates at the **invoice-line** level (where leakage actually hides).  
For each invoice line, it computes:

- **Variance ($)** = Billed − Expected  
- Flags a line as leakage if variance exceeds thresholds (min $ and/or %)
- Assigns a **root cause** using explainable rules:

**Ocean examples**
- **D&D Not Eligible / Free Time Misapplied**  
  Expected D&D = $0 (dwell ≤ free time) but D&D was billed.
- **D&D Days / Clock or Rate Mismatch**  
  Expected D&D exists, but variance is unusually large (clock start/stop errors, day-rate mismatch).

**Air examples**
- **Chargeable Weight / Dimensional Rating Issue**  
  Volumetric weight likely dominates gross (dims/divisor/rounding mismatch).
- **Service Level / Rating Mismatch**  
  Base/fuel billed higher than expected (wrong product/service level or rating logic).

**Cross-mode**
- **Duplicate Billing**
- **Contract / Rate Card Not Applied**
- **Surcharge / Accessorial Misapplied**

This is intentionally **audit-friendly**: every line has a plain-English explanation that an AP or logistics analyst can validate.

---

### What it needs for real life (3M-style)
To turn this into a production-grade model, you need 3 data layers:

#### 1) Invoice line items (the “truth”)
- invoice_id, shipment_id, charge code/category, quantity, unit rate, amount, currency
- billed party, invoice date, carrier/forwarder, lane, service level

#### 2) Shipment facts + events (to validate eligibility)
**Ocean**
- container milestones: discharge, availability, pickup, return empty (and terminal/port)
- free time terms and when the D&D clock starts/stops

**Air**
- AWB, pieces, dims, gross weight, chargeable weight, divisor rules
- tender/uplift/arrival/delivery timestamps (for storage/handling eligibility)

#### 3) Contract & surcharge logic (to compute “Expected”)
- contract rates by lane/service/equipment, effective dates
- fuel/security formulas, currency rules, accessorial eligibility rules
- free time terms by port/terminal/carrier and any special clauses

---

### Why invoice-line modeling matters
If you model only at invoice header or shipment header, you miss:
- duplicated **line items**
- accessorial code drift (THC/DOC/handling variants)
- partial credits and re-rated lines
- D&D billed on only some containers on a shipment

---

### What leadership will like
- Clear leakage number ($) + trend over time
- Pareto of top root causes
- Top offenders by carrier/forwarder/port/airport
- “Preventable vs non-preventable” split (next step)

---

### Next step upgrades (if you want to impress)
1) Replace the simplified “expected_amount” with a real expected-cost engine:
   - ocean: contract + surcharges + D&D clock engine
   - air: chargeable-weight calculator + product/service mapping
2) Add a “dispute packet generator” PDF/email template
3) Add supervised learning once you have labels (credited/disputed lines)
""")

    st.markdown("#### Expected CSV schema (minimum)")
    st.code(
        "\n".join(REQUIRED_COLS) +
        "\n\nOptional (helps RCA quality):\n" +
        "\n".join(OPTIONAL_COLS)
    )


# -----------------------------
# 13) Footer
# -----------------------------
st.caption("Tip: start with synthetic data for the demo; then swap in 3M invoice-line extracts + event milestones for a real pilot.")
