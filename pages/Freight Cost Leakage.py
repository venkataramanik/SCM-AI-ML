
# app.py
# ------------------------------------------------------------
# Ground Freight Invoice Simulation (Parcel / LTL / TL) — POC
# ------------------------------------------------------------
# Run:
#   pip install streamlit pandas numpy plotly matplotlib
#   streamlit run app.py
#
# What this app does:
# 1) Synthesizes realistic ground freight invoices for ONE plant (100 each: Parcel/LTL/TL)
# 2) Shows current-state metrics (cost by mode, shipment count, load factor, lane mix)
# 3) Runs a simulation (pooling + consolidation + routing) to show "what’s possible"
# 4) Emits a Root Cause Analysis (RCA) table + Ishikawa (Fishbone) diagram
# 5) Shows a simple "before vs after" lane map (visual)
#
# Notes:
# - This is a proof-of-concept. It intentionally uses INTERNAL proxies derived from invoice data
#   (e.g., TL $/mile from TL invoices; LTL $/lb from LTL invoices).
# - “Savings” are scenario-based and depend on business constraints (service levels, cutoffs, etc.).
# - We avoid external carrier rates. We use physics + simple cost rules + conservative uplifts.

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Ground Freight Invoice Simulation (POC)", layout="wide")
st.title("🚚 Ground Freight Invoice Simulation (Parcel • LTL • TL) — POC")


# -----------------------------
# 1) Short, crisp summary (for execs)
# -----------------------------
with st.expander("1) Why this simulation (short summary)", expanded=True):
    st.markdown(
        """
**Why this simulation exists (in one breath):**  
Freight invoices tell us what we *paid*. This simulation tests what *could have happened* if we applied simple, realistic shipping rules (pooling, consolidation, multi-stop),
so we can quantify **structural savings opportunities** and identify **why they’re not happening** (policy, cutoffs, receiver constraints, expedites, routing practices).

**What it is (and is not):**
- ✅ **It is** a “counterfactual” model: *same demand patterns, different execution rules.*
- ✅ **It is** carrier-agnostic: uses **internal proxies** derived from the invoices themselves.
- ❌ **It is not** a procurement / carrier-bid optimizer (needs bids/tenders for that).
        """.strip()
    )


# -----------------------------
# 2) Explain the logic (non-overly-technical language)
# -----------------------------
with st.expander("2) How the logic works (plain language)", expanded=True):
    st.markdown(
        """
### The big idea
We treat each invoice like a “shipment record” (date, origin, destination, mode, weight, paid amount).  
Then we ask: **If we pooled shipments within a reasonable time window, would fewer/larger moves be cheaper?**

### The simple physics + cost rules we use
**A) Load factor (capacity usage):**  
We compute a simple utilization measure:
- Parcel capacity proxy: **70 lb**
- LTL “effective capacity” proxy: **12,000 lb** (where TL often becomes more economical)
- TL capacity: **44,000 lb**

Load Factor = `weight / capacity` (a simple “how full was the move?”)

**B) Pricing behavior (no carrier rates required):**
- **TL behaves like $/mile** → we compute a TL $/mile proxy from TL invoices.
- **LTL behaves like $/lb (+handling)** → we compute an LTL $/lb proxy from LTL invoices.
- **Parcel behaves like minimum charges** → we compute Parcel $/lb proxy mainly for diagnostics.

**C) Conservative simulation (to avoid over-claiming):**
When we simulate a new move (like combining LTLs into one TL), we add a small conservative uplift (e.g., +5% to +10%) so savings aren’t “too good to be true.”

### What scenarios we run
- **Time-based consolidation:** group shipments on the same lane within X days (e.g., 3 days)
- **O-D lane consolidation (weekly):** same lane within 7 days
- **Parcel → LTL pooling:** combine many parcels into 1 LTL move (same lane, short window)
- **LTL → TL consolidation:** combine multiple LTLs into 1 TL move
- **Multi-stop routing:** combine multiple TL moves to the same metro within a short window into 1 multi-stop TL
        """.strip()
    )


# -----------------------------
# Data + simulation utilities
# -----------------------------
@dataclass
class Params:
    seed: int = 21
    origin: str = "PLANT-01"
    n_per_mode: int = 100

    # Simulation knobs
    time_window_days_ltl_to_tl: int = 3
    weekly_window_days: int = 7
    parcel_pool_window_days: int = 1
    multistop_window_days: int = 2
    max_multistops: int = 5

    # Thresholds
    parcel_to_ltl_min_lbs: float = 200
    ltl_to_tl_min_lbs: float = 12000

    # Conservative uplifts
    uplift_tl: float = 1.05
    uplift_ltl: float = 1.10
    uplift_multistop: float = 1.05


DESTS = [
    ("ATL-Metro", 380, 33.7490, -84.3880),
    ("CHI-Metro", 650, 41.8781, -87.6298),
    ("NYC-Metro", 900, 40.7128, -74.0060),
    ("DAL-Metro", 780, 32.7767, -96.7970),
    ("MIA-Metro", 660, 25.7617, -80.1918),
    ("DEN-Metro", 1400, 39.7392, -104.9903),
    ("SEA-Metro", 2500, 47.6062, -122.3321),
    ("PHX-Metro", 1750, 33.4484, -112.0740),
]

LANE_WEIGHTS = {
    "ATL-Metro": 0.22,
    "CHI-Metro": 0.18,
    "NYC-Metro": 0.17,
    "DAL-Metro": 0.12,
    "MIA-Metro": 0.10,
    "DEN-Metro": 0.09,
    "SEA-Metro": 0.07,
    "PHX-Metro": 0.05,
}

CAPACITY_LBS = {"Parcel": 70.0, "LTL": 12000.0, "TL": 44000.0}


def gen_accessorial(mode: str) -> List[str]:
    acc = []
    if mode in ["LTL", "TL"]:
        if random.random() < 0.20:
            acc.append("Appointment")
        if random.random() < 0.14:
            acc.append("Detention")
    if mode == "LTL" and random.random() < 0.12:
        acc.append("Reweigh")
    return acc


def infer_cbm_from_weight(weight_lbs: float, mode: str) -> float:
    # only for utilization-type metrics
    lb_per_cbm = {"Parcel": 140, "LTL": 190, "TL": 230}
    return max(0.05, weight_lbs / lb_per_cbm[mode])


def estimate_pallets(weight_lbs: float) -> int:
    # rough: 1 pallet ~ 1200 lbs
    return max(1, int(round(weight_lbs / 1200)))


def price_invoice(mode: str, miles: int, weight_lbs: float, accessorials: List[str], expedite_flag: bool) -> Tuple[float, float, float, float]:
    """
    Synthetic pricing shaped like real economics.
    """
    if mode == "Parcel":
        base = max(10, 0.95 * weight_lbs) + np.random.uniform(3, 18)  # min + handling
        base += 0.02 * miles  # zone effect
    elif mode == "LTL":
        base = 260 + 0.07 * miles + (0.32 * (weight_lbs ** 0.92))
    elif mode == "TL":
        base = np.random.uniform(1.85, 2.55) * miles
    else:
        base = 1000

    fuel = 0.12 * base

    acc_charge = 0.0
    for a in accessorials:
        if a == "Appointment":
            acc_charge += 75
        if a == "Detention":
            acc_charge += 240
        if a == "Reweigh":
            acc_charge += 65

    if expedite_flag:
        acc_charge += 0.18 * base  # expedite premium

    total = base + fuel + acc_charge
    return base, fuel, acc_charge, total


def synthesize_invoices(p: Params) -> pd.DataFrame:
    random.seed(p.seed)
    np.random.seed(p.seed)

    dest_choices = list(LANE_WEIGHTS.keys())
    dest_probs = np.array([LANE_WEIGHTS[d] for d in dest_choices], dtype=float)
    dest_probs = dest_probs / dest_probs.sum()

    miles_map = {d: m for d, m, _, _ in DESTS}

    start = datetime(2025, 11, 1)
    days_span = 30

    rows = []
    invoice_id = 1
    for mode in ["Parcel", "LTL", "TL"]:
        for _ in range(p.n_per_mode):
            dest = np.random.choice(dest_choices, p=dest_probs)
            miles = int(miles_map[dest])

            ship_date = start + timedelta(days=int(np.random.uniform(0, days_span)))
            ship_hour = int(
                np.random.choice(
                    [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                    p=[0.05, 0.08, 0.10, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10, 0.05, 0.02],
                )
            )
            ship_time = ship_date.replace(hour=ship_hour, minute=int(np.random.uniform(0, 60)))

            if mode == "Parcel":
                weight = float(np.random.uniform(2, 70))
            elif mode == "LTL":
                weight = float(np.random.gamma(shape=2.4, scale=1800)) + 350
                weight = float(min(max(weight, 450), 9000))
            else:
                weight = float(np.random.uniform(12000, 42000))

            expedite = (random.random() < (0.22 if dest in ["NYC-Metro", "CHI-Metro"] else 0.12))
            if mode == "TL" and random.random() < 0.65:
                expedite = False

            acc = gen_accessorial(mode)
            base, fuel, acc_charge, total = price_invoice(mode, miles, weight, acc, expedite)

            rows.append(
                {
                    "Invoice_ID": f"INV-{invoice_id:05d}",
                    "Plant": p.origin,
                    "Ship_Timestamp": ship_time,
                    "Ship_Date": ship_time.date(),
                    "Ship_Hour": ship_hour,
                    "Dest_Region": dest,
                    "Lane_ID": f"{p.origin}->{dest}",
                    "Miles": miles,
                    "Mode": mode,
                    "Weight_lbs": round(weight, 2),
                    "Volume_CBM": round(infer_cbm_from_weight(weight, mode), 2),
                    "Pallets_Est": estimate_pallets(weight),
                    "Expedite_Flag": bool(expedite),
                    "Accessorials": ",".join(acc) if acc else "",
                    "Base_Charge": round(base, 2),
                    "Fuel_Surcharge": round(fuel, 2),
                    "Accessorial_Charge": round(acc_charge, 2),
                    "Total_Paid": round(total, 2),
                }
            )
            invoice_id += 1

    df = pd.DataFrame(rows).sort_values("Ship_Timestamp").reset_index(drop=True)
    df["Capacity_lbs"] = df["Mode"].map(CAPACITY_LBS)
    df["Load_Factor"] = (df["Weight_lbs"] / df["Capacity_lbs"]).clip(upper=1.2)
    return df


def compute_internal_proxies(df: pd.DataFrame) -> Dict[str, float]:
    proxies: Dict[str, float] = {}

    tl = df[df["Mode"] == "TL"].copy()
    ltl = df[df["Mode"] == "LTL"].copy()
    parc = df[df["Mode"] == "Parcel"].copy()

    proxies["TL_per_mile"] = float((tl["Total_Paid"] / tl["Miles"]).median())
    proxies["LTL_per_lb"] = float((ltl["Total_Paid"] / ltl["Weight_lbs"]).median())
    proxies["Parcel_per_lb"] = float((parc["Total_Paid"] / parc["Weight_lbs"]).median())
    proxies["Stop_increment"] = 150.0  # POC default; tune with your data later
    return proxies


def infer_blockers(chunk: pd.DataFrame, lever_hint: str) -> List[str]:
    blockers: List[str] = []

    # invoice signals → likely blockers (ranked-ish)
    if chunk["Expedite_Flag"].mean() >= 0.25:
        blockers.append("SLA / expedite overrides (validate with requested delivery date)")

    if chunk["Accessorials"].str.contains("Appointment|Detention", regex=True).mean() >= 0.25:
        blockers.append("Receiver appointment windows / dock constraints")

    if (chunk["Ship_Hour"] >= 17).mean() >= 0.25:
        blockers.append("Late order cutoff / same-day release drives small shipments")

    ship_days = chunk["Ship_Date"].nunique()
    if ship_days >= 3 and len(chunk) >= 4:
        blockers.append("Daily ship cadence (policy) blocks pooling")

    # lever-specific structural blockers
    if "Parcel→LTL" in lever_hint:
        blockers.append("Parcel defaulting (no parcel-to-freight threshold policy)")
    if "LTL→TL" in lever_hint:
        blockers.append("No LTL-to-TL trigger / tendering optimization")
    if "Consolidation" in lever_hint:
        blockers.append("No consolidation rule / pooling visibility (TMS/ERP)")
    if "Routing" in lever_hint:
        blockers.append("No route optimization / dispatch practice")

    # de-dup preserve order
    seen = set()
    out = []
    for b in blockers:
        if b not in seen:
            out.append(b)
            seen.add(b)
    return out


def add_opportunity(opps: List[dict], lever: str, inv_ids: List[str], actual_cost: float, simulated_cost: float, notes: str, blockers: List[str]):
    savings = actual_cost - simulated_cost
    if savings <= 0:
        return
    opps.append(
        {
            "Lever": lever,
            "Invoice_Count": len(inv_ids),
            "Invoices": ",".join(inv_ids[:15]) + ("..." if len(inv_ids) > 15 else ""),
            "Actual_Cost": round(actual_cost, 2),
            "Simulated_Cost": round(simulated_cost, 2),
            "Savings": round(savings, 2),
            "Savings_%": round(100 * savings / actual_cost, 1),
            "Likely_Blockers": "; ".join(blockers) if blockers else "Unknown (needs order data)",
            "Notes": notes,
        }
    )


def run_simulation(df: pd.DataFrame, proxies: Dict[str, float], p: Params) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Waterfall-ish simulation: we propose opportunities, but ALSO build an "After" shipment set
    by applying transformations sequentially to reduce double counting.

    Steps:
    1) Pool parcels -> LTL (per lane, 0-1 day)
    2) Consolidate LTL -> TL (per lane, 0-3 days)
    3) Weekly lane consolidation (LTL -> TL) (optional extra catch)
    4) Multi-stop TL bundling (per destination region, 0-2 days)

    Returns:
    - opportunities table (what we changed, savings, blockers)
    - after_df (simulated shipment records)
    """
    work = df.copy().sort_values("Ship_Timestamp").reset_index(drop=True)
    opps: List[dict] = []

    # Helper: create a new simulated shipment row
    def make_row(mode: str, ship_time: datetime, dest_region: str, miles: int, weight_lbs: float, src_ids: List[str], notes: str) -> dict:
        lf = float(min(1.2, weight_lbs / CAPACITY_LBS[mode]))
        return {
            "Invoice_ID": "SIM-" + "-".join(src_ids[:3]) + ("..." if len(src_ids) > 3 else ""),
            "Plant": work["Plant"].iloc[0],
            "Ship_Timestamp": ship_time,
            "Ship_Date": ship_time.date(),
            "Ship_Hour": ship_time.hour,
            "Dest_Region": dest_region,
            "Lane_ID": f"{work['Plant'].iloc[0]}->{dest_region}",
            "Miles": miles,
            "Mode": mode,
            "Weight_lbs": round(weight_lbs, 2),
            "Volume_CBM": round(infer_cbm_from_weight(weight_lbs, mode), 2),
            "Pallets_Est": estimate_pallets(weight_lbs),
            "Expedite_Flag": False,
            "Accessorials": "",
            "Base_Charge": None,
            "Fuel_Surcharge": None,
            "Accessorial_Charge": None,
            "Total_Paid": None,  # filled after pricing
            "Capacity_lbs": CAPACITY_LBS[mode],
            "Load_Factor": lf,
            "Sim_Notes": notes,
        }

    # Pricing for simulated moves using proxies
    def price_simulated(mode: str, miles: int, weight_lbs: float, stops: int = 1) -> float:
        if mode == "TL":
            return float(proxies["TL_per_mile"] * miles * p.uplift_tl + (stops - 1) * proxies["Stop_increment"])
        if mode == "LTL":
            return float(max(350, proxies["LTL_per_lb"] * weight_lbs) * p.uplift_ltl)
        if mode == "Parcel":
            return float(proxies["Parcel_per_lb"] * weight_lbs)  # not used for sim typically
        raise ValueError("Unsupported mode")

    # ---------- Step 1: Parcel -> LTL pooling ----------
    after_rows = []
    remaining = work.copy()

    parcels = remaining[remaining["Mode"] == "Parcel"].copy()
    non_parcels = remaining[remaining["Mode"] != "Parcel"].copy()

    pooled_parcel_ids = set()

    for lane, g in parcels.groupby("Lane_ID"):
        g = g.sort_values("Ship_Timestamp").reset_index(drop=True)
        i = 0
        while i < len(g):
            sd = g.loc[i, "Ship_Date"]
            window_end = (pd.to_datetime(sd) + pd.Timedelta(days=p.parcel_pool_window_days)).date()
            chunk = g[(g["Ship_Date"] >= sd) & (g["Ship_Date"] <= window_end)]

            if len(chunk) >= 5 and float(chunk["Weight_lbs"].sum()) >= p.parcel_to_ltl_min_lbs:
                total_wt = float(chunk["Weight_lbs"].sum())
                actual = float(chunk["Total_Paid"].sum())
                miles = int(chunk["Miles"].median())
                dest = chunk["Dest_Region"].iloc[0]

                sim_cost = price_simulated("LTL", miles, total_wt)
                blockers = infer_blockers(chunk, "Parcel→LTL Consolidation")

                add_opportunity(
                    opps,
                    "Mode shift (Parcel→LTL via pooling)",
                    chunk["Invoice_ID"].tolist(),
                    actual,
                    sim_cost,
                    f"{lane}; {len(chunk)} parcels in {p.parcel_pool_window_days}d; total_wt={int(total_wt)} lbs",
                    blockers,
                )

                ship_time = chunk["Ship_Timestamp"].max()  # ship at the latest time in the pooled set
                r = make_row("LTL", ship_time, dest, miles, total_wt, chunk["Invoice_ID"].tolist(), "Pooled parcels → 1 LTL")
                r["Total_Paid"] = round(sim_cost, 2)
                after_rows.append(r)

                pooled_parcel_ids.update(chunk["Invoice_ID"].tolist())
                i += len(chunk)
            else:
                i += 1

    # Keep unpooled parcels as-is
    unpooled_parcels = parcels[~parcels["Invoice_ID"].isin(pooled_parcel_ids)].copy()
    after_df = pd.concat([non_parcels, unpooled_parcels, pd.DataFrame(after_rows)], ignore_index=True)
    after_df = after_df.sort_values("Ship_Timestamp").reset_index(drop=True)

    # ---------- Step 2: LTL -> TL consolidation (lane + time window) ----------
    work2 = after_df.copy()
    ltls = work2[work2["Mode"] == "LTL"].copy()
    others = work2[work2["Mode"] != "LTL"].copy()

    consolidated_ltl_ids = set()
    new_rows = []

    for lane, g in ltls.groupby("Lane_ID"):
        g = g.sort_values("Ship_Timestamp").reset_index(drop=True)
        i = 0
        while i < len(g):
            sd = g.loc[i, "Ship_Date"]
            window_end = (pd.to_datetime(sd) + pd.Timedelta(days=p.time_window_days_ltl_to_tl)).date()
            chunk = g[(g["Ship_Date"] >= sd) & (g["Ship_Date"] <= window_end)]

            if len(chunk) >= 3 and float(chunk["Weight_lbs"].sum()) >= p.ltl_to_tl_min_lbs:
                total_wt = float(chunk["Weight_lbs"].sum())
                actual = float(chunk["Total_Paid"].sum())
                miles = int(chunk["Miles"].median())
                dest = chunk["Dest_Region"].iloc[0]

                sim_cost = price_simulated("TL", miles, total_wt, stops=1)
                blockers = infer_blockers(chunk, "LTL→TL Consolidation")

                add_opportunity(
                    opps,
                    "Consolidation (LTL→TL by lane+time)",
                    chunk["Invoice_ID"].tolist(),
                    actual,
                    sim_cost,
                    f"{lane}; {len(chunk)} LTL in {p.time_window_days_ltl_to_tl}d; total_wt={int(total_wt)} lbs",
                    blockers,
                )

                ship_time = chunk["Ship_Timestamp"].max()
                r = make_row("TL", ship_time, dest, miles, total_wt, chunk["Invoice_ID"].tolist(), "Consolidated LTL → 1 TL")
                r["Total_Paid"] = round(sim_cost, 2)
                new_rows.append(r)

                consolidated_ltl_ids.update(chunk["Invoice_ID"].tolist())
                i += len(chunk)
            else:
                i += 1

    remaining_ltl = ltls[~ltls["Invoice_ID"].isin(consolidated_ltl_ids)].copy()
    after_df2 = pd.concat([others, remaining_ltl, pd.DataFrame(new_rows)], ignore_index=True)
    after_df2 = after_df2.sort_values("Ship_Timestamp").reset_index(drop=True)

    # ---------- Step 3: Weekly lane consolidation (catch remaining LTL) ----------
    work3 = after_df2.copy()
    ltls3 = work3[work3["Mode"] == "LTL"].copy()
    others3 = work3[work3["Mode"] != "LTL"].copy()

    consolidated_week_ids = set()
    new_rows2 = []

    for lane, g in ltls3.groupby("Lane_ID"):
        g = g.sort_values("Ship_Timestamp").reset_index(drop=True)
        i = 0
        while i < len(g):
            sd = g.loc[i, "Ship_Date"]
            window_end = (pd.to_datetime(sd) + pd.Timedelta(days=p.weekly_window_days)).date()
            chunk = g[(g["Ship_Date"] >= sd) & (g["Ship_Date"] <= window_end)]

            if len(chunk) >= 4 and float(chunk["Weight_lbs"].sum()) >= p.ltl_to_tl_min_lbs * 1.1:
                total_wt = float(chunk["Weight_lbs"].sum())
                actual = float(chunk["Total_Paid"].sum())
                miles = int(chunk["Miles"].median())
                dest = chunk["Dest_Region"].iloc[0]

                # slightly more conservative than lane+time
                sim_cost = float(proxies["TL_per_mile"] * miles * 1.07)
                blockers = infer_blockers(chunk, "LTL→TL Consolidation")

                add_opportunity(
                    opps,
                    "O-D pair consolidation (LTL→TL weekly)",
                    chunk["Invoice_ID"].tolist(),
                    actual,
                    sim_cost,
                    f"{lane}; {len(chunk)} LTL in {p.weekly_window_days}d; total_wt={int(total_wt)} lbs",
                    blockers,
                )

                ship_time = chunk["Ship_Timestamp"].max()
                r = make_row("TL", ship_time, dest, miles, total_wt, chunk["Invoice_ID"].tolist(), "Weekly consolidated LTL → 1 TL")
                r["Total_Paid"] = round(sim_cost, 2)
                new_rows2.append(r)

                consolidated_week_ids.update(chunk["Invoice_ID"].tolist())
                i += len(chunk)
            else:
                i += 1

    remaining_ltl3 = ltls3[~ltls3["Invoice_ID"].isin(consolidated_week_ids)].copy()
    after_df3 = pd.concat([others3, remaining_ltl3, pd.DataFrame(new_rows2)], ignore_index=True)
    after_df3 = after_df3.sort_values("Ship_Timestamp").reset_index(drop=True)

    # ---------- Step 4: Multi-stop TL bundling (destination region + time window) ----------
    work4 = after_df3.copy()
    tls = work4[work4["Mode"] == "TL"].copy()
    others4 = work4[work4["Mode"] != "TL"].copy()

    bundled_tl_ids = set()
    new_rows3 = []

    for dest, g in tls.groupby("Dest_Region"):
        g = g.sort_values("Ship_Timestamp").reset_index(drop=True)
        i = 0
        while i < len(g):
            sd = g.loc[i, "Ship_Date"]
            window_end = (pd.to_datetime(sd) + pd.Timedelta(days=p.multistop_window_days)).date()
            chunk = g[(g["Ship_Date"] >= sd) & (g["Ship_Date"] <= window_end)]

            if len(chunk) >= 3:
                actual = float(chunk["Total_Paid"].sum())
                miles = int(chunk["Miles"].median())
                total_wt = float(chunk["Weight_lbs"].sum())

                stops = min(len(chunk), p.max_multistops)
                sim_cost = float((proxies["TL_per_mile"] * miles + (stops - 1) * proxies["Stop_increment"]) * p.uplift_multistop)

                blockers = infer_blockers(chunk, "Routing Consolidation")

                add_opportunity(
                    opps,
                    "Routing (single TLs→multi-stop TL)",
                    chunk["Invoice_ID"].tolist(),
                    actual,
                    sim_cost,
                    f"{work4['Plant'].iloc[0]}->{dest}; {len(chunk)} TLs in {p.multistop_window_days}d; simulated {stops}-stop route",
                    blockers,
                )

                ship_time = chunk["Ship_Timestamp"].max()
                r = make_row("TL", ship_time, dest, miles, total_wt, chunk["Invoice_ID"].tolist(), f"Bundled {len(chunk)} TLs → 1 multi-stop TL (stops={stops})")
                r["Total_Paid"] = round(sim_cost, 2)
                new_rows3.append(r)

                bundled_tl_ids.update(chunk["Invoice_ID"].tolist())
                i += len(chunk)
            else:
                i += 1

    remaining_tls = tls[~tls["Invoice_ID"].isin(bundled_tl_ids)].copy()
    after_df = pd.concat([others4, remaining_tls, pd.DataFrame(new_rows3)], ignore_index=True)
    after_df = after_df.sort_values("Ship_Timestamp").reset_index(drop=True)

    opp_df = pd.DataFrame(opps).sort_values("Savings", ascending=False).reset_index(drop=True) if opps else pd.DataFrame()
    return opp_df, after_df


def summarize_by_mode(df: pd.DataFrame) -> pd.DataFrame:
    baseline_total = float(df["Total_Paid"].sum())
    out = (
        df.groupby("Mode", as_index=False)
        .agg(
            Shipments=("Invoice_ID", "count"),
            Total_Cost=("Total_Paid", "sum"),
            Avg_Cost=("Total_Paid", "mean"),
            Median_Cost=("Total_Paid", "median"),
            Total_Weight_lbs=("Weight_lbs", "sum"),
            Avg_Weight_lbs=("Weight_lbs", "mean"),
            Avg_Load_Factor=("Load_Factor", "mean"),
            Expedite_Rate=("Expedite_Flag", "mean"),
        )
        .sort_values("Total_Cost", ascending=False)
        .reset_index(drop=True)
    )
    out["Cost_Share_%"] = (100 * out["Total_Cost"] / baseline_total).round(2)
    out["Avg_Load_Factor"] = out["Avg_Load_Factor"].round(3)
    out["Expedite_Rate_%"] = (100 * out["Expedite_Rate"]).round(1)
    out.drop(columns=["Expedite_Rate"], inplace=True)
    return out


def lever_summary(opp_df: pd.DataFrame, baseline_total: float) -> pd.DataFrame:
    if opp_df is None or opp_df.empty:
        return pd.DataFrame()
    s = (
        opp_df.groupby("Lever", as_index=False)
        .agg(Opportunities=("Lever", "count"), Actual_Cost=("Actual_Cost", "sum"), Simulated_Cost=("Simulated_Cost", "sum"), Savings=("Savings", "sum"))
        .sort_values("Savings", ascending=False)
        .reset_index(drop=True)
    )
    s["Savings_%_of_Baseline"] = (100 * s["Savings"] / baseline_total).round(2)
    return s


def rca_pareto(opp_df: pd.DataFrame) -> pd.DataFrame:
    if opp_df is None or opp_df.empty:
        return pd.DataFrame()
    ser = opp_df["Likely_Blockers"].str.split("; ").explode()
    vc = ser.value_counts().reset_index()
    vc.columns = ["Root_Cause", "Count"]
    return vc


def blocker_to_category(blocker: str) -> str:
    b = blocker.lower()
    if "sla" in b or "expedite" in b:
        return "Service / Customer Promise"
    if "appointment" in b or "dock" in b or "detention" in b:
        return "Operations / Receiving Constraints"
    if "cutoff" in b:
        return "Planning / Order Cadence"
    if "tms" in b or "visibility" in b or "tendering" in b or "trigger" in b:
        return "Process / Systems"
    if "route optimization" in b or "dispatch" in b:
        return "Transportation Planning"
    if "policy" in b or "defaulting" in b:
        return "Policy / Governance"
    return "Other"


def ishikawa_fishbone(rca_df: pd.DataFrame):
    """
    Draw a simple Ishikawa diagram using matplotlib.
    We group RCA causes into categories and show top causes per category.
    """
    if rca_df is None or rca_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No RCA causes to display (run simulation).", ha="center", va="center")
        ax.axis("off")
        return fig

    tmp = rca_df.copy()
    tmp["Category"] = tmp["Root_Cause"].apply(blocker_to_category)

    # Take top 3 per category
    grouped = (
        tmp.sort_values("Count", ascending=False)
        .groupby("Category", as_index=False)
        .head(3)
    )

    categories = grouped["Category"].unique().tolist()
    # Split categories across top and bottom
    top_cats = categories[::2]
    bottom_cats = categories[1::2]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    # Spine
    ax.plot([0.05, 0.95], [0.5, 0.5], linewidth=2)
    ax.text(0.96, 0.5, "Missed Consolidation / Routing Opportunities", va="center", fontsize=12)

    def draw_branch(cat_list, y_dir: float):
        # y_dir = +1 for top, -1 for bottom
        x_positions = np.linspace(0.15, 0.85, num=max(2, len(cat_list)))
        for x, cat in zip(x_positions, cat_list):
            # main branch line
            ax.plot([x, x + 0.08], [0.5, 0.5 + 0.22 * y_dir], linewidth=1.5)
            ax.text(x + 0.085, 0.5 + 0.22 * y_dir, cat, va="center", fontsize=10)

            causes = grouped[grouped["Category"] == cat].sort_values("Count", ascending=False)
            # sub-branches
            y0 = 0.5 + 0.22 * y_dir
            for j, (_, row) in enumerate(causes.iterrows()):
                y = y0 + (0.07 * y_dir) * (j + 1)
                ax.plot([x + 0.03, x + 0.11], [y0, y], linewidth=1.0)
                label = f"- {row['Root_Cause']} ({int(row['Count'])})"
                ax.text(x + 0.115, y, label, va="center", fontsize=9)

    draw_branch(top_cats, +1)
    draw_branch(bottom_cats, -1)

    return fig


def lane_map(before_df: pd.DataFrame, after_df: pd.DataFrame, origin_lat: float, origin_lon: float):
    """
    Simple 'map' using Plotly Scattergeo.
    We draw lane lines where thickness reflects shipment count.
    """
    coords = {d: (lat, lon) for d, _, lat, lon in DESTS}

    def lane_lines(df: pd.DataFrame) -> List[go.Scattergeo]:
        agg = df.groupby("Dest_Region", as_index=False).agg(Shipments=("Invoice_ID", "count"), Total_Cost=("Total_Paid", "sum"))
        traces = []
        for _, r in agg.iterrows():
            dest = r["Dest_Region"]
            if dest not in coords:
                continue
            lat2, lon2 = coords[dest]
            width = max(1, min(10, int(round(r["Shipments"] / 6))))
            traces.append(
                go.Scattergeo(
                    lat=[origin_lat, lat2],
                    lon=[origin_lon, lon2],
                    mode="lines",
                    line=dict(width=width),
                    hoverinfo="text",
                    text=f"{dest}<br>Shipments: {int(r['Shipments'])}<br>Total: ${r['Total_Cost']:,.0f}",
                    showlegend=False,
                )
            )
        return traces

    fig = go.Figure()

    # Before
    for t in lane_lines(before_df):
        fig.add_trace(t)

    # markers
    fig.add_trace(
        go.Scattergeo(
            lat=[origin_lat],
            lon=[origin_lon],
            mode="markers+text",
            text=["PLANT-01"],
            textposition="bottom center",
            marker=dict(size=10),
            showlegend=False,
        )
    )

    dest_lats = []
    dest_lons = []
    dest_names = []
    for d, _, lat, lon in DESTS:
        dest_lats.append(lat)
        dest_lons.append(lon)
        dest_names.append(d)
    fig.add_trace(
        go.Scattergeo(
            lat=dest_lats,
            lon=dest_lons,
            mode="markers+text",
            text=dest_names,
            textposition="top center",
            marker=dict(size=6),
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Lane Map — BEFORE (line thickness ~ shipment count)",
        geo=dict(scope="usa", projection_type="albers usa", showland=True),
        margin=dict(l=10, r=10, t=40, b=10),
        height=520,
    )

    # After figure
    fig2 = go.Figure()
    for t in lane_lines(after_df):
        fig2.add_trace(t)
    fig2.add_trace(
        go.Scattergeo(
            lat=[origin_lat],
            lon=[origin_lon],
            mode="markers+text",
            text=["PLANT-01"],
            textposition="bottom center",
            marker=dict(size=10),
            showlegend=False,
        )
    )
    fig2.add_trace(
        go.Scattergeo(
            lat=dest_lats,
            lon=dest_lons,
            mode="markers+text",
            text=dest_names,
            textposition="top center",
            marker=dict(size=6),
            showlegend=False,
        )
    )
    fig2.update_layout(
        title="Lane Map — AFTER (line thickness ~ shipment count)",
        geo=dict(scope="usa", projection_type="albers usa", showland=True),
        margin=dict(l=10, r=10, t=40, b=10),
        height=520,
    )
    return fig, fig2


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("POC Controls")

seed = st.sidebar.number_input("Random seed", min_value=1, max_value=10_000, value=21, step=1)
n_per_mode = st.sidebar.slider("Invoices per mode", min_value=25, max_value=250, value=100, step=25)

st.sidebar.subheader("Simulation parameters")
time_window = st.sidebar.slider("LTL→TL consolidation window (days)", 1, 7, 3)
weekly_window = st.sidebar.slider("Weekly lane pooling window (days)", 3, 14, 7)
parcel_window = st.sidebar.slider("Parcel pooling window (days)", 0, 2, 1)
multistop_window = st.sidebar.slider("Multi-stop bundling window (days)", 1, 5, 2)

parcel_min_lbs = st.sidebar.number_input("Parcel→LTL minimum pooled lbs", min_value=50.0, max_value=1000.0, value=200.0, step=50.0)
ltl_min_lbs = st.sidebar.number_input("LTL→TL minimum pooled lbs", min_value=5000.0, max_value=20000.0, value=12000.0, step=1000.0)

uplift_tl = st.sidebar.slider("Conservative uplift for TL sims", 1.00, 1.20, 1.05, 0.01)
uplift_ltl = st.sidebar.slider("Conservative uplift for LTL sims", 1.00, 1.30, 1.10, 0.01)
uplift_ms = st.sidebar.slider("Conservative uplift for multi-stop sims", 1.00, 1.20, 1.05, 0.01)

max_stops = st.sidebar.slider("Max stops for simulated multi-stop TL", 3, 8, 5)

st.sidebar.subheader("Data input")
uploaded = st.sidebar.file_uploader("Optional: Upload your invoice CSV (same columns or close)", type=["csv"])
use_uploaded = st.sidebar.checkbox("Use uploaded CSV (if provided)", value=False)

p = Params(
    seed=int(seed),
    n_per_mode=int(n_per_mode),
    time_window_days_ltl_to_tl=int(time_window),
    weekly_window_days=int(weekly_window),
    parcel_pool_window_days=int(parcel_window),
    multistop_window_days=int(multistop_window),
    parcel_to_ltl_min_lbs=float(parcel_min_lbs),
    ltl_to_tl_min_lbs=float(ltl_min_lbs),
    uplift_tl=float(uplift_tl),
    uplift_ltl=float(uplift_ltl),
    uplift_multistop=float(uplift_ms),
    max_multistops=int(max_stops),
)


# -----------------------------
# Data load (synthetic or uploaded)
# -----------------------------
def coerce_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to map common invoice CSV column names to the expected schema.
    If you upload your own, you'll likely tweak this mapping.
    """
    df = df_in.copy()
    rename_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["invoice", "invoice_id", "inv_id"]:
            rename_map[c] = "Invoice_ID"
        elif cl in ["ship_date", "date"]:
            rename_map[c] = "Ship_Date"
        elif cl in ["ship_timestamp", "ship_datetime", "timestamp", "datetime"]:
            rename_map[c] = "Ship_Timestamp"
        elif cl in ["dest", "destination", "dest_region", "destination_region"]:
            rename_map[c] = "Dest_Region"
        elif cl in ["mode", "ship_mode", "transport_mode"]:
            rename_map[c] = "Mode"
        elif cl in ["weight", "weight_lbs", "lbs"]:
            rename_map[c] = "Weight_lbs"
        elif cl in ["miles", "distance", "distance_miles"]:
            rename_map[c] = "Miles"
        elif cl in ["total", "total_paid", "amount", "invoice_total"]:
            rename_map[c] = "Total_Paid"
        elif cl in ["accessorials", "accessorial"]:
            rename_map[c] = "Accessorials"
        elif cl in ["expedite", "expedite_flag"]:
            rename_map[c] = "Expedite_Flag"
    df = df.rename(columns=rename_map)

    # Required minimal fields for analysis
    required = ["Invoice_ID", "Mode", "Weight_lbs", "Miles", "Total_Paid", "Dest_Region"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV missing required columns: {missing}")

    # Create plant / timestamp if missing
    if "Plant" not in df.columns:
        df["Plant"] = p.origin
    if "Ship_Timestamp" not in df.columns:
        # build from Ship_Date with noon
        df["Ship_Timestamp"] = pd.to_datetime(df["Ship_Date"]) + pd.Timedelta(hours=12)
    else:
        df["Ship_Timestamp"] = pd.to_datetime(df["Ship_Timestamp"])

    df["Ship_Date"] = pd.to_datetime(df["Ship_Timestamp"]).dt.date
    df["Ship_Hour"] = pd.to_datetime(df["Ship_Timestamp"]).dt.hour
    df["Lane_ID"] = df["Plant"].astype(str) + "->" + df["Dest_Region"].astype(str)

    if "Volume_CBM" not in df.columns:
        # infer cbm by mode (rough)
        df["Volume_CBM"] = df.apply(lambda r: infer_cbm_from_weight(float(r["Weight_lbs"]), str(r["Mode"])), axis=1).round(2)

    if "Pallets_Est" not in df.columns:
        df["Pallets_Est"] = df["Weight_lbs"].apply(lambda w: estimate_pallets(float(w)))

    if "Accessorials" not in df.columns:
        df["Accessorials"] = ""

    if "Expedite_Flag" not in df.columns:
        df["Expedite_Flag"] = False

    df["Capacity_lbs"] = df["Mode"].map(CAPACITY_LBS).fillna(12000.0)
    df["Load_Factor"] = (df["Weight_lbs"] / df["Capacity_lbs"]).clip(upper=1.2)

    # Ensure numeric
    df["Weight_lbs"] = pd.to_numeric(df["Weight_lbs"], errors="coerce")
    df["Miles"] = pd.to_numeric(df["Miles"], errors="coerce")
    df["Total_Paid"] = pd.to_numeric(df["Total_Paid"], errors="coerce")

    df = df.dropna(subset=["Weight_lbs", "Miles", "Total_Paid"]).reset_index(drop=True)
    return df


if use_uploaded and uploaded is not None:
    try:
        df = coerce_columns(pd.read_csv(uploaded))
        st.success(f"Loaded uploaded CSV: {len(df):,} rows.")
    except Exception as e:
        st.error(f"Failed to load uploaded CSV: {e}")
        df = synthesize_invoices(p)
else:
    df = synthesize_invoices(p)

proxies = compute_internal_proxies(df)
opp_df, after_df = run_simulation(df, proxies, p)

baseline_total = float(df["Total_Paid"].sum())
after_total = float(after_df["Total_Paid"].sum()) if "Total_Paid" in after_df.columns else None
savings = baseline_total - after_total


# -----------------------------
# 3) Show the current invoices
# -----------------------------
st.header("3) Current invoices (baseline)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Plant", p.origin)
c2.metric("Invoices (baseline)", f"{len(df):,}")
c3.metric("Total paid (baseline)", f"${baseline_total:,.0f}")
c4.metric("TL $/mile proxy (internal)", f"${proxies['TL_per_mile']:.2f}")

st.subheader("Baseline — cost by mode / shipments / load factor")
st.dataframe(summarize_by_mode(df), use_container_width=True)

with st.expander("View current invoice lines (first 50)", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

st.download_button(
    "Download synthetic baseline invoices (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="baseline_invoices_ground_poc.csv",
    mime="text/csv",
)

# -----------------------------
# 4) Show after simulation
# -----------------------------
st.header("4) After simulation (what’s possible under the chosen rules)")

a1, a2, a3, a4 = st.columns(4)
a1.metric("Invoices (after)", f"{len(after_df):,}")
a2.metric("Total paid (after)", f"${after_total:,.0f}")
a3.metric("Modeled savings (after vs baseline)", f"${savings:,.0f}")
a4.metric("Modeled savings %", f"{100*savings/baseline_total:.1f}%")

st.subheader("After — cost by mode / shipments / load factor")
st.dataframe(summarize_by_mode(after_df), use_container_width=True)

with st.expander("View simulated shipment lines (first 50)", expanded=False):
    st.dataframe(after_df.head(50), use_container_width=True)

st.download_button(
    "Download simulated after-shipments (CSV)",
    data=after_df.to_csv(index=False).encode("utf-8"),
    file_name="after_simulation_shipments_ground_poc.csv",
    mime="text/csv",
)

# -----------------------------
# Opportunity + RCA section
# -----------------------------
st.header("5) Opportunity & Root Cause Analysis (RCA)")

if opp_df is None or opp_df.empty:
    st.info("No opportunities found under current parameters. Try widening windows or lowering thresholds.")
else:
    st.subheader("Savings opportunities (top 25)")
    st.dataframe(opp_df.head(25), use_container_width=True)

    st.subheader("Savings by lever")
    st.dataframe(lever_summary(opp_df, baseline_total), use_container_width=True)

    st.subheader("RCA Pareto (why it’s not happening)")
    rca_df = rca_pareto(opp_df)
    st.dataframe(rca_df, use_container_width=True)

    # Ishikawa diagram
    st.subheader("Ishikawa (Fishbone) — likely root causes")
    fig = ishikawa_fishbone(rca_df)
    st.pyplot(fig, clear_figure=True)

# -----------------------------
# Before/After Map
# -----------------------------
st.header("6) Map — before vs after (lane visualization)")

# Rough plant location (you can change this to your actual plant coordinates)
PLANT_LAT, PLANT_LON = 36.1627, -86.7816  # Nashville-ish (neutral)

fig_before, fig_after = lane_map(df, after_df, PLANT_LAT, PLANT_LON)
m1, m2 = st.columns(2)
with m1:
    st.plotly_chart(fig_before, use_container_width=True)
with m2:
    st.plotly_chart(fig_after, use_container_width=True)

# -----------------------------
# Wrap-up: how to use this POC
# -----------------------------
with st.expander("How to use this POC with real PDFs (next step)", expanded=False):
    st.markdown(
        """
If your real invoices are PDFs, the next step is simply an **extraction layer** that produces the same table columns
(`Ship_Date/Time, Origin, Dest, Mode, Weight, Miles or ZIPs, Total_Paid, Accessorials`).  
Once you have that table, this simulation engine runs unchanged.

Typical extraction tools:
- If PDFs are text-based: `pdfplumber` (tables)  
- If scanned: OCR (Azure Form Recognizer / AWS Textract / Tesseract)  
        """.strip()
    )
```
