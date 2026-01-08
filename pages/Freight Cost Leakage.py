# -*- coding: utf-8 -*-
# app.py (POC v3)
# Ground Freight Invoice Simulation (Parcel / LTL / TL) - POC
#
# Key features:
# - Synthetic invoices (default: 1000 per mode => 3000 total)
# - Baseline vs After metrics
# - Lane delta table (Before vs After shipments/cost)
# - Map BEFORE vs AFTER (same fixed width scale -> visible change)
# - RCA Pareto + Ishikawa (fishbone) diagram
#
# Run:
#   pip install streamlit pandas numpy plotly matplotlib
#   streamlit run app.py

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

st.set_page_config(page_title="Ground Freight Invoice Simulation (POC)", layout="wide")
st.title("Ground Freight Invoice Simulation (Parcel / LTL / TL) - POC")

with st.expander("Why this simulation (short)", expanded=True):
    st.markdown(
        """
Invoices show what we paid. This simulation tests what could have happened if we applied simple, realistic
shipping rules (pooling, consolidation, multi-stop). It outputs:
1) structural savings opportunities, and
2) likely reasons they are not happening (policy, cutoffs, receiver constraints, expedites, routing practice).

Carrier-agnostic: proxies come from invoice data itself. Not a bid/tender optimizer.
        """.strip()
    )

@dataclass
class Params:
    seed: int = 21
    origin: str = "PLANT-01"
    n_per_mode: int = 1000  # <-- default 1000 per mode => 3000 total

    parcel_pool_days: int = 1
    ltl_to_tl_days: int = 3
    multistop_days: int = 2
    max_stops: int = 5

    parcel_to_ltl_min_lbs: float = 200.0
    ltl_to_tl_min_lbs: float = 12000.0

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
LANE_WEIGHTS = {"ATL-Metro":0.22,"CHI-Metro":0.18,"NYC-Metro":0.17,"DAL-Metro":0.12,"MIA-Metro":0.10,"DEN-Metro":0.09,"SEA-Metro":0.07,"PHX-Metro":0.05}
CAPACITY_LBS = {"Parcel":70.0, "LTL":12000.0, "TL":44000.0}

def gen_accessorial(mode: str) -> List[str]:
    acc: List[str] = []
    if mode in ["LTL","TL"]:
        if random.random() < 0.20: acc.append("Appointment")
        if random.random() < 0.14: acc.append("Detention")
    if mode == "LTL" and random.random() < 0.12:
        acc.append("Reweigh")
    return acc

def infer_cbm_from_weight(weight_lbs: float, mode: str) -> float:
    lb_per_cbm = {"Parcel":140, "LTL":190, "TL":230}
    return max(0.05, weight_lbs / lb_per_cbm[mode])

def estimate_pallets(weight_lbs: float) -> int:
    return max(1, int(round(weight_lbs / 1200)))

def price_invoice(mode: str, miles: int, weight_lbs: float, accessorials: List[str], expedite: bool) -> float:
    if mode == "Parcel":
        base = max(10, 0.95*weight_lbs) + np.random.uniform(3,18)
        base += 0.02*miles
    elif mode == "LTL":
        base = 260 + 0.07*miles + (0.32*(weight_lbs**0.92))
    else:  # TL
        base = np.random.uniform(1.85,2.55)*miles
    fuel = 0.12*base
    acc = 0.0
    for a in accessorials:
        if a == "Appointment": acc += 75
        if a == "Detention": acc += 240
        if a == "Reweigh": acc += 65
    if expedite:
        acc += 0.18*base
    return float(base + fuel + acc)

def synthesize(p: Params) -> pd.DataFrame:
    random.seed(p.seed)
    np.random.seed(p.seed)
    dests = list(LANE_WEIGHTS.keys())
    probs = np.array([LANE_WEIGHTS[d] for d in dests], dtype=float)
    probs = probs / probs.sum()
    miles_map = {d:m for d,m,_,_ in DESTS}

    start = datetime(2025,11,1)
    rows = []
    inv = 1
    for mode in ["Parcel","LTL","TL"]:
        for _ in range(p.n_per_mode):
            dest = np.random.choice(dests, p=probs)
            miles = int(miles_map[dest])

            ship_date = start + timedelta(days=int(np.random.uniform(0,30)))
            ship_hour = int(np.random.choice([9,10,11,12,13,14,15,16,17,18,19],
                                             p=[0.05,0.08,0.10,0.12,0.12,0.12,0.12,0.12,0.10,0.05,0.02]))
            ship_time = ship_date.replace(hour=ship_hour, minute=int(np.random.uniform(0,60)))

            if mode == "Parcel":
                w = float(np.random.uniform(2,70))
            elif mode == "LTL":
                w = float(np.random.gamma(shape=2.4, scale=1800)) + 350
                w = float(min(max(w,450),9000))
            else:
                w = float(np.random.uniform(12000,42000))

            expedite = (random.random() < (0.22 if dest in ["NYC-Metro","CHI-Metro"] else 0.12))
            if mode == "TL" and random.random() < 0.65:
                expedite = False

            acc = gen_accessorial(mode)
            total = price_invoice(mode, miles, w, acc, expedite)

            rows.append({
                "Invoice_ID": f"INV-{inv:06d}",
                "Plant": p.origin,
                "Ship_Timestamp": ship_time,
                "Ship_Date": ship_time.date(),
                "Ship_Hour": ship_hour,
                "Dest_Region": dest,
                "Lane_ID": f"{p.origin}->{dest}",
                "Miles": miles,
                "Mode": mode,
                "Weight_lbs": round(w,2),
                "Volume_CBM": round(infer_cbm_from_weight(w, mode),2),
                "Pallets_Est": estimate_pallets(w),
                "Expedite_Flag": bool(expedite),
                "Accessorials": ",".join(acc) if acc else "",
                "Total_Paid": round(total,2),
            })
            inv += 1

    df = pd.DataFrame(rows).sort_values("Ship_Timestamp").reset_index(drop=True)
    df["Capacity_lbs"] = df["Mode"].map(CAPACITY_LBS)
    df["Load_Factor"] = (df["Weight_lbs"]/df["Capacity_lbs"]).clip(upper=1.2)
    return df

def proxies(df: pd.DataFrame) -> Dict[str,float]:
    tl = df[df["Mode"]=="TL"].copy()
    ltl = df[df["Mode"]=="LTL"].copy()
    parc = df[df["Mode"]=="Parcel"].copy()
    return {
        "TL_per_mile": float((tl["Total_Paid"]/tl["Miles"]).median()),
        "LTL_per_lb": float((ltl["Total_Paid"]/ltl["Weight_lbs"]).median()),
        "Parcel_per_lb": float((parc["Total_Paid"]/parc["Weight_lbs"]).median()),
        "Stop_increment": 150.0,
    }

def summarize_by_mode(df: pd.DataFrame) -> pd.DataFrame:
    total = float(df["Total_Paid"].sum())
    out = (df.groupby("Mode", as_index=False)
           .agg(Shipments=("Invoice_ID","count"),
                Total_Cost=("Total_Paid","sum"),
                Avg_Cost=("Total_Paid","mean"),
                Median_Cost=("Total_Paid","median"),
                Total_Weight_lbs=("Weight_lbs","sum"),
                Avg_Weight_lbs=("Weight_lbs","mean"),
                Avg_Load_Factor=("Load_Factor","mean"),
                Expedite_Rate=("Expedite_Flag","mean"))
           .sort_values("Total_Cost", ascending=False)
           .reset_index(drop=True))
    out["Cost_Share_%"] = (100*out["Total_Cost"]/total).round(2)
    out["Avg_Load_Factor"] = out["Avg_Load_Factor"].round(3)
    out["Expedite_Rate_%"] = (100*out["Expedite_Rate"]).round(1)
    out = out.drop(columns=["Expedite_Rate"])
    return out

def infer_blockers(chunk: pd.DataFrame) -> List[str]:
    b = []
    if chunk["Expedite_Flag"].mean() >= 0.25:
        b.append("SLA / expedite overrides (validate with requested delivery date)")
    if chunk["Accessorials"].str.contains("Appointment|Detention", regex=True).mean() >= 0.25:
        b.append("Receiver appointment windows / dock constraints")
    if (chunk["Ship_Hour"] >= 17).mean() >= 0.25:
        b.append("Late order cutoff / same-day release drives small shipments")
    ship_days = chunk["Ship_Date"].nunique()
    if ship_days >= 3 and len(chunk) >= 6:
        b.append("Daily ship cadence (policy) blocks pooling")
    return b

def rca_pareto(opp_df: pd.DataFrame) -> pd.DataFrame:
    if opp_df.empty:
        return pd.DataFrame()
    ser = opp_df["Likely_Blockers"].str.split("; ").explode()
    vc = ser.value_counts().reset_index()
    vc.columns = ["Root_Cause","Count"]
    return vc

def blocker_to_category(blocker: str) -> str:
    bl = blocker.lower()
    if "sla" in bl or "expedite" in bl:
        return "Service / Customer Promise"
    if "appointment" in bl or "dock" in bl or "detention" in bl:
        return "Operations / Receiving"
    if "cutoff" in bl or "cadence" in bl:
        return "Planning / Order Cadence"
    if "tender" in bl or "trigger" in bl or "tms" in bl or "visibility" in bl:
        return "Process / Systems"
    if "route" in bl or "dispatch" in bl:
        return "Transportation Planning"
    return "Other"

def ishikawa(rca_df: pd.DataFrame):
    if rca_df.empty:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.text(0.5,0.5,"No RCA to display.", ha="center", va="center")
        ax.axis("off")
        return fig

    tmp = rca_df.copy()
    tmp["Category"] = tmp["Root_Cause"].apply(blocker_to_category)
    grouped = tmp.sort_values("Count", ascending=False).groupby("Category", as_index=False).head(3)

    cats = grouped["Category"].unique().tolist()
    top = cats[::2]
    bottom = cats[1::2]

    fig, ax = plt.subplots(figsize=(14,6))
    ax.axis("off")
    ax.plot([0.05,0.95],[0.5,0.5], linewidth=2)
    ax.text(0.96,0.5,"Missed consolidation / routing opportunities", va="center", fontsize=12)

    def draw(cat_list, ydir):
        xs = np.linspace(0.15,0.85, num=max(2,len(cat_list)))
        for x, cat in zip(xs, cat_list):
            ax.plot([x, x+0.08],[0.5, 0.5+0.22*ydir], linewidth=1.5)
            ax.text(x+0.085, 0.5+0.22*ydir, cat, va="center", fontsize=10)
            causes = grouped[grouped["Category"]==cat].sort_values("Count", ascending=False)
            y0 = 0.5+0.22*ydir
            for j, (_, r) in enumerate(causes.iterrows()):
                y = y0 + (0.07*ydir)*(j+1)
                ax.plot([x+0.03, x+0.11],[y0, y], linewidth=1.0)
                ax.text(x+0.115, y, f"- {r['Root_Cause']} ({int(r['Count'])})", va="center", fontsize=9)

    draw(top, +1)
    draw(bottom, -1)
    return fig

def lane_agg(df_in: pd.DataFrame) -> pd.DataFrame:
    return (df_in.groupby("Dest_Region", as_index=False)
            .agg(Shipments=("Invoice_ID","count"),
                 Total_Cost=("Total_Paid","sum"),
                 Total_Weight=("Weight_lbs","sum")))

def lane_map(df_in: pd.DataFrame, title: str, width_scale_base: int):
    coords = {d:(lat,lon) for d,_,lat,lon in DESTS}
    origin_lat, origin_lon = 36.1627, -86.7816

    agg = lane_agg(df_in)
    base = max(1, int(width_scale_base))

    fig = go.Figure()
    for _, r in agg.iterrows():
        dest = r["Dest_Region"]
        if dest not in coords:
            continue
        lat2, lon2 = coords[dest]
        width = 1.0 + 11.0 * math.sqrt(float(r["Shipments"]) / base)
        fig.add_trace(go.Scattergeo(
            lat=[origin_lat, lat2],
            lon=[origin_lon, lon2],
            mode="lines",
            line=dict(width=width),
            hoverinfo="text",
            text=f"{dest}<br>Shipments: {int(r['Shipments'])}<br>Total: ${r['Total_Cost']:,.0f}",
            showlegend=False,
        ))

    fig.add_trace(go.Scattergeo(
        lat=[origin_lat], lon=[origin_lon],
        mode="markers+text",
        text=["PLANT-01"], textposition="bottom center",
        marker=dict(size=10),
        showlegend=False,
    ))

    dest_lats, dest_lons, dest_names = [], [], []
    for d,_,lat,lon in DESTS:
        dest_lats.append(lat); dest_lons.append(lon); dest_names.append(d)
    fig.add_trace(go.Scattergeo(
        lat=dest_lats, lon=dest_lons,
        mode="markers+text",
        text=dest_names, textposition="top center",
        marker=dict(size=6),
        showlegend=False,
    ))

    fig.update_layout(title=title, geo=dict(scope="usa", projection_type="albers usa", showland=True),
                      margin=dict(l=10,r=10,t=40,b=10), height=520)
    return fig

def run_sim(df: pd.DataFrame, pr: Dict[str,float], p: Params) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy().sort_values("Ship_Timestamp").reset_index(drop=True)
    opps = []

    def price_sim(mode: str, miles: int, weight: float, stops: int=1) -> float:
        if mode == "LTL":
            return float(max(350, pr["LTL_per_lb"]*weight) * p.uplift_ltl)
        if mode == "TL":
            return float(pr["TL_per_mile"]*miles*p.uplift_tl + (stops-1)*pr["Stop_increment"])
        raise ValueError("mode")

    # Step 1: Parcel -> LTL pooling
    parcels = work[work["Mode"]=="Parcel"].copy()
    non_parcels = work[work["Mode"]!="Parcel"].copy()

    pooled = set()
    new_rows = []

    for lane, g in parcels.groupby("Lane_ID"):
        g = g.sort_values("Ship_Timestamp").reset_index(drop=True)
        i = 0
        while i < len(g):
            sd = g.loc[i,"Ship_Date"]
            end = (pd.to_datetime(sd) + pd.Timedelta(days=p.parcel_pool_days)).date()
            chunk = g[(g["Ship_Date"]>=sd) & (g["Ship_Date"]<=end)]
            total_wt = float(chunk["Weight_lbs"].sum())

            if len(chunk) >= 8 and total_wt >= p.parcel_to_ltl_min_lbs:
                actual = float(chunk["Total_Paid"].sum())
                miles = int(chunk["Miles"].median())
                dest = chunk["Dest_Region"].iloc[0]
                sim_cost = price_sim("LTL", miles, total_wt, 1)
                blockers = infer_blockers(chunk) + ["Parcel defaulting (no parcel-to-freight threshold policy)"]
                opps.append({"Lever":"Mode shift (Parcel to LTL via pooling)","Invoice_Count":len(chunk),
                             "Actual_Cost":round(actual,2),"Simulated_Cost":round(sim_cost,2),
                             "Savings":round(actual-sim_cost,2),"Likely_Blockers":"; ".join(blockers),
                             "Notes":f"{lane}; parcels pooled={len(chunk)}; total_wt={int(total_wt)} lb"})
                ship_time = chunk["Ship_Timestamp"].max()
                new_rows.append({"Invoice_ID":"SIM-PAR-"+str(hash(tuple(chunk["Invoice_ID"].tolist()))%1_000_000),
                                 "Plant":p.origin,"Ship_Timestamp":ship_time,"Ship_Date":ship_time.date(),"Ship_Hour":ship_time.hour,
                                 "Dest_Region":dest,"Lane_ID":lane,"Miles":miles,"Mode":"LTL",
                                 "Weight_lbs":round(total_wt,2),"Volume_CBM":round(infer_cbm_from_weight(total_wt,"LTL"),2),
                                 "Pallets_Est":estimate_pallets(total_wt),"Expedite_Flag":False,"Accessorials":"",
                                 "Total_Paid":round(sim_cost,2)})
                pooled.update(chunk["Invoice_ID"].tolist())
                i += len(chunk)
            else:
                i += 1

    unpooled = parcels[~parcels["Invoice_ID"].isin(pooled)].copy()
    after = pd.concat([non_parcels, unpooled, pd.DataFrame(new_rows)], ignore_index=True).sort_values("Ship_Timestamp").reset_index(drop=True)

    # Step 2: LTL -> TL consolidation (lane, X days)
    ltls = after[after["Mode"]=="LTL"].copy()
    non_ltl = after[after["Mode"]!="LTL"].copy()

    used = set()
    new_rows2 = []

    for lane, g in ltls.groupby("Lane_ID"):
        g = g.sort_values("Ship_Timestamp").reset_index(drop=True)
        i = 0
        while i < len(g):
            sd = g.loc[i,"Ship_Date"]
            end = (pd.to_datetime(sd) + pd.Timedelta(days=p.ltl_to_tl_days)).date()
            chunk = g[(g["Ship_Date"]>=sd) & (g["Ship_Date"]<=end)]
            total_wt = float(chunk["Weight_lbs"].sum())

            if len(chunk) >= 3 and total_wt >= p.ltl_to_tl_min_lbs:
                actual = float(chunk["Total_Paid"].sum())
                miles = int(chunk["Miles"].median())
                dest = chunk["Dest_Region"].iloc[0]
                sim_cost = price_sim("TL", miles, total_wt, 1)
                blockers = infer_blockers(chunk) + ["No LTL-to-TL trigger / tendering optimization", "No consolidation rule / pooling visibility (TMS/ERP)"]
                opps.append({"Lever":"Consolidation (LTL to TL by lane+time)","Invoice_Count":len(chunk),
                             "Actual_Cost":round(actual,2),"Simulated_Cost":round(sim_cost,2),
                             "Savings":round(actual-sim_cost,2),"Likely_Blockers":"; ".join(blockers),
                             "Notes":f"{lane}; LTL consolidated={len(chunk)}; total_wt={int(total_wt)} lb"})
                ship_time = chunk["Ship_Timestamp"].max()
                new_rows2.append({"Invoice_ID":"SIM-LTL-"+str(hash(tuple(chunk["Invoice_ID"].tolist()))%1_000_000),
                                  "Plant":p.origin,"Ship_Timestamp":ship_time,"Ship_Date":ship_time.date(),"Ship_Hour":ship_time.hour,
                                  "Dest_Region":dest,"Lane_ID":lane,"Miles":miles,"Mode":"TL",
                                  "Weight_lbs":round(total_wt,2),"Volume_CBM":round(infer_cbm_from_weight(total_wt,"TL"),2),
                                  "Pallets_Est":estimate_pallets(total_wt),"Expedite_Flag":False,"Accessorials":"",
                                  "Total_Paid":round(sim_cost,2)})
                used.update(chunk["Invoice_ID"].tolist())
                i += len(chunk)
            else:
                i += 1

    remaining = ltls[~ltls["Invoice_ID"].isin(used)].copy()
    after2 = pd.concat([non_ltl, remaining, pd.DataFrame(new_rows2)], ignore_index=True).sort_values("Ship_Timestamp").reset_index(drop=True)

    # Step 3: Multi-stop TL (dest region, 0-2 days)
    tls = after2[after2["Mode"]=="TL"].copy()
    non_tl = after2[after2["Mode"]!="TL"].copy()

    used_tl = set()
    new_rows3 = []

    for dest, g in tls.groupby("Dest_Region"):
        g = g.sort_values("Ship_Timestamp").reset_index(drop=True)
        i = 0
        while i < len(g):
            sd = g.loc[i,"Ship_Date"]
            end = (pd.to_datetime(sd) + pd.Timedelta(days=p.multistop_days)).date()
            chunk = g[(g["Ship_Date"]>=sd) & (g["Ship_Date"]<=end)]
            if len(chunk) >= 3:
                actual = float(chunk["Total_Paid"].sum())
                miles = int(chunk["Miles"].median())
                total_wt = float(chunk["Weight_lbs"].sum())
                stops = min(len(chunk), p.max_stops)
                sim_cost = float((pr["TL_per_mile"]*miles + (stops-1)*pr["Stop_increment"]) * p.uplift_multistop)
                blockers = infer_blockers(chunk) + ["No route optimization / dispatch practice"]
                opps.append({"Lever":"Routing (single TLs to multi-stop TL)","Invoice_Count":len(chunk),
                             "Actual_Cost":round(actual,2),"Simulated_Cost":round(sim_cost,2),
                             "Savings":round(actual-sim_cost,2),"Likely_Blockers":"; ".join(blockers),
                             "Notes":f"{p.origin}->{dest}; TL bundled={len(chunk)}; stops={stops}"})
                ship_time = chunk["Ship_Timestamp"].max()
                new_rows3.append({"Invoice_ID":"SIM-RT-"+str(hash(tuple(chunk["Invoice_ID"].tolist()))%1_000_000),
                                  "Plant":p.origin,"Ship_Timestamp":ship_time,"Ship_Date":ship_time.date(),"Ship_Hour":ship_time.hour,
                                  "Dest_Region":dest,"Lane_ID":f"{p.origin}->{dest}","Miles":miles,"Mode":"TL",
                                  "Weight_lbs":round(total_wt,2),"Volume_CBM":round(infer_cbm_from_weight(total_wt,"TL"),2),
                                  "Pallets_Est":estimate_pallets(total_wt),"Expedite_Flag":False,"Accessorials":"",
                                  "Total_Paid":round(sim_cost,2)})
                used_tl.update(chunk["Invoice_ID"].tolist())
                i += len(chunk)
            else:
                i += 1

    remaining_tl = tls[~tls["Invoice_ID"].isin(used_tl)].copy()
    after3 = pd.concat([non_tl, remaining_tl, pd.DataFrame(new_rows3)], ignore_index=True).sort_values("Ship_Timestamp").reset_index(drop=True)

    after3["Capacity_lbs"] = after3["Mode"].map(CAPACITY_LBS)
    after3["Load_Factor"] = (after3["Weight_lbs"]/after3["Capacity_lbs"]).clip(upper=1.2)

    opp_df = pd.DataFrame(opps)
    if not opp_df.empty:
        opp_df = opp_df.sort_values("Savings", ascending=False).reset_index(drop=True)
    return opp_df, after3

# ------------------ Sidebar controls ------------------
st.sidebar.header("Controls")
seed = st.sidebar.number_input("Seed", 1, 10000, 21)
n_per_mode = st.sidebar.number_input("Invoices per mode (default 1000)", 100, 5000, 1000, 100)

parcel_pool_days = st.sidebar.slider("Parcel pooling window (days)", 0, 2, 1)
ltl_to_tl_days = st.sidebar.slider("LTL to TL window (days)", 1, 7, 3)
multistop_days = st.sidebar.slider("Multi-stop window (days)", 1, 5, 2)
parcel_min_lbs = st.sidebar.number_input("Parcel to LTL min pooled lbs", 50.0, 2000.0, 200.0, 50.0)
ltl_min_lbs = st.sidebar.number_input("LTL to TL min pooled lbs", 5000.0, 25000.0, 12000.0, 1000.0)
uplift_tl = st.sidebar.slider("TL uplift", 1.00, 1.30, 1.05, 0.01)
uplift_ltl = st.sidebar.slider("LTL uplift", 1.00, 1.50, 1.10, 0.01)
uplift_ms = st.sidebar.slider("Multi-stop uplift", 1.00, 1.30, 1.05, 0.01)
max_stops = st.sidebar.slider("Max stops", 3, 10, 5)

p = Params(
    seed=int(seed), n_per_mode=int(n_per_mode),
    parcel_pool_days=int(parcel_pool_days),
    ltl_to_tl_days=int(ltl_to_tl_days),
    multistop_days=int(multistop_days),
    max_stops=int(max_stops),
    parcel_to_ltl_min_lbs=float(parcel_min_lbs),
    ltl_to_tl_min_lbs=float(ltl_min_lbs),
    uplift_tl=float(uplift_tl),
    uplift_ltl=float(uplift_ltl),
    uplift_multistop=float(uplift_ms),
)

df = synthesize(p)
pr = proxies(df)
opp_df, after_df = run_sim(df, pr, p)

baseline_total = float(df["Total_Paid"].sum())
after_total = float(after_df["Total_Paid"].sum())
savings = baseline_total - after_total

# ------------------ Headline ------------------
st.info(f"Data volume: {p.n_per_mode:,} invoices per mode => {3*p.n_per_mode:,} total invoices (Parcel+LTL+TL).")

st.header("Baseline vs After (headline)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Invoices (baseline)", f"{len(df):,}")
k2.metric("Invoices (after)", f"{len(after_df):,}")
k3.metric("Baseline total", f"${baseline_total:,.0f}")
k4.metric("After total", f"${after_total:,.0f}")
st.metric("Modeled savings", f"${savings:,.0f}", f"{100*savings/baseline_total:.1f}%")

# ------------------ Tables ------------------
st.subheader("Baseline by mode")
st.dataframe(summarize_by_mode(df), use_container_width=True)
st.subheader("After by mode")
st.dataframe(summarize_by_mode(after_df), use_container_width=True)

# ------------------ Lane delta ------------------
st.header("Lane delta (shipments and cost) - BEFORE vs AFTER")
before_lane = lane_agg(df).rename(columns={"Shipments":"Shipments_Before","Total_Cost":"Cost_Before","Total_Weight":"Weight_Before"})
after_lane = lane_agg(after_df).rename(columns={"Shipments":"Shipments_After","Total_Cost":"Cost_After","Total_Weight":"Weight_After"})
delta = before_lane.merge(after_lane, on="Dest_Region", how="outer").fillna(0)
delta["Shipment_Delta"] = delta["Shipments_After"] - delta["Shipments_Before"]
delta["Cost_Delta"] = delta["Cost_After"] - delta["Cost_Before"]
delta = delta.sort_values("Shipment_Delta").reset_index(drop=True)
st.dataframe(delta, use_container_width=True)

# ------------------ RCA + Ishikawa ------------------
st.header("Opportunities and RCA")
if opp_df.empty:
    st.info("No opportunities under current settings (try widening windows / lowering thresholds).")
else:
    st.subheader("Top opportunities (first 25)")
    st.dataframe(opp_df.head(25), use_container_width=True)

    st.subheader("RCA Pareto")
    rca_df = rca_pareto(opp_df)
    st.dataframe(rca_df, use_container_width=True)

    st.subheader("Ishikawa diagram (fishbone)")
    st.pyplot(ishikawa(rca_df), clear_figure=True)

# ------------------ Maps (same scaling) ------------------
st.header("Map (before vs after) - same width scale")
before_max = int(lane_agg(df)["Shipments"].max())
m1, m2 = st.columns(2)
with m1:
    st.plotly_chart(lane_map(df, "BEFORE (width scale fixed to BEFORE max shipments)", before_max), use_container_width=True)
with m2:
    st.plotly_chart(lane_map(after_df, "AFTER (same width scale -> should look thinner)", before_max), use_container_width=True)

st.caption("If the map still looks similar, the Lane delta table above shows the real shipment reductions by destination.")
