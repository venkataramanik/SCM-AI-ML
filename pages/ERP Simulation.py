# app.py
# Streamlit SCOR ERP Simulator (Plan–Source–Make–Deliver–Return–Enable)
# Drop this file next to a folder named `SCOR_Sim_v1` containing the CSVs listed below.
# Run:  streamlit run app.py

import os
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="SCOR ERP Simulator", layout="wide")

# ---------- Utilities ----------
@st.cache_data
def load_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)

def ensure_paths(base_path):
    needed = [
        "items.csv","boms.csv","work_centers.csv","routings.csv",
        "suppliers.csv","transport_lanes.csv","policies.csv",
        "inventories.csv","demand_forecast.csv","customer_orders.csv",
        "kpi_targets.csv"
    ]
    missing = [f for f in needed if not os.path.exists(os.path.join(base_path,f))]
    return missing

# ---------- Simulation Core ----------
def place_po(item_id, qty, today, inbound, lane_choice="auto"):
    # Simple mapping of suppliers and lanes
    if item_id == "RM-10":
        supplier_id = "SUP-IN-01"; lane = "IN-US-Ocean"; lt=14
    elif item_id == "RM-20":
        supplier_id = "SUP-IN-02"; lane = "IN-US-Ocean"; lt=18
    else:
        supplier_id = "SUP-CN-01"; lane = "CN-US-Ocean"; lt=20

    # Transit times (ocean vs air)
    if lane_choice == "air":
        transit = 5 if "CN" in lane else 6
    else:
        transit = 28 if "CN" in lane else 30

    arrival = today + pd.Timedelta(days=lt + transit)
    inbound.append((item_id, int(qty), arrival))
    return inbound

def run_sim(
    base_path,
    horizon_days=90,
    demand_surge_pct=0.0,
    supplier_delay_days=0,
    expedite_threshold=0.0,
    apply_capacity=False,
    asm_capacity=800,
    test_capacity=700,
    ss_multiplier=1.0,
    random_seed=42
):
    np.random.seed(random_seed)

    items = load_csv(os.path.join(base_path,"items.csv"))
    boms = load_csv(os.path.join(base_path,"boms.csv"))
    work_centers = load_csv(os.path.join(base_path,"work_centers.csv"))
    routings = load_csv(os.path.join(base_path,"routings.csv"))
    suppliers = load_csv(os.path.join(base_path,"suppliers.csv"))
    transport_lanes = load_csv(os.path.join(base_path,"transport_lanes.csv"))
    policies = load_csv(os.path.join(base_path,"policies.csv")).set_index("item_id").to_dict("index")
    inventories_df = load_csv(os.path.join(base_path,"inventories.csv"))
    demand_forecast = load_csv(os.path.join(base_path,"demand_forecast.csv"), parse_dates=["date"])
    orders = load_csv(os.path.join(base_path,"customer_orders.csv"), parse_dates=["date"]).copy()

    # Demand surge
    if demand_surge_pct != 0.0:
        factor = 1.0 + demand_surge_pct/100.0
        orders["order_qty"] = (orders["order_qty"] * factor).round().astype(int)

    # Due dates and shipment fields
    orders["due_date"] = orders["date"] + pd.to_timedelta(
        np.random.choice([3,5,7], size=len(orders), p=[0.4,0.4,0.2]), unit="D"
    )
    orders["shipped_qty"] = 0
    orders.loc[:, "ship_date"] = pd.NaT

    # Inventory dict
    inv = inventories_df.set_index(["item_id","location"])["on_hand"].to_dict()

    # Simple cost model
    sales_price = {"FG-100":120.0, "FG-200":140.0}
    unit_cost_rm = {"RM-10":10.0, "RM-20":5.0, "RM-30":20.0}
    logistics_cost = 0.0
    revenue = 0.0
    cogs = 0.0

    # Inbound pipeline
    inbound = []

    # Timeline
    start_date = orders["date"].min()
    end_date = start_date + pd.Timedelta(days=horizon_days)

    # Capacity per WC (hrs/day)
    wc_caps = {"WC-ASM": asm_capacity, "WC-TEST": test_capacity}

    kpi_records = []

    # Precompute routing hours per unit
    rt_map = routings.groupby("item_id").apply(lambda g: g["std_hours_per_unit"].sum()).to_dict()

    supplier_delay_days = int(supplier_delay_days)

    while start_date < end_date:
        # Receive inbound (apply supplier delay)
        received_today = [x for x in inbound if (x[2] + pd.Timedelta(days=supplier_delay_days)).date() == start_date.date()]
        for (itm, q, _) in received_today:
            inv[(itm,"MAIN")] = inv.get((itm,"MAIN"), 0) + q
        inbound = [x for x in inbound if x not in received_today]

        # PLAN: RM reorder
        for rm in ["RM-10","RM-20","RM-30"]:
            lvl = inv.get((rm,"MAIN"), 0)
            pol = policies.get(rm, {})
            rpoint = pol.get("reorder_point", 0)
            ss = pol.get("safety_stock", 0) * ss_multiplier
            if lvl < rpoint:
                qty = int(ss + rpoint - lvl + 1000)
                inbound = place_po(rm, qty, start_date, inbound, lane_choice="auto")
                logistics_cost += 0.05*qty  # placeholder

        # MAKE: produce to match today's orders (simple build-to-order), optionally capacity-bound
        todays_orders = orders.loc[orders["date"] == start_date.date()].copy()
        rem_hours = (
            {"WC-ASM": wc_caps["WC-ASM"], "WC-TEST": wc_caps["WC-TEST"]}
            if apply_capacity else {"WC-ASM": 1e9, "WC-TEST": 1e9}
        )

        for _, so in todays_orders.iterrows():
            fg = so["item_id"]
            qty = int(so["order_qty"])

            # Max producible due to RM
            bom_rows = boms.loc[boms["parent_item_id"] == fg]
            can_make = qty
            for _, b in bom_rows.iterrows():
                on_hand = inv.get((b["component_item_id"],"MAIN"), 0)
                can_make = min(can_make, int(on_hand // b["qty_per"]))

            # Capacity constraint
            if apply_capacity:
                rt_fg = routings[routings["item_id"] == fg]
                total_std = rt_fg["std_hours_per_unit"].sum()
                prod_qty_cap = can_make
                for _, r in rt_fg.iterrows():
                    wc = r["wc_id"]
                    cap_units = int(rem_hours[wc] / (r["std_hours_per_unit"] if r["std_hours_per_unit"] > 0 else 1e9))
                    prod_qty_cap = min(prod_qty_cap, cap_units)
                prod_qty = max(0, min(can_make, prod_qty_cap))
                # Deduct hours
                for _, r in rt_fg.iterrows():
                    wc = r["wc_id"]
                    rem_hours[wc] -= prod_qty * r["std_hours_per_unit"]
                    rem_hours[wc] = max(0.0, rem_hours[wc])
            else:
                prod_qty = can_make

            # Consume RM and add FG
            for _, b in bom_rows.iterrows():
                inv[(b["component_item_id"],"MAIN")] = inv.get((b["component_item_id"],"MAIN"), 0) - int(b["qty_per"]*prod_qty)
            inv[(fg,"MAIN")] = inv.get((fg,"MAIN"), 0) + prod_qty

        # DELIVER: ship open orders by due date
        open_orders = orders[(orders["shipped_qty"] < orders["order_qty"]) & (orders["due_date"] >= start_date)].copy()
        for idx, so in open_orders.iterrows():
            fg = so["item_id"]
            need = int(so["order_qty"] - so["shipped_qty"])
            available = inv.get((fg,"MAIN"), 0)
            ship = min(need, available)
            if ship > 0:
                inv[(fg,"MAIN")] = available - ship
                orders.at[idx, "shipped_qty"] += ship
                orders.at[idx, "ship_date"] = start_date
                revenue += ship * sales_price.get(fg, 100.0)
                # COGS from RM
                bom_rows = boms.loc[boms["parent_item_id"] == fg]
                unit_cogs = sum(unit_cost_rm.get(r["component_item_id"], 0) * r["qty_per"] for _, r in bom_rows.iterrows())
                cogs += ship * unit_cogs

        # Simple expedite trigger: if backlog ratio >= threshold, speed up future inbound (simulate ocean→air)
        backlog_orders = orders[(orders["shipped_qty"] < orders["order_qty"]) & (orders["due_date"] >= start_date)]
        backlog_ratio = 0.0
        if len(backlog_orders) > 0:
            demand_left = (backlog_orders["order_qty"] - backlog_orders["shipped_qty"]).sum()
            fg_on_hand = sum(inv.get((k,"MAIN"), 0) for k in ["FG-100","FG-200"])
            backlog_ratio = max(0.0, (demand_left - fg_on_hand) / max(1, demand
