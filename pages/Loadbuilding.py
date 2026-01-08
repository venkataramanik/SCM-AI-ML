# app.py
# Complete "Rate + Route + Plan" POC with business rules (synthetic)
#
# pip install streamlit pandas numpy plotly matplotlib ortools
# streamlit run app.py

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# OR-Tools is optional at import time; app will warn if missing
try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ORTOOLS_OK = True
except Exception:
    ORTOOLS_OK = False

# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="Rate + Route + Plan (Synthetic TMS-lite)", layout="wide")
st.title("🚚 Rate + Route + Plan — Synthetic TMS-lite (Parcel + LTL + TL + Multi-stop + Rules)")

st.caption(
    "This is a realistic POC: synthetic orders + synthetic rates + business rules + OR-Tools VRP for TL multi-stop routing."
)

# -----------------------------
# Network: Sites + Dest zones
# -----------------------------
SITES = [
    ("PLANT-01", 36.1627, -86.7816),   # Nashville-ish
    ("PLANT-02", 34.0522, -118.2437),  # LA-ish
    ("PLANT-03", 41.2565, -95.9345),   # Omaha-ish
]
DESTS = [
    ("ATL", 380, 33.7490, -84.3880),
    ("CHI", 650, 41.8781, -87.6298),
    ("NYC", 900, 40.7128, -74.0060),
    ("DAL", 780, 32.7767, -96.7970),
    ("MIA", 660, 25.7617, -80.1918),
    ("DEN", 1400, 39.7392, -104.9903),
    ("SEA", 2500, 47.6062, -122.3321),
    ("PHX", 1750, 33.4484, -112.0740),
]

DEST_WEIGHTS = {"ATL": 0.18, "CHI": 0.17, "NYC": 0.18, "DAL": 0.12, "MIA": 0.10, "DEN": 0.10, "SEA": 0.08, "PHX": 0.07}

# -----------------------------
# Rating primitives
# -----------------------------
LTL_CLASS_FACTOR = {
    50: 0.85, 55: 0.92, 60: 1.00, 65: 1.08, 70: 1.16, 77.5: 1.28, 85: 1.42,
    92.5: 1.60, 100: 1.80, 110: 2.05, 125: 2.40, 150: 3.00,
}
CAPACITY_LBS = {"Parcel": 70.0, "LTL": 12000.0, "TL": 44000.0}

# Trailer cube ~ 3,500 ft3 typical; use configurable later
DEFAULT_TL_CUBE_FT3 = 3500.0

def distance_zone(miles: float) -> str:
    if miles <= 250:
        return "Z1"
    if miles <= 750:
        return "Z2"
    if miles <= 1500:
        return "Z3"
    return "Z4"

def coords_for_site(site: str) -> Tuple[float, float]:
    for s, lat, lon in SITES:
        if s == site:
            return lat, lon
    return SITES[0][1], SITES[0][2]

def coords_for_dest(dest: str) -> Tuple[float, float]:
    for d, _, lat, lon in DESTS:
        if d == dest:
            return lat, lon
    return DESTS[0][2], DESTS[0][3]

def miles_for_dest(dest: str) -> int:
    for d, miles, _, _ in DESTS:
        if d == dest:
            return int(miles)
    return 1000

def ft3_from_weight(weight_lbs: float, density_lb_per_ft3: float) -> float:
    return weight_lbs / max(1e-6, density_lb_per_ft3)

# -----------------------------
# Synthetic generators
# -----------------------------
def synthesize_orders(
    n_orders: int,
    days_span: int,
    n_customers: int,
    seed: int,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    start = date(2025, 11, 1)

    customers = [f"CUST-{i+1:03d}" for i in range(n_customers)]
    site_ids = [s for s, _, _ in SITES]
    dests = list(DEST_WEIGHTS.keys())
    probs = np.array([DEST_WEIGHTS[d] for d in dests], dtype=float)
    probs = probs / probs.sum()

    rows = []
    for i in range(n_orders):
        origin = random.choice(site_ids)
        dest = str(np.random.choice(dests, p=probs))
        miles = miles_for_dest(dest)
        ship_dt = (start + timedelta(days=int(np.random.uniform(0, days_span))))
        # Ready date can be same day or +1 (late release simulation)
        ready_offset = int(np.random.choice([0, 0, 0, 1, 1, 2], p=[0.40, 0.0, 0.0, 0.40, 0.0, 0.20]))
        ready_dt = ship_dt + timedelta(days=ready_offset)

        # Due date: 1-7 days after ship_dt
        due_days = int(np.random.choice([1,2,3,4,5,6,7], p=[0.10,0.18,0.20,0.18,0.14,0.12,0.08]))
        due_dt = ship_dt + timedelta(days=due_days)

        # Weight distribution: many small/medium, some heavy
        wt = float(np.random.gamma(shape=2.2, scale=900)) + 20
        wt = float(min(max(wt, 5), 20000))

        # Density and cube
        density = float(np.random.uniform(8, 14))  # lb/ft3
        cube_ft3 = float(max(1.0, ft3_from_weight(wt, density)))
        cbm = cube_ft3 / 35.3147

        pallets = max(1, int(round(wt / 1200)))

        expedite = bool(random.random() < (0.20 if dest in ["NYC", "CHI"] else 0.12))
        priority = "EXPEDITE" if expedite else "STANDARD"

        ltl_class = float(np.random.choice([55,60,65,70,77.5,85,92.5,100,110,125], p=[0.06,0.18,0.18,0.16,0.12,0.10,0.08,0.07,0.03,0.02]))

        compat_group = str(np.random.choice(["NORMAL", "HAZMAT", "CHILLED", "FRAGILE"], p=[0.80, 0.06, 0.08, 0.06]))

        rows.append({
            "order_id": f"ORD-{i+1:06d}",
            "origin": origin,
            "dest_zone": dest,
            "miles": miles,
            "ship_date": ship_dt,
            "ready_date": ready_dt,
            "due_date": due_dt,
            "customer_id": random.choice(customers),
            "priority": priority,
            "expedite_flag": expedite,
            "weight_lbs": round(wt, 2),
            "cube_ft3": round(cube_ft3, 2),
            "volume_cbm": round(cbm, 3),
            "pallets": pallets,
            "ltl_class": ltl_class,
            "compat_group": compat_group,
        })

    df = pd.DataFrame(rows).sort_values(["origin","dest_zone","ship_date"]).reset_index(drop=True)
    return df

def synthesize_rates(seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(seed)
    np.random.seed(seed)

    # Parcel: zone + weight breaks
    parcel_breaks = [(0,10),(10,30),(30,70)]
    parcel_rows = []
    for origin, _, _ in SITES:
        for dest, miles, _, _ in DESTS:
            z = distance_zone(miles)
            base_per_lb = {"Z1":0.90,"Z2":1.05,"Z3":1.20,"Z4":1.40}[z]
            minc = {"Z1":8,"Z2":10,"Z3":12,"Z4":14}[z]
            for lo, hi in parcel_breaks:
                adj = 1.10 if hi<=10 else (1.00 if hi<=30 else 0.92)
                parcel_rows.append({
                    "origin": origin,
                    "dest_zone": dest,
                    "dist_zone": z,
                    "break_lo_lbs": lo,
                    "break_hi_lbs": hi,
                    "rate_per_lb": round(base_per_lb*adj, 3),
                    "min_charge": float(minc),
                    "fixed_charge": 0.0,
                })
    parcel = pd.DataFrame(parcel_rows)

    # LTL: dist_zone + weight breaks; class factor applied later
    ltl_breaks = [(0,500),(500,2000),(2000,5000),(5000,12000),(12000,20000)]
    ltl_rows = []
    for origin, _, _ in SITES:
        for z in ["Z1","Z2","Z3","Z4"]:
            base60 = {"Z1":0.20,"Z2":0.26,"Z3":0.34,"Z4":0.44}[z]
            minc = {"Z1":220,"Z2":280,"Z3":340,"Z4":420}[z]
            for lo, hi in ltl_breaks:
                if hi <= 500:
                    w_adj = 1.40
                elif hi <= 2000:
                    w_adj = 1.15
                elif hi <= 5000:
                    w_adj = 1.00
                elif hi <= 12000:
                    w_adj = 0.88
                else:
                    w_adj = 0.82
                ltl_rows.append({
                    "origin": origin,
                    "dist_zone": z,
                    "break_lo_lbs": lo,
                    "break_hi_lbs": hi,
                    "base_rate_per_lb_class60": round(base60*w_adj, 4),
                    "min_charge": float(minc),
                    "fixed_charge": 0.0,
                })
    ltl = pd.DataFrame(ltl_rows)

    # TL: $/mile + fixed + stop charge by lane
    tl_rows = []
    for origin, _, _ in SITES:
        for dest, miles, _, _ in DESTS:
            z = distance_zone(miles)
            per_mile = {"Z1":1.85,"Z2":2.05,"Z3":2.25,"Z4":2.45}[z]
            tl_rows.append({
                "origin": origin,
                "dest_zone": dest,
                "dist_zone": z,
                "miles": miles,
                "rate_per_mile": float(per_mile),
                "fixed_charge": 120.0,
                "stop_charge": 150.0,
            })
    tl = pd.DataFrame(tl_rows)
    return parcel, ltl, tl

# -----------------------------
# Rating functions
# -----------------------------
def rate_parcel(order: pd.Series, parcel: pd.DataFrame) -> float:
    o = str(order["origin"]); d = str(order["dest_zone"])
    wt = float(order["weight_lbs"])
    sub = parcel[(parcel.origin==o) & (parcel.dest_zone==d)]
    br = sub[(sub.break_lo_lbs < wt) & (wt <= sub.break_hi_lbs)].head(1)
    if br.empty:
        br = sub.sort_values("break_hi_lbs").tail(1)
    r = float(br.rate_per_lb.iloc[0])
    minc = float(br.min_charge.iloc[0])
    return float(max(minc, wt*r + float(br.fixed_charge.iloc[0])))

def rate_ltl(total_wt: float, origin: str, miles: float, ltl_class: float, expedite: bool, ltl: pd.DataFrame) -> float:
    z = distance_zone(miles)
    sub = ltl[(ltl.origin==origin) & (ltl.dist_zone==z)]
    br = sub[(sub.break_lo_lbs < total_wt) & (total_wt <= sub.break_hi_lbs)].head(1)
    if br.empty:
        br = sub.sort_values("break_hi_lbs").tail(1)
    base60 = float(br.base_rate_per_lb_class60.iloc[0])
    minc = float(br.min_charge.iloc[0])
    class_factor = float(LTL_CLASS_FACTOR.get(float(ltl_class), 1.0))
    cost = max(minc, total_wt*base60*class_factor + float(br.fixed_charge.iloc[0]))
    if expedite:
        cost *= 1.18
    return float(cost)

def rate_tl(origin: str, dest: str, stops: int, expedite: bool, tl: pd.DataFrame) -> float:
    row = tl[(tl.origin==origin) & (tl.dest_zone==dest)].head(1)
    if row.empty:
        # fallback to zone
        miles = miles_for_dest(dest)
        z = distance_zone(miles)
        row = tl[(tl.origin==origin) & (tl.dist_zone==z)].head(1)
    miles = float(row.miles.iloc[0])
    per_mile = float(row.rate_per_mile.iloc[0])
    fixed = float(row.fixed_charge.iloc[0])
    stop_charge = float(row.stop_charge.iloc[0])
    cost = miles*per_mile + fixed + max(0, stops-1)*stop_charge
    if expedite:
        cost *= 1.12
    return float(cost)

# -----------------------------
# Business rules engine
# -----------------------------
# Rules supported:
# - FORCE_MODE: customer must ship Parcel/LTL/TL
# - SHIP_ALONE: customer/order cannot consolidate with others
# - DO_NOT_SHIP_WITH_GROUP: compat group cannot mix with another group (e.g., HAZMAT not with CHILLED)
# - MAX_STOPS: per customer max stops for TL route
# - NO_POOLING: per customer no pooling across days

def default_rules(seed: int) -> pd.DataFrame:
    random.seed(seed)
    # Example: a few customers force LTL, some ship alone, plus incompatibilities
    rules = []
    # Global incompatibilities
    rules += [
        {"rule_type":"DO_NOT_SHIP_WITH_GROUP","key":"HAZMAT","value":"CHILLED"},
        {"rule_type":"DO_NOT_SHIP_WITH_GROUP","key":"HAZMAT","value":"FRAGILE"},
        {"rule_type":"DO_NOT_SHIP_WITH_GROUP","key":"CHILLED","value":"FRAGILE"},
    ]
    # Customer-specific
    for i in range(1, 6):
        rules.append({"rule_type":"FORCE_MODE","key":f"CUST-{i:03d}","value":"LTL"})
    for i in range(6, 9):
        rules.append({"rule_type":"SHIP_ALONE","key":f"CUST-{i:03d}","value":"TRUE"})
    for i in range(9, 12):
        rules.append({"rule_type":"MAX_STOPS","key":f"CUST-{i:03d}","value":"2"})
    for i in range(12, 15):
        rules.append({"rule_type":"NO_POOLING","key":f"CUST-{i:03d}","value":"TRUE"})
    return pd.DataFrame(rules)

def parse_rules(rules_df: pd.DataFrame) -> dict:
    d = {"FORCE_MODE":{}, "SHIP_ALONE":set(), "MAX_STOPS":{}, "NO_POOLING":set(), "DONT_MIX":set()}
    for _, r in rules_df.iterrows():
        rt = str(r["rule_type"]).strip().upper()
        key = str(r["key"]).strip()
        val = str(r["value"]).strip()
        if rt == "FORCE_MODE":
            d["FORCE_MODE"][key] = val.upper()
        elif rt == "SHIP_ALONE" and val.upper() in ["TRUE","1","YES"]:
            d["SHIP_ALONE"].add(key)
        elif rt == "MAX_STOPS":
            try:
                d["MAX_STOPS"][key] = int(float(val))
            except:
                pass
        elif rt == "NO_POOLING" and val.upper() in ["TRUE","1","YES"]:
            d["NO_POOLING"].add(key)
        elif rt == "DO_NOT_SHIP_WITH_GROUP":
            d["DONT_MIX"].add((key.upper(), val.upper()))
            d["DONT_MIX"].add((val.upper(), key.upper()))
    return d

def violates_mix(g: pd.DataFrame, dont_mix: set) -> bool:
    groups = set(g["compat_group"].astype(str).str.upper().tolist())
    for a in groups:
        for b in groups:
            if a != b and (a,b) in dont_mix:
                return True
    return False

# -----------------------------
# Baseline execution (no optimization)
# -----------------------------
def baseline_shipments(orders: pd.DataFrame, parcel: pd.DataFrame, ltl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, o in orders.iterrows():
        wt = float(o.weight_lbs)
        exp = bool(o.expedite_flag)
        miles = float(o.miles)
        origin = str(o.origin)
        dest = str(o.dest_zone)
        cls = float(o.ltl_class)

        if wt <= 70:
            mode = "Parcel"
            cost = rate_parcel(o, parcel)
        else:
            mode = "LTL"
            cost = rate_ltl(wt, origin, miles, cls, exp, ltl)

        rows.append({
            "Load_ID": f"BASE-{o.order_id}",
            "Mode": mode,
            "origin": origin,
            "dest_zone": dest,
            "ship_date": o.ship_date,
            "total_weight": wt,
            "total_cube": float(o.cube_ft3),
            "total_cost": float(cost),
            "order_count": 1
        })
    return pd.DataFrame(rows)

# -----------------------------
# Pooling buckets (with NO_POOLING rule)
# -----------------------------
def assign_pool_bucket(df: pd.DataFrame, pool_days: int, no_pool_customers: set) -> pd.DataFrame:
    x = df.copy().sort_values(["origin","dest_zone","ship_date"]).reset_index(drop=True)
    buckets = []
    last_key = None
    last_date = None
    current_bucket = None
    for _, r in x.iterrows():
        key = (r["origin"], r["dest_zone"])
        d = r["ship_date"]
        cust = str(r["customer_id"])
        # no pooling customers always get their own bucket per ship_date
        if cust in no_pool_customers or pool_days == 0:
            buckets.append(f"{cust}-{d}")
            last_key = None
            last_date = None
            current_bucket = None
            continue

        if key != last_key:
            current_bucket = str(uuid.uuid4())[:8]
            last_date = d
            last_key = key
        else:
            gap = (pd.to_datetime(d) - pd.to_datetime(last_date)).days
            if gap > pool_days:
                current_bucket = str(uuid.uuid4())[:8]
            last_date = d
        buckets.append(current_bucket)
    x["pool_bucket"] = buckets
    return x

# -----------------------------
# OR-Tools VRP builder for TL routes
# -----------------------------
def build_tl_vrp(
    orders: pd.DataFrame,
    max_stops: int,
    tl_weight_cap: float,
    tl_cube_cap: float,
    avg_mph: float,
    service_min_per_stop: int,
    hard_time_windows: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build multi-stop TL routes using OR-Tools.
    Here, "nodes" are orders (one drop each) for a SINGLE origin site.
    We assume same origin and same ship_date for all orders passed in.

    Returns:
      routes_df: one row per route (Load_ID)
      route_lines_df: order assignments with sequence
    """
    if not ORTOOLS_OK:
        raise RuntimeError("OR-Tools not installed. Install: pip install ortools")

    if orders.empty:
        return pd.DataFrame(), pd.DataFrame()

    origin = str(orders.origin.iloc[0])
    ship_date = orders.ship_date.min()

    # Create nodes: depot(0) + orders(1..n)
    n = len(orders)
    # Use "distance" proxy: miles to each dest_zone; between destinations approximate by abs difference in miles (toy)
    miles_to = orders["miles"].astype(float).tolist()
    # transit minutes roughly miles / mph * 60
    def travel_min(i: int, j: int) -> int:
        if i == 0:  # depot to node j
            m = miles_to[j-1]
        elif j == 0:  # node i to depot
            m = miles_to[i-1]
        else:
            # crude inter-stop travel approximation
            m = abs(miles_to[i-1] - miles_to[j-1]) + 60
        return int(max(1, (m / max(1e-6, avg_mph)) * 60))

    # Time windows:
    # We'll map ready_date/due_date to minutes from a day-0 epoch.
    # Use ship_date as day 0. ready date: 0.., due date: .. up to days_span
    base_dt = pd.to_datetime(ship_date)
    ready_days = (pd.to_datetime(orders.ready_date) - base_dt).dt.days.clip(lower=0).tolist()
    due_days = (pd.to_datetime(orders.due_date) - base_dt).dt.days.clip(lower=0).tolist()

    # Convert to minutes window: [ready_day*1440, (due_day+1)*1440 - 1]
    # (deliver any time within due day)
    time_windows = [(0, 24*60*30)]  # depot
    for rd, dd in zip(ready_days, due_days):
        a = int(rd * 1440)
        b = int((dd + 1) * 1440 - 1)
        time_windows.append((a, b))

    # Demands
    weights = [0] + orders.weight_lbs.astype(float).round(0).astype(int).tolist()
    cubes = [0] + orders.cube_ft3.astype(float).round(0).astype(int).tolist()

    # Vehicles: allow up to n vehicles, but penalize using many by fixed costs at rating step
    num_vehicles = min(n, 25)  # cap for speed
    depot = 0

    manager = pywrapcp.RoutingIndexManager(n + 1, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # Cost: minimize travel time (proxy); we'll rate dollars after route assignment using TL rate table
    def time_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return travel_min(i, j)

    transit_cb = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # Weight capacity dimension
    def weight_callback(from_index):
        node = manager.IndexToNode(from_index)
        return weights[node]
    w_cb = routing.RegisterUnaryTransitCallback(weight_callback)
    routing.AddDimensionWithVehicleCapacity(
        w_cb, 0, [int(tl_weight_cap)] * num_vehicles, True, "Weight"
    )

    # Cube capacity dimension
    def cube_callback(from_index):
        node = manager.IndexToNode(from_index)
        return cubes[node]
    c_cb = routing.RegisterUnaryTransitCallback(cube_callback)
    routing.AddDimensionWithVehicleCapacity(
        c_cb, 0, [int(tl_cube_cap)] * num_vehicles, True, "Cube"
    )

    # Time window dimension
    routing.AddDimension(
        transit_cb,
        service_min_per_stop,                 # waiting slack
        24 * 60 * 30,                        # horizon
        False,                               # don't force start cumul to zero
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Apply time windows
    for node in range(n + 1):
        idx = manager.NodeToIndex(node)
        a, b = time_windows[node]
        time_dim.CumulVar(idx).SetRange(a, b)

    # Limit stops per vehicle (soft)
    # We'll do this by adding a "Count" dimension.
    def count_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        # count a stop when moving to a non-depot node
        return 1 if j != 0 else 0
    cnt_cb = routing.RegisterTransitCallback(count_callback)
    routing.AddDimension(cnt_cb, 0, max_stops, True, "Stops")
    stops_dim = routing.GetDimensionOrDie("Stops")

    # If hard time windows requested, set high penalty for dropping nodes; else allow dropping with penalty
    penalty = 1000000 if hard_time_windows else 20000
    for node in range(1, n + 1):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Search parameters
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(8)  # keep POC snappy

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return pd.DataFrame(), pd.DataFrame()

    # Extract routes
    route_rows = []
    line_rows = []

    used_vehicle = 0
    for v in range(num_vehicles):
        index = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue  # unused vehicle

        used_vehicle += 1
        load_id = f"TLR-{uuid.uuid4().hex[:10].upper()}"
        seq = 0
        route_orders = []
        total_w = 0
        total_c = 0
        # Gather nodes
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                o = orders.iloc[node - 1]
                route_orders.append(str(o.order_id))
                total_w += float(o.weight_lbs)
                total_c += float(o.cube_ft3)
                tmin = solution.Value(time_dim.CumulVar(index))
                line_rows.append({
                    "Load_ID": load_id,
                    "Mode": "TL",
                    "origin": origin,
                    "dest_zone": str(o.dest_zone),
                    "ship_date": ship_date,
                    "order_id": str(o.order_id),
                    "seq": seq,
                    "weight_lbs": float(o.weight_lbs),
                    "cube_ft3": float(o.cube_ft3),
                    "ready_date": o.ready_date,
                    "due_date": o.due_date,
                    "time_cumul_min": int(tmin),
                })
                seq += 1
            index = solution.Value(routing.NextVar(index))

        route_rows.append({
            "Load_ID": load_id,
            "Mode": "TL",
            "origin": origin,
            "ship_date": ship_date,
            "order_count": len(route_orders),
            "total_weight_lbs": round(total_w, 2),
            "total_cube_ft3": round(total_c, 2),
            "stops": len(route_orders),
        })

    return pd.DataFrame(route_rows), pd.DataFrame(line_rows)

# -----------------------------
# Planning: Rate + Rules + TL routing + fallback
# -----------------------------
def plan_all(
    orders: pd.DataFrame,
    parcel: pd.DataFrame,
    ltl: pd.DataFrame,
    tl: pd.DataFrame,
    rules: dict,
    pool_days: int,
    tl_min_weight: float,
    tl_weight_cap: float,
    tl_cube_cap: float,
    avg_mph: float,
    service_min_per_stop: int,
    hard_time_windows: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      baseline_df, loads_df, load_lines_df, rca_df
    """
    baseline_df = baseline_shipments(orders, parcel, ltl)

    # Apply FORCE_MODE and SHIP_ALONE upfront
    force_mode = rules["FORCE_MODE"]
    ship_alone = rules["SHIP_ALONE"]
    max_stops_by_cust = rules["MAX_STOPS"]
    no_pool_customers = rules["NO_POOLING"]
    dont_mix = rules["DONT_MIX"]

    o = orders.copy()
    o["forced_mode"] = o["customer_id"].map(force_mode).fillna("")
    o["ship_alone"] = o["customer_id"].isin(ship_alone)

    # Pool buckets (respect NO_POOLING customers)
    og = assign_pool_bucket(o, pool_days=pool_days, no_pool_customers=no_pool_customers)

    planned_loads = []
    planned_lines = []
    rca_rows = []

    # Step A: create "locked" loads for ship_alone or forced Parcel/LTL
    locked_ids = set()
    for _, r in og.iterrows():
        if bool(r["ship_alone"]) or str(r["forced_mode"]) in ["PARCEL","LTL"]:
            load_id = f"LOCK-{uuid.uuid4().hex[:10].upper()}"
            mode = "Parcel" if str(r["forced_mode"]) == "PARCEL" else ("LTL" if str(r["forced_mode"]) == "LTL" else None)
            if mode is None:
                # ship alone, choose parcel if <=70 else ltl
                mode = "Parcel" if float(r["weight_lbs"]) <= 70 else "LTL"

            origin = str(r["origin"]); dest = str(r["dest_zone"])
            miles = float(r["miles"]); cls = float(r["ltl_class"]); exp = bool(r["expedite_flag"])
            wt = float(r["weight_lbs"]); cube = float(r["cube_ft3"])

            if mode == "Parcel":
                cost = rate_parcel(r, parcel)
            else:
                cost = rate_ltl(wt, origin, miles, cls, exp, ltl)

            planned_loads.append({
                "Load_ID": load_id, "Mode": mode, "origin": origin, "dest_zone": dest,
                "ship_date": r["ship_date"], "order_count": 1,
                "total_weight_lbs": round(wt,2), "total_cube_ft3": round(cube,2),
                "total_cost": round(float(cost),2),
                "RCA": "SHIP_ALONE/FORCED_MODE"
            })
            planned_lines.append({
                "Load_ID": load_id, "Mode": mode, "origin": origin, "dest_zone": dest, "ship_date": r["ship_date"],
                "order_id": r["order_id"], "customer_id": r["customer_id"], "compat_group": r["compat_group"],
                "weight_lbs": wt, "cube_ft3": cube, "ready_date": r["ready_date"], "due_date": r["due_date"],
            })
            locked_ids.add(str(r["order_id"]))

    # Remaining candidates for pooling/routing
    rem = og[~og["order_id"].astype(str).isin(locked_ids)].copy()

    # Step B: within each pool bucket, decide whether it should be Parcel/LTL or candidate TL routing
    # We'll only consider TL routing for groups that:
    # - do not violate mixing rules
    # - have enough weight to plausibly consolidate
    # - do not have FORCE_MODE=TL? (we will treat it as allowed, but still must be feasible)
    for (origin, dest, bucket), g in rem.groupby(["origin","dest_zone","pool_bucket"]):
        g = g.sort_values("ship_date").reset_index(drop=True)

        # If group violates mixing rules, split by compat_group
        if violates_mix(g, dont_mix):
            for cg, gg in g.groupby("compat_group"):
                rem.loc[gg.index, "pool_bucket"] = f"{bucket}-{cg}"
            continue

    rem = rem.sort_values(["origin","dest_zone","ship_date"]).reset_index(drop=True)

    # Collect TL candidates into one dataset per (origin, ship_date)
    tl_candidate_orders = []
    non_tl_orders = []

    for (origin, dest, bucket), g in rem.groupby(["origin","dest_zone","pool_bucket"]):
        g = g.sort_values("ship_date").reset_index(drop=True)

        miles = float(g["miles"].median())
        total_wt = float(g["weight_lbs"].sum())
        exp_rate = float(g["expedite_flag"].mean())
        expedite = exp_rate >= 0.25
        rep_class = float(np.average(g["ltl_class"], weights=g["weight_lbs"]))

        # Candidate costs:
        parcel_feasible = (g["weight_lbs"].max() <= 70)
        parcel_cost = float(sum(rate_parcel(r, parcel) for _, r in g.iterrows())) if parcel_feasible else float("inf")
        ltl_cost = rate_ltl(total_wt, str(origin), miles, rep_class, expedite, ltl)

        # TL feasibility is not just min weight; also allow if enough density/cube usage
        tl_feasible = total_wt >= tl_min_weight

        # If customer routing guide forces TL (FORCE_MODE=TL), treat as feasible candidate even if threshold not met (but will likely be costly)
        forced_tl = any(str(x).upper() == "TL" for x in g["forced_mode"].astype(str).tolist())
        if forced_tl:
            tl_feasible = True

        tl_cost_1stop = rate_tl(str(origin), str(dest), stops=1, expedite=expedite, tl=tl) if tl_feasible else float("inf")

        # Choose "pre-plan" mode:
        # - If TL is cheapest AND group size >= 2 (so routing makes sense), push to TL routing pool.
        # - Else commit to Parcel or LTL as a single load.
        best_mode = min([("Parcel", parcel_cost), ("LTL", ltl_cost), ("TL", tl_cost_1stop)], key=lambda x: x[1])[0]

        blockers = []
        if exp_rate >= 0.25:
            blockers.append("SLA/expedite pressure")
        if not tl_feasible:
            blockers.append("Below TL threshold")
        if pool_days == 0:
            blockers.append("No pooling window")
        if g["ship_date"].nunique() >= 3 and len(g) >= 6:
            blockers.append("Daily cadence blocks pooling")

        # If best mode is TL and there are multiple orders, route them (multi-stop possible later at origin/day scope)
        if best_mode == "TL" and len(g) >= 2:
            tl_candidate_orders.append(g)
            rca_rows.append({
                "scope":"BUCKET", "origin": origin, "dest_zone": dest, "pool_bucket": bucket,
                "chosen_stage":"TL_CANDIDATE", "blockers":"; ".join(blockers) if blockers else "None"
            })
        else:
            # Single load (Parcel or LTL)
            load_id = f"LOAD-{uuid.uuid4().hex[:10].upper()}"
            ship_date = g["ship_date"].min()
            mode = "Parcel" if (best_mode=="Parcel" and parcel_feasible) else "LTL"
            if best_mode == "Parcel" and not parcel_feasible:
                mode = "LTL"

            if mode == "Parcel":
                cost = float(sum(rate_parcel(r, parcel) for _, r in g.iterrows()))
            else:
                cost = float(ltl_cost)

            planned_loads.append({
                "Load_ID": load_id, "Mode": mode, "origin": str(origin), "dest_zone": str(dest),
                "ship_date": ship_date, "order_count": int(len(g)),
                "total_weight_lbs": round(float(g["weight_lbs"].sum()), 2),
                "total_cube_ft3": round(float(g["cube_ft3"].sum()), 2),
                "total_cost": round(float(cost), 2),
                "RCA": "; ".join(blockers) if blockers else "None"
            })
            for _, r in g.iterrows():
                planned_lines.append({
                    "Load_ID": load_id, "Mode": mode, "origin": str(origin), "dest_zone": str(dest), "ship_date": r["ship_date"],
                    "order_id": r["order_id"], "customer_id": r["customer_id"], "compat_group": r["compat_group"],
                    "weight_lbs": float(r["weight_lbs"]), "cube_ft3": float(r["cube_ft3"]),
                    "ready_date": r["ready_date"], "due_date": r["due_date"],
                })
            rca_rows.append({
                "scope":"BUCKET", "origin": origin, "dest_zone": dest, "pool_bucket": bucket,
                "chosen_stage": f"{mode}_LOAD", "blockers":"; ".join(blockers) if blockers else "None"
            })

    # Step C: OR-Tools routing for TL candidates
    if tl_candidate_orders:
        tl_cand = pd.concat(tl_candidate_orders, ignore_index=True)
        # Route per origin + ship_date (more realistic) and allow mixing destinations in same TL route
        for (origin, ship_dt), g in tl_cand.groupby(["origin","ship_date"]):
            # Determine max stops rule for this set: use minimum of customer max stops if defined, else app max
            custs = g["customer_id"].astype(str).unique().tolist()
            cust_caps = [max_stops_by_cust.get(c, None) for c in custs]
            cust_caps = [x for x in cust_caps if isinstance(x, int)]
            rule_max_stops = min(cust_caps) if cust_caps else None

            routes_df, route_lines_df = build_tl_vrp(
                orders=g.reset_index(drop=True),
                max_stops=int(rule_max_stops) if rule_max_stops else int(app_max_stops),
                tl_weight_cap=float(tl_weight_cap),
                tl_cube_cap=float(tl_cube_cap),
                avg_mph=float(avg_mph),
                service_min_per_stop=int(service_min_per_stop),
                hard_time_windows=bool(hard_time_windows),
            )

            if routes_df.empty:
                # Fallback: if routing fails, rate each bucket as LTL
                for (dest, bucket), gg in g.groupby(["dest_zone","pool_bucket"]):
                    miles = float(gg["miles"].median())
                    exp = bool(gg["expedite_flag"].mean() >= 0.25)
                    rep_class = float(np.average(gg["ltl_class"], weights=gg["weight_lbs"]))
                    total_wt = float(gg["weight_lbs"].sum())
                    ltl_cost = rate_ltl(total_wt, str(origin), miles, rep_class, exp, ltl)
                    load_id = f"FALL-{uuid.uuid4().hex[:10].upper()}"
                    planned_loads.append({
                        "Load_ID": load_id, "Mode":"LTL", "origin": str(origin), "dest_zone": str(dest),
                        "ship_date": gg["ship_date"].min(), "order_count": int(len(gg)),
                        "total_weight_lbs": round(total_wt,2), "total_cube_ft3": round(float(gg["cube_ft3"].sum()),2),
                        "total_cost": round(float(ltl_cost),2),
                        "RCA":"VRP failed -> LTL fallback"
                    })
                    for _, r in gg.iterrows():
                        planned_lines.append({
                            "Load_ID": load_id, "Mode":"LTL", "origin": str(origin), "dest_zone": str(dest),
                            "ship_date": r["ship_date"], "order_id": r["order_id"], "customer_id": r["customer_id"],
                            "compat_group": r["compat_group"], "weight_lbs": float(r["weight_lbs"]), "cube_ft3": float(r["cube_ft3"]),
                            "ready_date": r["ready_date"], "due_date": r["due_date"],
                        })
                continue

            # Rate each TL route using TL rate and stop charges
            for _, rr in routes_df.iterrows():
                load_id = str(rr["Load_ID"])
                stops = int(rr["stops"])
                # Expedite for route if any order expedite
                route_orders = route_lines_df[route_lines_df["Load_ID"] == load_id]["order_id"].astype(str).tolist()
                subset = g[g["order_id"].astype(str).isin(route_orders)]
                exp = bool(subset["expedite_flag"].mean() >= 0.25)

                # For rating, choose "representative" dest as the most common dest; in real TMS you'd compute actual multi-stop miles
                # Here we use median miles to approximate.
                dest_mode = subset["dest_zone"].mode()
                dest_for_rate = str(dest_mode.iloc[0]) if not dest_mode.empty else str(subset["dest_zone"].iloc[0])

                cost = rate_tl(str(origin), dest_for_rate, stops=stops, expedite=exp, tl=tl)

                planned_loads.append({
                    "Load_ID": load_id, "Mode":"TL", "origin": str(origin), "dest_zone": "MULTI",
                    "ship_date": ship_dt, "order_count": int(rr["order_count"]),
                    "total_weight_lbs": float(rr["total_weight_lbs"]),
                    "total_cube_ft3": float(rr["total_cube_ft3"]),
                    "total_cost": round(float(cost),2),
                    "RCA":"VRP multi-stop TL"
                })

            # Add route lines
            for _, lr in route_lines_df.iterrows():
                oid = str(lr["order_id"])
                orec = g[g["order_id"].astype(str)==oid].iloc[0]
                planned_lines.append({
                    "Load_ID": str(lr["Load_ID"]), "Mode":"TL", "origin": str(origin), "dest_zone": str(orec.dest_zone),
                    "ship_date": ship_dt, "order_id": oid, "customer_id": str(orec.customer_id),
                    "compat_group": str(orec.compat_group), "weight_lbs": float(orec.weight_lbs), "cube_ft3": float(orec.cube_ft3),
                    "ready_date": orec.ready_date, "due_date": orec.due_date,
                    "seq": int(lr.get("seq", 0)),
                    "time_cumul_min": int(lr.get("time_cumul_min", 0)),
                })

    loads_df = pd.DataFrame(planned_loads)
    lines_df = pd.DataFrame(planned_lines)
    rca_df = pd.DataFrame(rca_rows)

    # Clean ordering
    if not loads_df.empty:
        loads_df = loads_df.sort_values(["ship_date","origin","Mode"]).reset_index(drop=True)
    if not lines_df.empty:
        sort_cols = ["Load_ID","seq"] if "seq" in lines_df.columns else ["Load_ID","order_id"]
        lines_df = lines_df.sort_values(sort_cols).reset_index(drop=True)

    return baseline_df, loads_df, lines_df, rca_df

# -----------------------------
# RCA & Ishikawa
# -----------------------------
def rca_pareto(rca_df: pd.DataFrame) -> pd.DataFrame:
    if rca_df.empty:
        return pd.DataFrame()
    ser = rca_df["blockers"].fillna("").str.split("; ").explode()
    ser = ser[(ser != "") & (ser != "None")]
    vc = ser.value_counts().reset_index()
    vc.columns = ["Root_Cause","Count"]
    return vc

def blocker_to_category(blocker: str) -> str:
    b = blocker.lower()
    if "sla" in b or "expedite" in b:
        return "Service / SLA"
    if "pool" in b or "cadence" in b:
        return "Planning / Cadence"
    if "threshold" in b or "weight" in b:
        return "Load Feasibility"
    if "vrp" in b:
        return "Solver / Constraints"
    return "Process / Systems"

def ishikawa(rca_p: pd.DataFrame):
    if rca_p.empty:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.text(0.5,0.5,"No RCA to display.", ha="center", va="center")
        ax.axis("off")
        return fig

    tmp = rca_p.copy()
    tmp["Category"] = tmp["Root_Cause"].apply(blocker_to_category)
    grouped = tmp.sort_values("Count", ascending=False).groupby("Category", as_index=False).head(4)

    cats = grouped["Category"].unique().tolist()
    top = cats[::2]
    bottom = cats[1::2]

    fig, ax = plt.subplots(figsize=(14,6))
    ax.axis("off")
    ax.plot([0.05,0.95],[0.5,0.5], linewidth=2)
    ax.text(0.96,0.5,"Why the optimized plan is constrained", va="center", fontsize=12)

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

# -----------------------------
# Maps
# -----------------------------
def lane_agg(df: pd.DataFrame, dest_col: str, cost_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Dest_Zone","Shipments","Total_Cost"])
    return df.groupby(dest_col, as_index=False).agg(Shipments=(dest_col,"count"), Total_Cost=(cost_col,"sum")).rename(columns={dest_col:"Dest_Zone"})

def lane_map(df: pd.DataFrame, title: str, base_scale: int, dest_col: str, cost_col: str, origin_site: Optional[str]=None) -> go.Figure:
    coords = {d:(lat,lon) for d,_,lat,lon in DESTS}
    if origin_site:
        o_lat, o_lon = coords_for_site(origin_site)
    else:
        o_lat, o_lon = coords_for_site(SITES[0][0])

    agg = lane_agg(df, dest_col, cost_col)
    base = max(1, int(base_scale))

    fig = go.Figure()
    for _, r in agg.iterrows():
        dest = str(r["Dest_Zone"])
        if dest not in coords:
            continue
        lat2, lon2 = coords[dest]
        width = 1.0 + 11.0 * math.sqrt(float(r["Shipments"]) / base)
        fig.add_trace(go.Scattergeo(
            lat=[o_lat, lat2], lon=[o_lon, lon2],
            mode="lines", line=dict(width=width),
            hoverinfo="text",
            text=f"{dest}<br>Shipments: {int(r['Shipments'])}<br>Total: ${r['Total_Cost']:,.0f}",
            showlegend=False
        ))

    fig.add_trace(go.Scattergeo(
        lat=[o_lat], lon=[o_lon], mode="markers+text",
        text=[origin_site or "ORIGIN"], textposition="bottom center",
        marker=dict(size=10), showlegend=False
    ))

    dest_lats, dest_lons, dest_names = [], [], []
    for d,_,lat,lon in DESTS:
        dest_lats.append(lat); dest_lons.append(lon); dest_names.append(d)
    fig.add_trace(go.Scattergeo(
        lat=dest_lats, lon=dest_lons, mode="markers+text",
        text=dest_names, textposition="top center",
        marker=dict(size=6), showlegend=False
    ))

    fig.update_layout(
        title=title,
        geo=dict(scope="usa", projection_type="albers usa", showland=True),
        margin=dict(l=10,r=10,t=40,b=10),
        height=520
    )
    return fig

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Synthetic data")
seed = st.sidebar.number_input("Seed", 1, 10000, 21)
n_orders = st.sidebar.number_input("Orders", 200, 20000, 4000, 200)
days_span = st.sidebar.slider("Ship date span (days)", 7, 60, 30)
n_customers = st.sidebar.number_input("Customers", 10, 500, 60, 10)

st.sidebar.header("Optimization knobs")
pool_days = st.sidebar.slider("Pooling window (days)", 0, 7, 2)
tl_min_weight = st.sidebar.number_input("TL threshold (lbs)", 5000, 44000, 12000, 1000)
tl_weight_cap = st.sidebar.number_input("TL weight capacity (lbs)", 10000, 50000, 44000, 1000)
tl_cube_cap = st.sidebar.number_input("TL cube capacity (ft3)", 500, 6000, int(DEFAULT_TL_CUBE_FT3), 100)

st.sidebar.header("Routing (OR-Tools)")
app_max_stops = st.sidebar.slider("Max stops per TL route (default)", 2, 10, 5)
avg_mph = st.sidebar.slider("Average speed (mph)", 30, 65, 50)
service_min_per_stop = st.sidebar.slider("Service time per stop (min)", 5, 60, 15)
hard_time_windows = st.sidebar.checkbox("Hard time windows", value=False, help="If enabled, solver heavily penalizes late deliveries.")

st.sidebar.header("Business rules")
use_default_rules = st.sidebar.checkbox("Use default synthetic rules", value=True)
rules_upload = st.sidebar.file_uploader("Or upload rules.csv (rule_type,key,value)", type=["csv"])

run = st.sidebar.button("Generate + Optimize")

# -----------------------------
# Main
# -----------------------------
if not ORTOOLS_OK:
    st.warning("OR-Tools is not installed in this environment. Install it to enable multi-stop TL routing: pip install ortools")

if run:
    orders = synthesize_orders(int(n_orders), int(days_span), int(n_customers), int(seed))
    parcel, ltl, tl = synthesize_rates(int(seed))

    if use_default_rules:
        rules_df = default_rules(int(seed))
    else:
        rules_df = pd.DataFrame(columns=["rule_type","key","value"])
    if rules_upload is not None:
        try:
            rules_df = pd.read_csv(rules_upload)
        except Exception as e:
            st.error(f"Could not read rules.csv: {e}")

    # Ensure columns exist
    if not rules_df.empty:
        for c in ["rule_type","key","value"]:
            if c not in rules_df.columns:
                st.error("rules.csv must have columns: rule_type,key,value")
                st.stop()

    rules = parse_rules(rules_df)

    # Show inputs
    st.subheader("Synthetic orders (sample)")
    st.dataframe(orders.head(30), use_container_width=True)

    st.subheader("Business rules (editable view)")
    st.dataframe(rules_df, use_container_width=True)

    baseline_df, loads_df, load_lines_df, rca_df = plan_all(
        orders=orders,
        parcel=parcel,
        ltl=ltl,
        tl=tl,
        rules=rules,
        pool_days=int(pool_days),
        tl_min_weight=float(tl_min_weight),
        tl_weight_cap=float(tl_weight_cap),
        tl_cube_cap=float(tl_cube_cap),
        avg_mph=float(avg_mph),
        service_min_per_stop=int(service_min_per_stop),
        hard_time_windows=bool(hard_time_windows),
    )

    # Headline
    base_total = float(baseline_df.total_cost.sum()) if not baseline_df.empty else 0.0
    after_total = float(loads_df.total_cost.sum()) if not loads_df.empty else 0.0
    savings = base_total - after_total

    st.header("Baseline vs Optimized (headline)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Orders", f"{len(orders):,}")
    k2.metric("Baseline shipments", f"{len(baseline_df):,}")
    k3.metric("Optimized loads", f"{len(loads_df):,}")
    k4.metric("Modeled savings", f"${savings:,.0f}", f"{(100*savings/base_total if base_total>0 else 0):.1f}%")

    # Mode mix
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Baseline mode mix")
        st.dataframe(
            baseline_df.groupby("Mode", as_index=False).agg(Shipments=("Mode","count"), Total_Cost=("total_cost","sum")).sort_values("Total_Cost", ascending=False),
            use_container_width=True
        )
    with c2:
        st.subheader("Optimized mode mix")
        st.dataframe(
            loads_df.groupby("Mode", as_index=False).agg(Loads=("Mode","count"), Total_Cost=("total_cost","sum")).sort_values("Total_Cost", ascending=False),
            use_container_width=True
        )

    # Output tables (what you asked)
    st.header("Planned output")
    st.subheader("Loads (Load_ID + mode + totals)")
    st.dataframe(loads_df, use_container_width=True)

    st.subheader("Load lines (orders in each load)")
    st.dataframe(load_lines_df.head(250), use_container_width=True)
    st.caption("Showing first 250 lines. Download full CSV below.")

    st.download_button("Download loads.csv", loads_df.to_csv(index=False).encode("utf-8"), "loads.csv", "text/csv")
    st.download_button("Download load_lines.csv", load_lines_df.to_csv(index=False).encode("utf-8"), "load_lines.csv", "text/csv")

    # Lane delta
    st.header("Lane delta (Before vs After)")
    before_lane = lane_agg(baseline_df, "dest_zone", "total_cost").rename(columns={"Shipments":"Shipments_Before","Total_Cost":"Cost_Before"})
    after_lane = lane_agg(loads_df[loads_df["dest_zone"].isin(list(DEST_WEIGHTS.keys()))], "dest_zone", "total_cost").rename(columns={"Shipments":"Shipments_After","Total_Cost":"Cost_After"})
    delta = before_lane.merge(after_lane, on="Dest_Zone", how="outer").fillna(0)
    delta["Shipment_Delta"] = delta["Shipments_After"] - delta["Shipments_Before"]
    delta["Cost_Delta"] = delta["Cost_After"] - delta["Cost_Before"]
    delta = delta.sort_values("Shipment_Delta").reset_index(drop=True)
    st.dataframe(delta, use_container_width=True)

    # Maps (use same scaling from BEFORE)
    st.header("Maps (Before vs After)")
    base_scale = int(before_lane["Shipments_Before"].max()) if len(before_lane) else 1
    m1, m2 = st.columns(2)
    with m1:
        st.plotly_chart(lane_map(baseline_df, "BEFORE (baseline shipments by dest)", base_scale, "dest_zone", "total_cost"), use_container_width=True)
    with m2:
        # For AFTER, convert MULTI to its component dests doesn't exist at load-level; we show dest loads only
        after_for_map = loads_df[loads_df["dest_zone"].isin(list(DEST_WEIGHTS.keys()))].copy()
        st.plotly_chart(lane_map(after_for_map, "AFTER (optimized loads by dest)", base_scale, "dest_zone", "total_cost"), use_container_width=True)

    # RCA + Ishikawa
    st.header("RCA (why consolidation/mode shift is constrained)")
    st.subheader("RCA events (sample)")
    st.dataframe(rca_df.head(80), use_container_width=True)

    st.subheader("RCA Pareto")
    rca_p = rca_pareto(rca_df)
    st.dataframe(rca_p, use_container_width=True)

    st.subheader("Ishikawa (Fishbone)")
    st.pyplot(ishikawa(rca_p), clear_figure=True)

else:
    st.info("Set sidebar controls and click **Generate + Optimize**.")
    st.caption("Tip: Increase pool_days and keep TL threshold ~12,000 lb to see strong consolidation into TL routes.")
