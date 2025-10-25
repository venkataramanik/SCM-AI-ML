#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cloud-ready Streamlit app:
Trailer/Container Load Optimization – Algorithms + KPIs + 3D Visualization

Includes:
• A) FFD (First-Fit Decreasing) – capacity-only, super fast
• B) MILP (PuLP) – capacity-only optimization (optional; app runs even if PuLP missing)
• C) 2D Shelf – geometry-aware packing with simple door-first unload

KPIs:
• Trailers used, avg weight & cube utilization, (for geometry) floor utilization
• Sequence blocking (door-first violations), CoG offset (approx), runtime

3D View:
• Interactive Plotly cube view of the load plan (color by stop)
• Download the 3D scene as a standalone HTML file
"""

import time, math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Plotly for 3D visualization
import plotly.graph_objects as go
from plotly.colors import qualitative as qcolors
import plotly.io as pio

# Optional MILP (PuLP)
_HAVE_PULP = False
try:
    import pulp
    _HAVEPULP_VERSION = pulp.__version__
    _HAVE_PULP = True
except Exception:
    _HAVEPULP_VERSION = None
    _HAVE_PULP = False


# =========================
# Data structures & helpers
# =========================
@dataclass
class TrailerSpec:
    length: float  # internal usable length (m)
    width: float   # internal usable width (m)
    height: float  # internal usable height (m)
    max_weight: float  # kg
    max_volume: float  # m^3
    door_at_y: float = 0.0  # y=0 at door; y increases inward


@dataclass
class Item:
    sku: str
    length: float
    width: float
    height: float
    weight: float
    stop: int           # 1 = earliest unload; higher = later
    stackable: bool = True
    can_rotate: bool = True

    def volume(self) -> float:
        return self.length * self.width * self.height


@dataclass
class Placement:
    trailer_id: int
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    weight: float
    stop: int
    sku: str


def generate_instance(n_items: int = 120, seed: int = 42, stops: int = 4) -> Tuple[TrailerSpec, List[Item]]:
    random.seed(seed); np.random.seed(seed)
    trailer = TrailerSpec(
        length=13.2, width=2.45, height=2.6,          # ~53' van inside (meters)
        max_weight=22000.0,
        max_volume=13.2*2.45*2.6
    )
    items: List[Item] = []
    for i in range(n_items):
        if random.random() < 0.6:  # case
            L = random.uniform(0.30, 0.80)
            W = random.uniform(0.25, 0.70)
            H = random.uniform(0.15, 0.50)
            wt = random.uniform(8, 40)
            stackable = True
        else:  # pallet
            L = random.uniform(1.0, 1.2)
            W = random.uniform(0.9, 1.1)
            H = random.uniform(0.8, 1.6)
            wt = random.uniform(150, 450)
            stackable = random.random() < 0.3
        items.append(Item(
            sku=f"SKU{i+1}", length=L, width=W, height=H,
            weight=wt, stop=random.randint(1, stops),
            stackable=stackable, can_rotate=True
        ))
    return trailer, items


# =========
# KPI utils
# =========
def kpis_capacity_only(assign: Dict[int, List[int]], items: List[Item], trailer: TrailerSpec) -> Dict[str, Any]:
    used = len(assign)
    w_util, v_util = [], []
    for t in assign:
        w = sum(items[i].weight for i in assign[t])
        v = sum(items[i].volume() for i in assign[t])
        w_util.append(min(100*w/trailer.max_weight, 100))
        v_util.append(min(100*v/trailer.max_volume, 100))
    return {
        "trailers": used,
        "avg_weight_util_%": float(np.mean(w_util)) if w_util else 0.0,
        "avg_cube_util_%": float(np.mean(v_util)) if v_util else 0.0,
        "linear_util_%": np.nan,
        "sequence_blocks": np.nan,
        "cog_offset_m": np.nan,
    }


def kpis_geometry(placements: List[Placement], trailer: TrailerSpec) -> Dict[str, Any]:
    if not placements:
        return {"trailers": 0, "avg_weight_util_%": 0, "avg_cube_util_%": 0,
                "linear_util_%": 0, "sequence_blocks": 0, "cog_offset_m": 0}
    by_trailer: Dict[int, List[Placement]] = {}
    for p in placements:
        by_trailer.setdefault(p.trailer_id, []).append(p)
    nT = len(by_trailer)
    w_utils, v_utils, lin_utils, cogs = [], [], [], []
    seq_viol = 0
    for t, plist in by_trailer.items():
        wsum = sum(p.weight for p in plist)
        vsum = sum(p.length*p.width*p.height for p in plist)
        w_utils.append(min(100*wsum/trailer.max_weight, 100))
        v_utils.append(min(100*vsum/trailer.max_volume, 100))
        used_area = sum(p.length*p.width for p in plist)
        floor_area = trailer.length*trailer.width
        lin_utils.append(min(100*used_area/floor_area, 100))

        # sequence: earlier-stop should be nearer door (smaller y)
        for i in range(len(plist)):
            for j in range(len(plist)):
                if plist[i].stop < plist[j].stop and (plist[i].y > plist[j].y):
                    seq_viol += 1

        # center of gravity offset from geometric center (x,y)
        totw = wsum or 1.0
        cx = sum((p.x+0.5*p.length)*p.weight for p in plist)/totw
        cy = sum((p.y+0.5*p.width )*p.weight  for p in plist)/totw
        cogs.append(math.dist((cx, cy), (trailer.length/2, trailer.width/2)))
    return {
        "trailers": nT,
        "avg_weight_util_%": float(np.mean(w_utils)),
        "avg_cube_util_%": float(np.mean(v_utils)),
        "linear_util_%": float(np.mean(lin_utils)),
        "sequence_blocks": int(seq_viol),
        "cog_offset_m": float(np.mean(cogs)),
    }


# ============================
# A) First-Fit Decreasing (FFD)
# ============================
def algo_ffd_capacity(items: List[Item], trailer: TrailerSpec) -> Dict[int, List[int]]:
    order = sorted(range(len(items)), key=lambda i: (items[i].volume(), items[i].weight), reverse=True)
    loads: Dict[int, List[int]] = {}
    rem_w, rem_v = {}, {}
    t_id = 0
    for i in order:
        placed = False
        for t in loads:
            if items[i].weight <= rem_w[t] and items[i].volume() <= rem_v[t]:
                loads[t].append(i); rem_w[t] -= items[i].weight; rem_v[t] -= items[i].volume(); placed = True; break
        if not placed:
            loads[t_id] = [i]
            rem_w[t_id] = trailer.max_weight - items[i].weight
            rem_v[t_id] = trailer.max_volume - items[i].volume()
            t_id += 1
    return loads


# ========================
# B) MILP (PuLP) – optional
# ========================
def algo_mip_binpacking(items: List[Item], trailer: TrailerSpec, timelimit=60) -> Optional[Dict[int, List[int]]]:
    if not _HAVE_PULP:
        return None
    n = len(items)
    ub = algo_ffd_capacity(items, trailer)
    T = len(ub)
    prob = pulp.LpProblem("BinPacking", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (range(n), range(T)), 0, 1, cat="Binary")
    y = pulp.LpVariable.dicts("y", range(T), 0, 1, cat="Binary")
    prob += pulp.lpSum([y[t] for t in range(T)])
    for i in range(n):
        prob += pulp.lpSum([x[i][t] for t in range(T)]) == 1
    for t in range(T):
        prob += pulp.lpSum([items[i].weight * x[i][t] for i in range(n)]) <= trailer.max_weight * y[t]
        prob += pulp.lpSum([items[i].volume() * x[i][t] for i in range(n)]) <= trailer.max_volume * y[t]
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=timelimit))
    assign: Dict[int, List[int]] = {}
    tid = 0
    for t in range(T):
        if y[t].value() and y[t].value() > 0.5:
            picks = [i for i in range(n) if x[i][t].value() and x[i][t].value() > 0.5]
            if picks:
                assign[tid] = picks
                tid += 1
    return assign if assign else ub


# ==================================
# C) 2D Shelf – geometry-aware loader
# ==================================
def algo_2d_shelf(items: List[Item], trailer: TrailerSpec) -> List[Placement]:
    # Strategy: pack by stop (earliest near door y≈0), rows (“shelves”) along x
    placements: List[Placement] = []
    t_id = 0; x = y = 0.0; shelf_depth = 0.0

    def new_shelf():
        nonlocal x, y, shelf_depth
        x = 0.0
        y += shelf_depth
        shelf_depth = 0.0

    def new_trailer():
        nonlocal t_id, x, y, shelf_depth
        t_id += 1
        x = y = 0.0
        shelf_depth = 0.0

    order = sorted(range(len(items)), key=lambda i: (items[i].stop, items[i].length*items[i].width), reverse=True)
    order = sorted(order, key=lambda i: items[i].stop)

    for i in order:
        it = items[i]
        L, W = it.length, it.width
        # rotate to reduce shelf depth if helpful
        if it.can_rotate and L < W:
            L, W = W, L

        placed = False
        attempts = 0
        while not placed and attempts < 1000:
            attempts += 1
            # new shelf if run out of length
            if x + L > trailer.length + 1e-9:
                if y + shelf_depth + W <= trailer.width + 1e-9:
                    new_shelf()
                else:
                    new_trailer()
            # capacity check (approx)
            cur = [p for p in placements if p.trailer_id == t_id]
            cur_w = sum(p.weight for p in cur)
            cur_v = sum(p.length*p.width*p.height for p in cur)
            if cur_w + it.weight > trailer.max_weight + 1e-9 or cur_v + it.volume() > trailer.max_volume + 1e-9:
                new_trailer(); continue

            # place
            px, py, pz = x, y, 0.0
            placements.append(Placement(
                trailer_id=t_id, x=px, y=py, z=pz,
                length=L, width=W, height=it.height,
                weight=it.weight, stop=it.stop, sku=it.sku
            ))
            x += L
            shelf_depth = max(shelf_depth, W)
            placed = True

    return placements


# ==================
# 3D Visualization
# ==================
def _cuboid_mesh(x, y, z, dx, dy, dz):
    X = [x, x+dx, x+dx, x,   x,   x+dx, x+dx, x   ]
    Y = [y, y,   y+dy, y+dy, y,   y,    y+dy, y+dy]
    Z = [z, z,   z,    z,    z+dz,z+dz, z+dz, z+dz]
    I = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2]
    J = [1, 2, 3, 5, 6, 7, 5, 6, 7, 4, 5, 6]
    K = [2, 3, 1, 6, 7, 5, 4, 7, 6, 5, 6, 5]
    return X, Y, Z, I, J, K


def plot_trailer_3d(placements: List[Placement], trailer: TrailerSpec, title="3D Load View"):
    fig = go.Figure()

    # Trailer wireframe
    tw, tl, th = trailer.width, trailer.length, trailer.height
    wf = go.Scatter3d(
        x=[0, tl, tl, 0, 0, 0, tl, tl, 0, 0, tl, tl],
        y=[0, 0,  tw, tw,0, 0, 0,  tw, tw,0, tw, tw],
        z=[0, 0,  0,  0, 0, th, th, th, th, th, th, 0],
        mode='lines',
        line=dict(width=3),
        name='Trailer'
    )
    fig.add_trace(wf)

    # Color by stop
    palette = qcolors.Set2 + qcolors.Set3 + qcolors.Plotly
    def color_for_stop(s): return palette[(s-1) % len(palette)]

    for p in placements:
        X, Y, Z, I, J, K = _cuboid_mesh(p.x, p.y, p.z, p.length, p.width, min(p.height, trailer.height))
        fig.add_trace(go.Mesh3d(
            x=X, y=Y, z=Z, i=I, j=J, k=K,
            color=color_for_stop(p.stop),
            opacity=0.6,
            name=f"{p.sku} (stop {p.stop})",
            hovertext=f"{p.sku}<br>Stop {p.stop}<br>{p.length:.2f}×{p.width:.2f}×{p.height:.2f} m<br>{p.weight:.0f} kg",
            hoverinfo="text"
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (length m)",
            yaxis_title="Y (width m; door at 0)",
            zaxis_title="Z (height m)",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    return fig


# =========================
# Streamlit App – UI & Run
# =========================
st.set_page_config(page_title="Load Optimization – Algo Shootout (3D)", layout="wide")
st.title("🚚 Load Optimization – Algorithm Shootout (3D)")
st.caption("FFD (capacity), MILP (capacity; optional), and 2D Shelf (geometry). Compare KPIs and see a 3D load view.")

# Sidebar controls
with st.sidebar:
    st.header("Parameters")
    n_items = st.slider("Number of items", 30, 300, 140, step=10)
    n_stops = st.slider("Delivery stops", 1, 8, 4, step=1)
    seed    = st.number_input("Random seed", value=42, step=1)
    run_btn = st.button("Run optimization")

    st.markdown("---")
    st.subheader("MILP (optional)")
    if _HAVE_PULP:
        st.success(f"PuLP available (v{_HAVEPULP_VERSION}). MILP will run.")
        timelimit = st.slider("MILP time limit (s)", 10, 180, 60, step=10)
    else:
        st.warning("PuLP not installed. MILP will be skipped.")

if run_btn:
    # Generate instance
    trailer, items = generate_instance(n_items=n_items, seed=int(seed), stops=n_stops)

    # A) FFD
    t0 = time.time()
    ffd = algo_ffd_capacity(items, trailer)
    tA = time.time() - t0
    rowA = {"algo":"A) FFD (capacity)", **kpis_capacity_only(ffd, items, trailer), "runtime_s": round(tA,3)}

    # B) MILP (optional)
    if _HAVE_PULP:
        t0 = time.time()
        mip = algo_mip_binpacking(items, trailer, timelimit=timelimit)
        tB = time.time() - t0
        rowB = {"algo":"B) MILP (capacity)", **kpis_capacity_only(mip, items, trailer), "runtime_s": round(tB,3)}
    else:
        rowB = {"algo":"B) MILP (capacity) – PuLP not installed",
                "trailers": np.nan, "avg_weight_util_%": np.nan, "avg_cube_util_%": np.nan,
                "linear_util_%": np.nan, "sequence_blocks": np.nan, "cog_offset_m": np.nan,
                "runtime_s": np.nan}

    # C) 2D Shelf (geometry)
    t0 = time.time()
    placements = algo_2d_shelf(items, trailer)
    tC = time.time() - t0
    rowC = {"algo":"C) 2D Shelf (geometry)", **kpis_geometry(placements, trailer), "runtime_s": round(tC,3)}

    # KPI table
    df = pd.DataFrame([rowA, rowB, rowC], columns=[
        "algo","trailers","avg_weight_util_%","avg_cube_util_%","linear_util_%",
        "sequence_blocks","cog_offset_m","runtime_s"
    ])
    st.subheader("KPI Comparison")
    st.dataframe(
        df.style.format({
            "avg_weight_util_%":"{:.1f}",
            "avg_cube_util_%":"{:.1f}",
            "linear_util_%":"{:.1f}",
            "cog_offset_m":"{:.2f}",
            "runtime_s":"{:.2f}"
        }),
        use_container_width=True
    )

    # Explanations (non-technical)
    with st.expander("What each algorithm is doing (plain English)", expanded=False):
        st.markdown("""
**A) FFD (First-Fit Decreasing)**  
Sort boxes by size, then drop each into the first trailer that still has room. If it doesn’t fit, open a new one.  
*Very fast; ignores shapes and unloading order.*

**B) MILP (Mathematical optimization)**  
A solver tries combinations to use the **fewest trailers** while staying under weight/volume caps.  
*Finds the theoretical best for small/medium sets; doesn’t place boxes physically.*

**C) 2D Shelf (Geometry-aware)**  
Pack row-by-row from the door inward, keeping **early-stop** freight near the door (simple unload sequence).  
*Gives a realistic floor plan and basic sequence logic; may need 1 extra trailer vs MILP for practicality.*
        """)

    # 3D Visualization
    st.subheader("3D Load View (geometry plan)")
    fig = plot_trailer_3d(placements, trailer, title=f"3D Load – 2D Shelf plan (items={n_items}, stops={n_stops})")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # Download 3D scene as HTML
    html_bytes = pio.to_html(fig, include_plotlyjs="cdn").encode("utf-8")
    st.download_button("Download 3D view (HTML)", data=html_bytes, file_name="load_view.html", mime="text/html")

else:
    st.info("Set your parameters on the left and click **Run optimization** to compare algorithms and view the 3D plan.")
