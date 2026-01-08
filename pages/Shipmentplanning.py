import streamlit as st
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import plotly.express as px
import random

# --- 1. DATA GENERATOR ---
def generate_random_orders(n=30):
    categories = ['Food', 'Chemicals', 'General', 'Retail']
    data = []
    # Depot (Index 0)
    data.append({'OrderID': 'DC-HUB', 'Weight': 0, 'Lat': 39.0, 'Lon': -94.5, 'Ready': 0, 'Type': 'DC', 'Urgent': False})
    for i in range(n):
        is_urgent = random.random() < 0.15
        data.append({
            'OrderID': f"ORD-{1000+i}",
            'Weight': random.randint(500, 12000), # Increased weight to force TL consolidation
            'Lat': 39.0 + random.uniform(-4.0, 4.0),
            'Lon': -94.5 + random.uniform(-4.0, 4.0),
            'Ready': 0 if is_urgent else random.randint(0, 3), 
            'Type': random.choice(categories),
            'Urgent': is_urgent
        })
    return pd.DataFrame(data)

# --- 2. OPTIMIZATION ENGINE ---
def run_optimization(df, rules):
    num_locs = len(df)
    manager = pywrapcp.RoutingIndexManager(num_locs, 15, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_fn(from_idx, to_idx):
        f = manager.IndexToNode(from_idx)
        t = manager.IndexToNode(to_idx)
        return int(np.hypot(df.iloc[f]['Lat']-df.iloc[t]['Lat'], df.iloc[f]['Lon']-df.iloc[t]['Lon']) * 100)
    
    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_fn))
    
    # Weight Constraint
    def weight_fn(idx): return int(df.iloc[manager.IndexToNode(idx)]['Weight'])
    routing.AddDimension(routing.RegisterUnaryTransitCallback(weight_fn), 0, rules['max_w'], True, 'Weight')

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    return routing, manager, routing.SolveWithParameters(search_params)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Strategic Freight Optimizer", layout="wide")

if 'raw_data' not in st.session_state:
    st.session_state.raw_data = generate_random_orders(30)

df = st.session_state.raw_data

# Header Section
st.title("🚛 Freight Consolidation Dashboard")
st.markdown("### Strategic Mode-Shift: LTL to Multi-Stop Truckload")

# Sidebar Controls
with st.sidebar:
    st.header("Planning Constraints")
    horizon = st.slider("Horizon (Days)", 0, 5, 2)
    max_w = st.number_input("Max Truck Weight (Lbs)", value=44000)
    if st.button("🔄 Generate New Scenario"):
        st.session_state.raw_data = generate_random_orders(30)
        st.rerun()

# --- THE EXECUTIVE METRICS (PRE-OPTIMIZATION) ---
st.subheader("📊 Network Summary")
col_o, col_l, col_s = st.columns(3)

# Filter by horizon
filtered_df = df[(df['Ready'] <= horizon) | (df['Urgent'] == True) | (df['Type'] == 'DC')].copy().reset_index(drop=True)
total_orders = len(filtered_df) - 1 # Excluding Depot

col_o.metric("Total Orders in Scope", total_orders)
col_l.metric("Initial Loads (Unconsolidated)", total_orders) # Before AI, every order is a load
col_s.metric("Current Mode", "100% LTL / Parcel")

st.divider()

# --- EXECUTION ---
if st.button("🚀 Execute Consolidation Logic"):
    routing, manager, solution = run_optimization(filtered_df, {'max_w': max_w})
    
    if solution:
        results = []
        for v_id in range(15):
            idx = routing.Start(v_id)
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node != 0:
                    res = filtered_df.iloc[node].copy()
                    res['Load_ID'] = f"TL-LOAD-{100+v_id}"
                    results.append(res)
                idx = solution.Value(routing.NextVar(idx))
        
        res_df = pd.DataFrame(results)
        final_load_count = res_df['Load_ID'].nunique()
        
        # --- THE RESULT BOX ---
        st.success(f"Optimization Complete: Reduced {total_orders} individual shipments into {final_load_count} multi-stop loads.")
        
        # Comparison Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimized Load Count", final_load_count, delta=f"{final_load_count - total_orders}", delta_color="inverse")
        m2.metric("Consolidation Ratio", f"{total_orders / final_load_count:.2f}:1")
        m3.metric("Avg Orders per Load", f"{len(res_df) / final_load_count:.1f}")

        # Show Results Table
        st.write("### 📋 Final Load Manifest")
        st.dataframe(res_df[['Load_ID', 'OrderID', 'Weight', 'Type', 'Urgent']], use_container_width=True)
    else:
        st.error("No solution found. Check weight constraints.")
else:
    st.info("Click the 'Execute' button above to see the consolidation in action.")
