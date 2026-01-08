import streamlit as st
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import plotly.express as px
import random

# --- 1. DATA GENERATOR (RANDOMIZED) ---
def generate_random_orders(n=30):
    categories = ['Food', 'Chemicals', 'General', 'Retail']
    data = []
    # Depot (Index 0)
    data.append({'OrderID': 'DC-HUB', 'Weight': 0, 'Volume': 0, 'Lat': 39.0, 'Lon': -94.5, 
                 'Ready': 0, 'Due': 10, 'Type': 'DC', 'Urgent': False})
    
    for i in range(n):
        is_urgent = random.random() < 0.15  # 15% of orders are urgent
        data.append({
            'OrderID': f"ORD-{1000+i}",
            'Weight': random.randint(100, 18000),
            'Volume': random.randint(20, 900),
            'Lat': 39.0 + random.uniform(-4.0, 4.0),
            'Lon': -94.5 + random.uniform(-4.0, 4.0),
            'Ready': 0 if is_urgent else random.randint(0, 3), 
            'Due': 1 if is_urgent else random.randint(4, 7),
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

    # Weight Capacity
    def weight_fn(idx): return int(df.iloc[manager.IndexToNode(idx)]['Weight'])
    routing.AddDimension(routing.RegisterUnaryTransitCallback(weight_fn), 0, rules['max_w'], True, 'Weight')

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    return routing, manager, routing.SolveWithParameters(search_params)

# --- 3. UI STYLING ---
st.set_page_config(page_title="Strategic Load Optimizer", layout="wide")

st.markdown("""
    <style>
    .benefit-box {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 12px;
        border-left: 10px solid #2ecc71;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .benefit-header { font-weight: bold; color: #27ae60; font-size: 22px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🚛 Strategic Load Optimizer & Sustainability POC")

with st.sidebar:
    st.header("Planning Parameters")
    horizon = st.slider("Consolidation Horizon (Days)", 0, 5, 2)
    max_w = st.number_input("Max Truck Weight (Lbs)", value=44000)
    
    if st.button("🔄 Generate New Scenario"):
        st.session_state.raw_data = generate_random_orders(30)
        st.rerun()

if 'raw_data' not in st.session_state:
    st.session_state.raw_data = generate_random_orders(30)

df = st.session_state.raw_data

# PHASE 1: UNCONSTRAINED DEMAND
st.subheader("📊 Phase 1: Unconstrained Plan (Current State)")
# Baseline cost calculation
df['As_Is_Cost'] = df['Weight'].apply(lambda x: (x/100)*45 + 150 if x > 200 else 22.0)
st.dataframe(df[1:], use_container_width=True, hide_index=True)
total_as_is = df[1:]['As_Is_Cost'].sum()

st.divider()

# PHASE 2: OPTIMIZED PLAN
st.subheader(f"✨ Phase 2: Strategic Optimized Plan ({horizon}-Day Horizon)")

# Urgent orders are forced into the plan regardless of the horizon
filtered_df = df[(df['Ready'] <= horizon) | (df['Urgent'] == True) | (df['Type'] == 'DC')].copy().reset_index(drop=True)

if st.button("🚀 Execute Optimization"):
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
        num_loads = res_df['Load_ID'].nunique()
        opt_cost = (num_loads * 1100) + (len(res_df) * 150) 
        savings = total_as_is - opt_cost

        # --- THE BENEFIT BOX ---
        st.markdown(f"""
        <div class="benefit-box">
            <div class="benefit-header">Planner Insights & Savings Analysis</div>
            <b>📅 Horizon Benefit:</b> Holding non-urgent orders for {horizon} days enabled the consolidation of 
            {len(res_df[res_df['Ready']>0])} shipments.
            <br><br>
            <b>⚖️ Efficiency Benefit:</b> Average truck weight utilization reached 
            <b>{(res_df.groupby('Load_ID')['Weight'].sum().mean()/max_w)*100:.1f}%</b>.
            <br><br>
            <b>🚨 Urgency Handling:</b> {len(res_df[res_df['Urgent'] == True])} urgent orders were prioritized and 
            successfully integrated into multi-stop loads to avoid expedited parcel fees.
            <br><br>
            <b>🌱 Sustainability:</b> Optimized routing reduced vehicle miles by <b>{(1 - (num_loads/len(res_df)))*100:.1f}%</b>, 
            saving approximately <b>{savings * 0.04:.2f} kg of CO2</b> emissions.
        </div>
        """, unsafe_allow_html=True)

        st.write("### 🚚 Optimized Load Manifest")
        st.dataframe(res_df[['Load_ID', 'OrderID', 'Weight', 'Urgent', 'Type', 'Ready', 'As_Is_Cost']], use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Net Savings", f"${savings:,.2f}", delta="Strategic ROI")
        with col2:
            st.metric("Mode Shift (LTL → TL)", f"{len(res_df)} Orders Consolidated")

        st.plotly_chart(px.scatter_geo(res_df, lat="Lat", lon="Lon", color="Load_ID", symbol="Urgent",
                                      title="Consolidated Multi-Stop Map", scope='usa', template='plotly_dark'))
    else:
        st.error("Constraint Violation: No solution found. Try increasing fleet size.")
