import streamlit as st
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import plotly.express as px
import random

# --- 1. SMART DATA GENERATOR ---
def generate_complex_data(n=45):
    types = ['Food', 'Chemicals', 'General', 'Retail']
    slots = ['08:00-12:00', '13:00-17:00']
    data = []
    
    # Depot (Index 0) - The Warehouse
    data.append({
        'OrderID': 'DC-HUB', 'Weight': 0, 'Volume': 0, 'Lat': 39.0, 'Lon': -94.5, 
        'Ready': 0, 'Due': 10, 'Type': 'DC', 'Slot': 'N/A', 'Urgent': False
    })
    
    for i in range(n):
        p_type = random.choice(types)
        is_urgent = random.random() < 0.10
        
        # Cube vs Weight Logic: Retail is high volume/low weight. Chemicals are high weight.
        if p_type == 'Retail':
            weight, volume = random.randint(500, 3000), random.randint(400, 800)
        else:
            weight, volume = random.randint(5000, 15000), random.randint(100, 400)
            
        data.append({
            'OrderID': f"ORD-{2000+i}",
            'Weight': weight,
            'Volume': volume,
            'Lat': 39.0 + random.uniform(-5.0, 5.0),
            'Lon': -94.5 + random.uniform(-5.0, 5.0),
            'Ready': 0 if is_urgent else random.randint(0, 3),
            'Due': 1 if is_urgent else random.randint(4, 6),
            'Type': p_type,
            'Slot': random.choice(slots),
            'Urgent': is_urgent
        })
    return pd.DataFrame(data)

# --- 2. THE OPTIMIZATION ENGINE ---
def run_advanced_optimization(df, rules):
    num_locs = len(df)
    # 20 trucks available
    manager = pywrapcp.RoutingIndexManager(num_locs, 20, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_fn(f_idx, t_idx):
        f, t = manager.IndexToNode(f_idx), manager.IndexToNode(t_idx)
        return int(np.hypot(df.iloc[f]['Lat']-df.iloc[t]['Lat'], df.iloc[f]['Lon']-df.iloc[t]['Lon']) * 100)
    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_fn))
    
    # Weight Constraint
    def weight_fn(idx): return int(df.iloc[manager.IndexToNode(idx)]['Weight'])
    routing.AddDimension(routing.RegisterUnaryTransitCallback(weight_fn), 0, rules['max_w'], True, 'Weight')

    # Volume Constraint
    def vol_fn(idx): return int(df.iloc[manager.IndexToNode(idx)]['Volume'])
    routing.AddDimension(routing.RegisterUnaryTransitCallback(vol_fn), 0, rules['max_v'], True, 'Volume')

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    return routing, manager, solution if (solution := routing.SolveWithParameters(search_params)) else (None, None, None)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Strategic Freight Simulator", layout="wide")

if 'sim_data' not in st.session_state:
    st.session_state.sim_data = generate_complex_data(45)

st.title("🚛 Strategic Load Factor & Smoothing Simulator")

# --- STEP 1: SHOW RAW ORDERS FIRST ---
st.subheader("📋 Phase 1: Unconstrained Demand (Incoming Orders)")
st.info("Below are all orders currently in the system across various Ready Dates and Product Types.")
st.dataframe(st.session_state.sim_data.iloc[1:], use_container_width=True, hide_index=True)

# Dashboard Sidebar
with st.sidebar:
    st.header("Simulation Settings")
    horizon = st.slider("Smoothing Horizon (Days)", 0, 4, 2, help="Allows AI to pull orders from future days to fill today's trucks.")
    max_w = st.slider("Truck Weight Cap (Lbs)", 20000, 48000, 44000)
    max_v = st.slider("Truck Cube Cap (CuFt)", 1500, 3500, 3200)
    st.divider()
    if st.button("🔄 Refresh Data"):
        st.session_state.sim_data = generate_complex_data(45)
        st.rerun()

# --- STEP 2: RUN OPTIMIZATION ---
st.divider()
st.subheader(f"✨ Phase 2: Optimized Multi-Stop Load Plan")

raw_df = st.session_state.sim_data
# Filtering based on Horizon + Urgent Flags
filtered_df = raw_df[(raw_df['Ready'] <= horizon) | (raw_df['Urgent'] == True) | (raw_df['Type'] == 'DC')].reset_index(drop=True)

if st.button("🚀 Calculate Optimized Loads"):
    routing, manager, solution = run_advanced_optimization(filtered_df, {'max_w': max_w, 'max_v': max_v})

    if solution:
        results = []
        for v_id in range(20):
            idx = routing.Start(v_id)
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node != 0:
                    item = filtered_df.iloc[node].copy()
                    item['Load_ID'] = f"TL-{100+v_id}"
                    results.append(item)
                idx = solution.Value(routing.NextVar(idx))
        
        res_df = pd.DataFrame(results)
        
        # --- PLANNER SUMMARY BOX ---
        st.markdown(f"""
        <div style="background-color: #f0f7ff; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff;">
            <h4 style="margin-top:0; color:#007bff;">💡 Strategic Insights</h4>
            <b>Consolidation Strategy:</b> Optimized {len(filtered_df)-1} orders into {res_df['Load_ID'].nunique()} loads. <br>
            <b>Limiting Factor:</b> { "VOLUME (Cube-out)" if res_df['Volume'].sum()/max_v > res_df['Weight'].sum()/max_w else "WEIGHT (Weigh-out)" } is the primary constraint.
        </div>
        """, unsafe_allow_html=True)

        # Performance Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Loads Built", res_df['Load_ID'].nunique())
        m2.metric("Weight Util %", f"{(res_df.groupby('Load_ID')['Weight'].sum().mean()/max_w)*100:.1f}%")
        m3.metric("Cube Util %", f"{(res_df.groupby('Load_ID')['Volume'].sum().mean()/max_v)*100:.1f}%")
        m4.metric("Consolidation Ratio", f"{(len(filtered_df)-1)/res_df['Load_ID'].nunique():.1f}:1")

        # Visualizing Weight vs Volume
        load_stats = res_df.groupby('Load_ID').agg({'Weight':'sum', 'Volume':'sum'}).reset_index()
        load_stats['Weight_Util'] = (load_stats['Weight'] / max_w) * 100
        load_stats['Volume_Util'] = (load_stats['Volume'] / max_v) * 100
        
        # Simple Plotly Express Bar Chart (Replacing go.Figure to prevent NameError)
        chart_data = pd.melt(load_stats[['Load_ID', 'Weight_Util', 'Volume_Util']], id_vars=['Load_ID'], 
                             var_name='Constraint', value_name='Utilization')
        
        fig = px.bar(chart_data, x='Load_ID', y='Utilization', color='Constraint', 
                     barmode='group', title="Truck Utilization: Weight vs Volume",
                     labels={'Utilization': 'Percent Full (%)'}, color_discrete_sequence=['#007bff', '#28a745'])
        fig.add_hline(y=100, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        st.write("### 🚚 Final Multi-Stop Load Schedule")
        st.dataframe(res_df[['Load_ID', 'OrderID', 'Type', 'Weight', 'Volume', 'Slot', 'Ready']], use_container_width=True)
    else:
        st.error("Solution Infeasible: The current orders exceed available truck capacity or logic limits. Adjust sliders.")
