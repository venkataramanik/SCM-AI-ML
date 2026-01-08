import streamlit as st
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import plotly.express as px

# --- 1. RATE & BUSINESS LOGIC ENGINE ---
class LogisticsEngine:
    @staticmethod
    def get_parcel_rate(weight):
        """Simulates base parcel rates + weight surcharges."""
        return 15.0 + (weight * 0.45)

    @staticmethod
    def get_ltl_rate(weight, distance):
        """Simulates LTL Class 70 rates with a $150 minimum."""
        if weight < 150: return LogisticsEngine.get_parcel_rate(weight)
        rate_cwt = (distance * 0.15) + 35.0 
        return max((weight / 100) * rate_cwt, 150.0)

    @staticmethod
    def get_tl_base_cost(distance):
        """Flat Truckload rate simulation."""
        return 500 + (distance * 2.50)

# --- 2. OPTIMIZATION SOLVER (OR-TOOLS) ---
def solve_logistics_vrp(df, rules):
    # Data preparation
    num_locations = len(df)
    # Depot is index 0
    locations = df[['Lat', 'Lon']].values
    weights = df['Weight'].tolist()
    
    manager = pywrapcp.RoutingIndexManager(num_locations, rules['fleet_size'], 0)
    routing = pywrapcp.RoutingModel(manager)

    # Distance Logic
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # Haversine-lite distance calculation
        return int(np.hypot(locations[from_node][0] - locations[to_node][0],
                            locations[from_node][1] - locations[to_node][1]) * 100)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Constraint: Weight Capacity
    def weight_callback(from_index):
        return weights[manager.IndexToNode(from_index)]
    
    weight_callback_index = routing.RegisterUnaryTransitCallback(weight_callback)
    routing.AddDimensionWithVehicleCapacity(
        weight_callback_index, 0, [rules['max_weight']] * rules['fleet_size'], True, 'Capacity'
    )

    # Constraint: Max Stops per Truck
    # Each stop adds '1' to the counter
    def stop_callback(from_index):
        return 1 if manager.IndexToNode(from_index) != 0 else 0
    
    stop_callback_index = routing.RegisterUnaryTransitCallback(stop_callback)
    routing.AddDimensionWithVehicleCapacity(
        stop_callback_index, 0, [rules['max_stops']] * rules['fleet_size'], True, 'Stops'
    )

    # Search Parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    
    solution = routing.SolveWithParameters(search_parameters)
    return routing, manager, solution

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Shipment Load Builder", layout="wide")

st.title("🚛 Shipment Load Builder & Mode Optimizer")
st.markdown("Optimize multi-stop truckloads using Google OR-Tools.")

# Sidebar Configuration
with st.sidebar:
    st.header("Planning Parameters")
    max_w = st.number_input("Max Truck Weight (Lbs)", value=45000)
    max_s = st.slider("Max Stops per Truck", 1, 10, 5)
    fleet_size = st.number_input("Trucks Available", value=10)
    stop_fee = st.number_input("Stop-off Charge ($)", value=150)
    
    st.divider()
    if st.button("Load Sample Dataset"):
        # Generate dummy data centered around a Midwest Hub
        data = {
            'OrderID': [f"ORD-{i}" for i in range(101, 116)],
            'Weight': [450, 12000, 8500, 50, 15000, 2200, 400, 9000, 11000, 30, 6000, 500, 150, 8000, 200],
            'Lat': [39.0, 39.5, 40.2, 38.8, 41.5, 40.0, 39.2, 41.0, 38.5, 40.5, 39.8, 40.1, 38.2, 41.2, 39.0],
            'Lon': [-94.0, -94.5, -93.5, -95.0, -92.0, -94.2, -93.8, -91.5, -95.5, -93.0, -94.1, -93.9, -96.0, -91.8, -94.5],
            'Dist_to_DC': [10, 60, 110, 45, 250, 35, 25, 210, 180, 85, 40, 55, 240, 230, 15]
        }
        # First row is the DC (Weight 0)
        dc = {'OrderID': 'DC-HUB', 'Weight': 0, 'Lat': 39.0, 'Lon': -94.5, 'Dist_to_DC': 0}
        df_dc = pd.DataFrame([dc])
        df_orders = pd.DataFrame(data)
        st.session_state['data'] = pd.concat([df_dc, df_orders]).reset_index(drop=True)

# Main Dashboard
if 'data' in st.session_state:
    df = st.session_state['data']
    
    col_map, col_stats = st.columns([2, 1])
    
    with col_map:
        st.subheader("Order Geography")
        fig = px.scatter_mapbox(df[1:], lat="Lat", lon="Lon", size="Weight", color="Weight",
                              hover_name="OrderID", zoom=5, mapbox_style="carto-positron")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("🚀 Build Optimized Loads"):
        routing, manager, solution = solve_logistics_vrp(df, {
            'max_weight': max_w, 'max_stops': max_s, 'fleet_size': fleet_size
        })
        
        if solution:
            loads = []
            for vehicle_id in range(fleet_size):
                index = routing.Start(vehicle_id)
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    if node_index != 0: # Exclude DC from result table
                        row = df.iloc[node_index].copy()
                        row['Load_ID'] = f"TL-LOAD-{vehicle_id + 1}"
                        row['Stop_Sequence'] = len([l for l in loads if l['Load_ID'] == row['Load_ID']]) + 1
                        
                        # Apply Mode-Shift Analysis for this specific order
                        row['Parcel_Cost'] = LogisticsEngine.get_parcel_rate(row['Weight'])
                        row['LTL_Cost'] = LogisticsEngine.get_ltl_rate(row['Weight'], row['Dist_to_DC'])
                        loads.append(row)
                    index = solution.Value(routing.NextVar(index))
            
            final_df = pd.DataFrame(loads)
            
            # --- Results Presentation ---
            st.subheader("📦 Final Dispatch Plan")
            
            # 1. Summary Metrics
            total_ltl = final_df['LTL_Cost'].sum()
            num_loads = final_df['Load_ID'].nunique()
            # Est. TL Cost: (Avg Distance Rate) + (Stop Fees)
            est_tl_cost = (num_loads * 950) + (len(final_df) * stop_fee)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Orders Processed", len(final_df))
            m2.metric("Total Load Units", num_loads)
            m3.metric("Estimated Savings", f"${(total_ltl - est_tl_cost):,.2f}", delta="vs Standalone LTL")

            # 2. Detailed Table
            st.dataframe(
                final_df[['Load_ID', 'Stop_Sequence', 'OrderID', 'Weight', 'LTL_Cost', 'Parcel_Cost']], 
                use_container_width=True,
                hide_index=True
            )
            
            # 3. Mode Shift Insight
            st.subheader("💡 Mode Shift Recommendations")
            parcel_shift = final_df[final_df['Parcel_Cost'] < final_df['LTL_Cost']]
            if not parcel_shift.empty:
                st.warning(f"Note: {len(parcel_shift)} orders are cheaper via Parcel than LTL. Consider pulling these from the Truckload.")
                st.dataframe(parcel_shift[['OrderID', 'Weight', 'Parcel_Cost', 'LTL_Cost']])
            
            st.download_button("Download Load Plan", final_df.to_csv(), "load_plan.csv")
        else:
            st.error("Solver could not find a valid solution. Try increasing the fleet size or decreasing weight constraints.")
else:
    st.info("Please click 'Load Sample Dataset' in the sidebar or upload your order file to begin.")
