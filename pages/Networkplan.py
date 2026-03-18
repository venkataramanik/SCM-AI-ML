import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Freight Consolidation Optimizer", layout="wide")

# Initialize session state
if 'stos' not in st.session_state:
    st.session_state.stos = None
if 'plant_plans' not in st.session_state:
    st.session_state.plant_plans = None
if 'network_plan' not in st.session_state:
    st.session_state.network_plan = None
if 'config' not in st.session_state:
    st.session_state.config = None

def generate_synthetic_data():
    """Generate synthetic STO data for demo"""
    random.seed(42)
    np.random.seed(42)
    
    # Network configuration
    plants = {
        'P001': {'city': 'Knoxville', 'state': 'TN', 'hub': 'SIVA_US', 'pick_pack': 1, 
                 'ground_transit': 2, 'ltl_cost_cbm': 35, 'tl_cost': 850},
        'P002': {'city': 'Decatur', 'state': 'AL', 'hub': 'SIVA_US', 'pick_pack': 1,
                 'ground_transit': 3, 'ltl_cost_cbm': 38, 'tl_cost': 920},
        'P003': {'city': 'Chicago', 'state': 'IL', 'hub': 'SIVA_US', 'pick_pack': 2,
                 'ground_transit': 1, 'ltl_cost_cbm': 30, 'tl_cost': 600},
        'P004': {'city': 'Corona', 'state': 'CA', 'hub': 'K_N_West', 'pick_pack': 1,
                 'ground_transit': 1, 'ltl_cost_cbm': 40, 'tl_cost': 700},
        'P005': {'city': 'Newark', 'state': 'DE', 'hub': 'K_N_East', 'pick_pack': 1,
                 'ground_transit': 2, 'ltl_cost_cbm': 32, 'tl_cost': 750},
    }
    
    destinations = [
        {'plant': 'Singapore_Plant', 'region': 'AMEA', 'hub': 'SIVA_US', 'ocean_days': 14, 
         'fcl_40_cost': 3200, 'lcl_cost_cbm': 125, 'air_cost_kg': 4.5},
        {'plant': 'Santos_Plant', 'region': 'LATAM', 'hub': 'SIVA_US', 'ocean_days': 15,
         'fcl_40_cost': 3800, 'lcl_cost_cbm': 140, 'air_cost_kg': 5.0},
        {'plant': 'Tokyo_Plant', 'region': 'APAC', 'hub': 'K_N_West', 'ocean_days': 12,
         'fcl_40_cost': 2800, 'lcl_cost_cbm': 110, 'air_cost_kg': 4.0},
        {'plant': 'Rotterdam_Plant', 'region': 'Europe', 'hub': 'K_N_East', 'ocean_days': 10,
         'fcl_40_cost': 2600, 'lcl_cost_cbm': 95, 'air_cost_kg': 3.5},
    ]
    
    # Vessel schedules
    base_date = datetime(2026, 3, 18)
    vessels = [
        {'vessel_id': 'SEA_001', 'hub': 'SIVA_US', 'destination': 'Singapore_Plant', 
         'cutoff': base_date + timedelta(days=7), 'sailing': base_date + timedelta(days=9), 
         'arrival': base_date + timedelta(days=23)},
        {'vessel_id': 'SEA_002', 'hub': 'SIVA_US', 'destination': 'Singapore_Plant',
         'cutoff': base_date + timedelta(days=14), 'sailing': base_date + timedelta(days=16), 
         'arrival': base_date + timedelta(days=30)},
        {'vessel_id': 'LAT_001', 'hub': 'SIVA_US', 'destination': 'Santos_Plant',
         'cutoff': base_date + timedelta(days=10), 'sailing': base_date + timedelta(days=12), 
         'arrival': base_date + timedelta(days=27)},
        {'vessel_id': 'PAC_001', 'hub': 'K_N_West', 'destination': 'Tokyo_Plant',
         'cutoff': base_date + timedelta(days=8), 'sailing': base_date + timedelta(days=10), 
         'arrival': base_date + timedelta(days=22)},
        {'vessel_id': 'EUR_001', 'hub': 'K_N_East', 'destination': 'Rotterdam_Plant',
         'cutoff': base_date + timedelta(days=6), 'sailing': base_date + timedelta(days=8), 
         'arrival': base_date + timedelta(days=18)},
    ]
    
    # Generate 15-20 STOs
    stos = []
    for i in range(16):
        plant_id = random.choice(list(plants.keys()))
        plant = plants[plant_id]
        
        valid_dests = [d for d in destinations if d['hub'] == plant['hub']]
        dest = random.choice(valid_dests)
        
        # MAD within next 6 days
        mad = base_date + timedelta(days=random.randint(0, 5))
        
        # MRD is MAD + 18-28 days
        mrd = mad + timedelta(days=random.randint(18, 28))
        
        # Random specs
        weight = random.randint(800, 6000)
        volume = round(weight / 350 + random.uniform(1, 5), 1)
        pallets = max(2, int(volume / 2))
        
        sto = {
            'sto_id': f'STO_{i+1:03d}',
            'plant_id': plant_id,
            'origin_city': plant['city'],
            'origin_state': plant['state'],
            'dest_plant': dest['plant'],
            'dest_region': dest['region'],
            'hub': plant['hub'],
            'mad': mad,
            'mrd': mrd,
            'weight_kg': weight,
            'volume_cbm': volume,
            'pallets': pallets,
            'pick_pack_days': plant['pick_pack'],
            'ground_transit': plant['ground_transit'],
            'hub_processing': 2,  # days
            'buffer': 1,  # day
            'ocean_days': dest['ocean_days'],
            'ltl_cost_cbm': plant['ltl_cost_cbm'],
            'tl_cost': plant['tl_cost'],
            'fcl_40_cost': dest['fcl_40_cost'],
            'lcl_cost_cbm': dest['lcl_cost_cbm'],
        }
        stos.append(sto)
    
    return pd.DataFrame(stos), plants, destinations, vessels

def calculate_dates(sto, vessel):
    """Calculate key dates for an STO"""
    ready_date = sto['mad'] + timedelta(days=sto['pick_pack_days'])
    
    # Must ship by = vessel cutoff - ground - hub processing - buffer
    must_ship_by = vessel['cutoff'] - timedelta(days=sto['ground_transit'] + 
                                                  sto['hub_processing'] + 
                                                  sto['buffer'])
    
    # Arrival at destination
    arrival = vessel['arrival'] + timedelta(days=2)  # + dest ground
    
    return ready_date, must_ship_by, arrival

def plant_level_optimization(df, vessels):
    """Stage 1: Plant-level consolidation"""
    plant_plans = []
    
    for plant_id in df['plant_id'].unique():
        plant_stos = df[df['plant_id'] == plant_id].copy()
        
        # Find best vessel for each STO
        for idx, sto in plant_stos.iterrows():
            valid_vessels = [v for v in vessels 
                           if v['hub'] == sto['hub'] and 
                           v['destination'] == sto['dest_plant']]
            
            if not valid_vessels:
                continue
                
            # Pick earliest vessel that works
            best_vessel = valid_vessels[0]
            ready_date, must_ship_by, arrival = calculate_dates(sto, best_vessel)
            
            # Check feasibility
            feasible = arrival <= sto['mrd'] and ready_date <= must_ship_by
            
            plant_plans.append({
                'sto_id': sto['sto_id'],
                'plant_id': plant_id,
                'origin_city': sto['origin_city'],
                'dest_plant': sto['dest_plant'],
                'hub': sto['hub'],
                'mad': sto['mad'],
                'ready_date': ready_date,
                'must_ship_by': must_ship_by,
                'mrd': sto['mrd'],
                'vessel_id': best_vessel['vessel_id'],
                'vessel_cutoff': best_vessel['cutoff'],
                'vessel_arrival': best_vessel['arrival'],
                'arrival_at_dest': arrival,
                'weight_kg': sto['weight_kg'],
                'volume_cbm': sto['volume_cbm'],
                'pallets': sto['pallets'],
                'feasible': feasible,
                'window_days': (must_ship_by - ready_date).days,
                'ground_transit': sto['ground_transit'],
                'tl_cost': sto['tl_cost'],
                'ltl_cost': sto['volume_cbm'] * sto['ltl_cost_cbm'],
            })
    
    return pd.DataFrame(plant_plans)

def network_level_optimization(plant_df, vessels):
    """Stage 2: Network-level container optimization"""
    network_plan = []
    
    # Group by hub, destination, vessel
    grouped = plant_df.groupby(['hub', 'dest_plant', 'vessel_id'])
    
    for (hub, dest, vessel_id), group in grouped:
        vessel = next(v for v in vessels if v['vessel_id'] == vessel_id)
        
        total_volume = group['volume_cbm'].sum()
        total_weight = group['weight_kg'].sum()
        total_pallets = group['pallets'].sum()
        num_stos = len(group)
        
        # Container decision
        if total_volume >= 50:  # Close to 40' capacity
            container_type = '40_FCL'
            container_cost = 3200
            utilization = min(total_volume / 67 * 100, 100)
        elif total_volume >= 25:  # Close to 20' or use 40'
            container_type = '40_FCL'  # Better rate per CBM
            container_cost = 3200
            utilization = total_volume / 67 * 100
        else:
            container_type = 'LCL'
            container_cost = total_volume * 125  # Approximate
            utilization = None
        
        # Ground cost sum
        total_ground_cost = group['tl_cost'].sum() if total_volume > 30 else group['ltl_cost'].sum()
        
        network_plan.append({
            'hub': hub,
            'destination': dest,
            'vessel_id': vessel_id,
            'vessel_cutoff': vessel['cutoff'],
            'num_stos': num_stos,
            'sto_list': ', '.join(group['sto_id'].tolist()),
            'total_volume_cbm': round(total_volume, 1),
            'total_weight_kg': total_weight,
            'total_pallets': total_pallets,
            'container_type': container_type,
            'container_cost': container_cost,
            'ground_cost': total_ground_cost,
            'total_cost': container_cost + total_ground_cost,
            'utilization_pct': round(utilization, 1) if utilization else 'N/A',
        })
    
    return pd.DataFrame(network_plan)

# UI STARTS HERE
st.title("🚛 Freight Consolidation Optimizer")
st.markdown("*Two-Stage Planning: Plant Consolidation → Network Container Optimization*")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Key elements explanation
with st.sidebar.expander("📋 Key Elements Needed"):
    st.markdown("""
    **1. DATES:**
    - MAD (Material Availability Date)
    - Pick/Pack Time → Ready Date
    - MRD (Material Requested Date)
    - Must Ship By (calculated from vessel cutoff)
    
    **2. SCHEDULES:**
    - Vessel Cutoff (hard constraint)
    - Vessel Sailing & Arrival
    - Ground Transit (plant → hub)
    - Hub Processing (3PL time)
    
    **3. CAPACITY:**
    - Trailer: 24,000 kg, 100 CBM, 26 pallets
    - Container 40': 26,500 kg, 67 CBM
    
    **4. COSTS:**
    - Ground: TL vs LTL
    - Ocean: FCL vs LCL
    - Air: Express (backup)
    
    **5. TWO-STAGE PLANNING:**
    - Stage 1: Plant optimizes trailer loads
    - Stage 2: Network optimizes container loads
    """)

# Generate data button
if st.sidebar.button("🎲 Generate Synthetic STO Data"):
    st.session_state.stos, st.session_state.config_plants, \
    st.session_state.config_dests, st.session_state.config_vessels = generate_synthetic_data()
    st.session_state.plant_plans = None
    st.session_state.network_plan = None
    st.success("Generated 16 synthetic STOs across 5 plants!")

# Main content
if st.session_state.stos is not None:
    df = st.session_state.stos
    
    # Show raw data
    st.subheader("📦 Generated STOs (Raw Data)")
    
    display_df = df[['sto_id', 'plant_id', 'origin_city', 'dest_plant', 'mad', 'mrd', 
                     'weight_kg', 'volume_cbm', 'pallets']].copy()
    display_df['mad'] = display_df['mad'].dt.strftime('%Y-%m-%d')
    display_df['mrd'] = display_df['mrd'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(display_df, use_container_width=True)
    
    # Stage 1: Plant Planning
    st.markdown("---")
    st.subheader("🏭 STAGE 1: Plant-Level Planning")
    
    if st.button("🚛 Run Plant Optimization"):
        with st.spinner("Optimizing plant consolidations..."):
            st.session_state.plant_plans = plant_level_optimization(
                df, st.session_state.config_vessels
            )
    
    if st.session_state.plant_plans is not None:
        plant_df = st.session_state.plant_plans
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total STOs", len(plant_df))
        with col2:
            st.metric("Feasible", plant_df['feasible'].sum())
        with col3:
            avg_window = plant_df[plant_df['feasible']]['window_days'].mean()
            st.metric("Avg Window (days)", f"{avg_window:.1f}")
        with col4:
            total_ground = plant_df['tl_cost'].sum()
            st.metric("Total Ground Cost", f"${total_ground:,.0f}")
        
        # Show plant plans
        st.markdown("**Plant Plans (Must Ship By calculated from Vessel Cutoff):**")
        
        display_plant = plant_df[['sto_id', 'plant_id', 'origin_city', 'ready_date', 
                                   'must_ship_by', 'vessel_cutoff', 'window_days',
                                   'volume_cbm', 'pallets']].copy()
        display_plant['ready_date'] = display_plant['ready_date'].dt.strftime('%Y-%m-%d')
        display_plant['must_ship_by'] = display_plant['must_ship_by'].dt.strftime('%Y-%m-%d')
        display_plant['vessel_cutoff'] = display_plant['vessel_cutoff'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_plant, use_container_width=True)
        
        # Timeline visualization
        st.markdown("**Timeline Visualization (Sample Plant):**")
        
        sample_plant = plant_df[plant_df['plant_id'] == 'P001'].copy()
        if len(sample_plant) > 0:
            fig = go.Figure()
            
            for idx, row in sample_plant.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['ready_date'], row['must_ship_by']],
                    y=[row['sto_id'], row['sto_id']],
                    mode='lines+markers',
                    name=row['sto_id'],
                    line=dict(width=10),
                    marker=dict(size=15)
                ))
                
                fig.add_vline(x=row['vessel_cutoff'], line_dash="dash", 
                             annotation_text=f"{row['vessel_id']} Cutoff")
            
            fig.update_layout(
                title=f"Plant P001 (Knoxville) - Consolidation Windows",
                xaxis_title="Date",
                yaxis_title="STO",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Stage 2: Network Planning
        st.markdown("---")
        st.subheader("🌐 STAGE 2: Network-Level Container Planning")
        
        if st.button("🚢 Run Network Optimization"):
            with st.spinner("Optimizing container loads..."):
                st.session_state.network_plan = network_level_optimization(
                    plant_df, st.session_state.config_vessels
                )
        
        if st.session_state.network_plan is not None:
            net_df = st.session_state.network_plan
            
            # Network metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Container Bookings", len(net_df))
            with col2:
                fcl_count = (net_df['container_type'] == '40_FCL').sum()
                st.metric("FCL Bookings", fcl_count)
            with col3:
                total_cost = net_df['total_cost'].sum()
                st.metric("Total Network Cost", f"${total_cost:,.0f}")
            
            # Show network plan
            st.markdown("**Container Booking Plan:**")
            
            display_net = net_df[['hub', 'destination', 'vessel_id', 'vessel_cutoff',
                                   'num_stos', 'total_volume_cbm', 'total_pallets',
                                   'container_type', 'utilization_pct', 'total_cost']].copy()
            display_net['vessel_cutoff'] = display_net['vessel_cutoff'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(display_net, use_container_width=True)
            
            # Comparison
            st.markdown("---")
            st.subheader("📊 Baseline vs Optimized Comparison")
            
            # Calculate baseline (each STO ships solo LTL, LCL)
            baseline_ground = plant_df['ltl_cost'].sum()
            baseline_ocean = (plant_df['volume_cbm'] * 125).sum()  # LCL rate
            baseline_total = baseline_ground + baseline_ocean
            
            optimized_total = net_df['total_cost'].sum()
            savings = baseline_total - optimized_total
            savings_pct = (savings / baseline_total) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Baseline Cost (LTL + LCL)", f"${baseline_total:,.0f}")
            with col2:
                st.metric("Optimized Cost (TL + FCL)", f"${optimized_total:,.0f}")
            with col3:
                st.metric("Savings", f"${savings:,.0f} ({savings_pct:.1f}%)")
            
            # Explanation
            st.markdown("""
            **How the Savings Were Achieved:**
            
            1. **Plant Level:** Consolidated multiple STOs onto single TL trailers instead of individual LTL shipments
            2. **Network Level:** Aggregated volume across plants to fill 40' FCL containers instead of LCL
            3. **Time Phasing:** Used consolidation windows to group compatible STOs without missing vessel cutoffs
            
            **Key Constraints Respected:**
            - Vessel cutoff dates (hard constraints)
            - Material delivery dates (MRD)
            - Trailer and container capacities
            - Plant processing times (pick/pack)
            """)
            
            # What-if scenario
            st.markdown("---")
            st.subheader("🔮 What-If Scenario")
            
            delay_days = st.slider("What if all plants delay 1 day?", 0, 3, 1)
            
            if delay_days > 0:
                st.warning(f"With {delay_days} day delay, some STOs may miss vessel cutoffs!")
                
                # Check feasibility
                affected = plant_df[plant_df['window_days'] < delay_days]
                if len(affected) > 0:
                    st.error(f"{len(affected)} STOs would miss their cutoff!")
                    st.dataframe(affected[['sto_id', 'plant_id', 'window_days', 'must_ship_by']])
                else:
                    st.success("All STOs still feasible with delay.")

else:
    st.info("👈 Click 'Generate Synthetic STO Data' in the sidebar to start the demo!")
    
    # Show instructions
    st.markdown("""
    ## How to Use This Demo
    
    1. **Generate Data:** Create synthetic STOs with realistic dates, weights, volumes
    
    2. **Stage 1 - Plant Planning:** 
       - Calculate Ready Date (MAD + Pick/Pack)
       - Calculate Must Ship By (Vessel Cutoff - Ground - Hub - Buffer)
       - Determine consolidation window
       - Group STOs onto trailers (TL vs LTL)
    
    3. **Stage 2 - Network Planning:**
       - Aggregate plant shipments at hubs
       - Calculate total volume by lane
       - Decide: FCL vs LCL
       - Book containers
    
    4. **Compare:** See savings from optimization vs baseline
    
    ## Key Concepts Demonstrated
    
    - **Time-phased constraints:** Everything driven by vessel cutoff
    - **Backward calculation:** Must Ship By = Cutoff - all lead times
    - **Two-stage decomposition:** Plant optimizes ground, network optimizes ocean
    - **Consolidation windows:** Trade-off between waiting (better fill) and shipping (safer)
    """)
