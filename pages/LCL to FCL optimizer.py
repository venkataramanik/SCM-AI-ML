import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Logistics Optimization Engine", layout="wide")

# --- 2. INDUSTRY STANDARD CONSTANTS ---
FCL_40HC_RATE = 3800.00   # Standard Ocean Freight
FCL_DRAYAGE = 450.00      # Plant to Port Trucking
LCL_WM_RATE = 165.00      # Per Revenue Ton (W/M)
LCL_CFS_FEE = 25.00       # Handling / Documentation
FCL_CAPACITY_CBM = 65.0   # 90% Efficiency threshold
FCL_MAX_KG = 20500.0      # Payload limit for road/rail
LCL_LEAD_TIME_PENALTY = 8 # Days lost in CFS hubs
CO2_FACTOR = 0.15         # Metric Tons CO2 per CBM saved

# --- 3. DATA GENERATION ENGINE (1,000 SHIPMENTS) ---
@st.cache_data
def generate_simulation_data():
    np.random.seed(42)
    start_date = datetime(2026, 1, 1)
    data = []
    for i in range(1000):
        vol = np.random.uniform(1.2, 14.0) # Range of LCL sizes
        # Random density (Weight vs Volume)
        wgt = vol * np.random.uniform(180, 1150) 
        data.append({
            'Order_ID': f'ORD-{i:04d}',
            'Ready_Date': start_date + timedelta(days=np.random.randint(0, 90)),
            'Vol_CBM': round(vol, 2),
            'Wgt_KG': round(wgt, 2),
            'Vol_CFT': round(vol * 35.31, 2),
            'Wgt_LB': round(wgt * 2.204, 2)
        })
    df = pd.DataFrame(data)
    # W/M Calculation: 1000kg = 1 CBM
    df['Rev_Ton'] = df[['Vol_CBM', 'Wgt_KG']].apply(lambda x: max(x[0], x[1]/1000), axis=1)
    df['LCL_Baseline_Cost'] = df['Rev_Ton'] * (LCL_WM_RATE + LCL_CFS_FEE)
    return df

# --- 4. CORE OPTIMIZATION LOGIC ---
def run_optimization(df, current_fcl_rate):
    df['Week'] = df['Ready_Date'].dt.isocalendar().week
    weekly_summary = df.groupby('Week').agg({
        'Vol_CBM': 'sum',
        'Wgt_KG': 'sum',
        'LCL_Baseline_Cost': 'sum',
        'Order_ID': 'count'
    }).reset_index()

    # Decision: How many FCLs can we build?
    # Logic: Uses the restrictive factor (Weight or Volume)
    weekly_summary['FCL_Count'] = weekly_summary.apply(
        lambda x: int(max(x['Vol_CBM'] // FCL_CAPACITY_CBM, x['Wgt_KG'] // FCL_MAX_KG)), axis=1
    )
    
    weekly_summary['FCL_Total_Cost'] = weekly_summary['FCL_Count'] * (current_fcl_rate + FCL_DRAYAGE)
    
    # Residuals go back to LCL
    weekly_summary['Residual_CBM'] = weekly_summary['Vol_CBM'] - (weekly_summary['FCL_Count'] * FCL_CAPACITY_CBM)
    weekly_summary['Residual_LCL_Cost'] = weekly_summary['Residual_CBM'].apply(lambda x: max(0, x * (LCL_WM_RATE + LCL_CFS_FEE)))
    
    weekly_summary['Optimized_Total_Cost'] = weekly_summary['FCL_Total_Cost'] + weekly_summary['Residual_LCL_Cost']
    weekly_summary['Savings'] = weekly_summary['LCL_Baseline_Cost'] - weekly_summary['Optimized_Total_Cost']
    
    return weekly_summary

# --- 5. STREAMLIT INTERFACE ---
st.title("📦 DC Operations: LCL-to-FCL Transformation Engine")
st.sidebar.header("Market Variable Controls")
live_fcl_rate = st.sidebar.slider("Market FCL Rate ($)", 2000, 8000, 3800)

raw_df = generate_1k_shipments = generate_simulation_data()
opt_df = run_optimization(raw_df, live_fcl_rate)

# TABS FOR ORGANIZATION
tab1, tab2, tab3, tab4 = st.tabs(["📊 Executive Dashboard", "📈 Sensitivity Analysis", "📄 Simulation Data", "📘 Logic & User Guide"])

with tab1:
    # High Level Metrics
    m1, m2, m3, m4 = st.columns(4)
    total_savings = opt_df['Savings'].sum()
    total_baseline = raw_df['LCL_Baseline_Cost'].sum()
    
    m1.metric("Total Baseline Spend", f"${total_baseline/1e3:.1f}K")
    m2.metric("Projected Savings", f"${total_savings/1e3:.1f}K", delta=f"{(total_savings/total_baseline)*100:.1f}%")
    m3.metric("Lead Time Recovery", f"{opt_df['FCL_Count'].sum() * LCL_LEAD_TIME_PENALTY} Days", help="Velocity gained by bypassing CFS")
    m4.metric("Carbon Avoided", f"{(total_savings/3800)*2.4:.1f} Tons CO2")

    # Visualizing the Spend Shift
    fig = px.bar(opt_df, x='Week', y=['LCL_Baseline_Cost', 'Optimized_Total_Cost'], 
                 barmode='group', title="Financial Impact: Weekly LCL vs. Strategic FCL",
                 labels={'value': 'Cost ($)', 'variable': 'Scenario'},
                 color_discrete_sequence=['#ff4b4b', '#00cc96'])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Market Rate Sensitivity Analysis")
    st.write("Stress-testing the consolidation strategy against rising FCL costs.")
    
    rates = np.arange(2000, 8500, 500)
    sens_data = []
    for r in rates:
        temp_opt = run_optimization(raw_df, r)
        sens_data.append({'FCL_Rate': r, 'Savings': temp_opt['Savings'].sum()})
    
    sens_df = pd.DataFrame(sens_data)
    fig_sens = px.line(sens_df, x='FCL_Rate', y='Savings', markers=True, 
                       title="Savings Decay relative to FCL Market Rate")
    fig_sens.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even Point")
    st.plotly_chart(fig_sens, use_container_width=True)

with tab3:
    st.subheader("Raw Shipment Pool (1,000 Data Points)")
    st.dataframe(raw_df, use_container_width=True)

with tab4:
    st.header("Executive User Guide & Logic Framework")
    
    st.markdown("""
    ### 1. Methodology: The 'Consolidation Engine'
    This tool solves the **Freight Consolidation Problem** by shifting the DC from a 'Reactive' shipping model to a 'Proactive' one.
    * **W/M Rule (Weight or Measure):** Following ocean standards, we calculate the 'Revenue Ton'. If cargo is extremely heavy ($>1,000kg/CBM$), the cost is calculated by weight.
    * **Temporal Binning:** Shipments are 'held' at the DC for up to 7 days to find volume synergies for a specific destination port.
    
    ### 2. Strategic Assumptions
    * **Utilization Safety (15%):** We assume 15% of the container is 'lost space' due to non-perfect stacking.
    * **Lead Time Recovery:** FCL containers are sealed at the DC and go straight to the vessel. LCL shipments spend an average of **8 days** being handled, sorted, and re-stuffed at 3rd party CFS hubs.
    
    ### 3. Risk Management (Sensitivity)
    The Sensitivity Analysis identifies the **Break-even Point**. If FCL rates spike due to market volatility (e.g., Red Sea disruptions), this tool identifies at exactly what price point the DC should revert to LCL to protect margins.
    
    ### 4. ESG & Sustainability
    By maximizing container utilization, we reduce the total number of vehicles required for port delivery, directly lowering the Scope 3 carbon footprint of the outbound supply chain.
    """)
