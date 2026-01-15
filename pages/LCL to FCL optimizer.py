import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="LCL to FCL Optimizer", layout="wide")

# --- 2. INDUSTRY STANDARDS (The 'Truth' Data) ---
# These are used to anchor the simulation in reality
FCL_40HC_RATE = 3800.00   
FCL_DRAYAGE = 450.00      
LCL_WM_RATE = 165.00      
LCL_CFS_FEE = 25.00       
FCL_CAPACITY_CBM = 65.0   # Usable cube (accounts for 15% air loss)
FCL_MAX_KG = 20500.0      # Weight limit before 'weighing out'
LCL_LEAD_TIME_PENALTY = 8 # Days lost in 3rd party consolidation hubs

# --- 3. DATA GENERATION (1,000 SHIPMENTS) ---
@st.cache_data
def generate_simulation_data():
    np.random.seed(42)
    start_date = datetime(2026, 1, 1)
    data = []
    for i in range(1000):
        vol = np.random.uniform(1.2, 14.0) 
        wgt = vol * np.random.uniform(180, 1100) 
        data.append({
            'Order_ID': f'ORD-{i:04d}',
            'Ready_Date': start_date + timedelta(days=np.random.randint(0, 90)),
            'Vol_CBM': round(vol, 2),
            'Wgt_KG': round(wgt, 2)
        })
    df = pd.DataFrame(data)
    # W/M Rule: Revenue Ton is the max of Weight (MT) or Volume (CBM)
    df['Rev_Ton'] = df[['Vol_CBM', 'Wgt_KG']].apply(lambda x: max(x[0], x[1]/1000), axis=1)
    df['LCL_Baseline_Cost'] = df['Rev_Ton'] * (LCL_WM_RATE + LCL_CFS_FEE)
    return df

# --- 4. OPTIMIZATION LOGIC ---
def run_optimization(df, current_fcl_rate):
    df['Week'] = df['Ready_Date'].dt.isocalendar().week
    weekly_summary = df.groupby('Week').agg({
        'Vol_CBM': 'sum',
        'Wgt_KG': 'sum',
        'LCL_Baseline_Cost': 'sum',
        'Order_ID': 'count'
    }).reset_index()

    # Determine FCL Count based on the restrictive factor
    weekly_summary['FCL_Count'] = weekly_summary.apply(
        lambda x: int(max(x['Vol_CBM'] // FCL_CAPACITY_CBM, x['Wgt_KG'] // FCL_MAX_KG)), axis=1
    )
    
    weekly_summary['FCL_Total_Cost'] = weekly_summary['FCL_Count'] * (current_fcl_rate + FCL_DRAYAGE)
    
    # Residuals (overflow) revert to LCL pricing
    weekly_summary['Residual_CBM'] = weekly_summary['Vol_CBM'] - (weekly_summary['FCL_Count'] * FCL_CAPACITY_CBM)
    weekly_summary['Residual_LCL_Cost'] = weekly_summary['Residual_CBM'].apply(lambda x: max(0, x * (LCL_WM_RATE + LCL_CFS_FEE)))
    
    weekly_summary['Optimized_Total_Cost'] = weekly_summary['FCL_Total_Cost'] + weekly_summary['Residual_LCL_Cost']
    weekly_summary['Savings'] = weekly_summary['LCL_Baseline_Cost'] - weekly_summary['Optimized_Total_Cost']
    
    return weekly_summary

# --- 5. UI RENDERING ---
st.title("🚢 DC Optimization: Strategic LCL-to-FCL Conversion")
st.markdown("### Simulation Analysis: 1,000 Shipments (Plant-to-Port)")

raw_df = generate_simulation_data()
opt_df = run_optimization(raw_df, FCL_40HC_RATE)

# TOP LEVEL KPI BAR
m1, m2, m3, m4 = st.columns(4)
total_savings = opt_df['Savings'].sum()
total_baseline = raw_df['LCL_Baseline_Cost'].sum()

m1.metric("LCL Baseline Spend", f"${total_baseline/1e3:.1f}K")
m2.metric("Projected Savings", f"${total_savings/1e3:.1f}K", delta=f"{(total_savings/total_baseline)*100:.1f}%")
m3.metric("Velocity Gain", f"{opt_df['FCL_Count'].sum() * LCL_LEAD_TIME_PENALTY} Days", help="Bypassing CFS Hubs")
m4.metric("Carbon (CO2) Avoided", f"{(total_savings/3800)*2.4:.1f} Tons")

# MAIN TABS
tab1, tab2, tab3 = st.tabs(["📊 The 'So What' Analysis", "📈 Rate Sensitivity", "📘 User Guide & Logic"])

with tab1:
    # THE PRIMARY CHART
    fig = px.bar(opt_df, x='Week', y=['LCL_Baseline_Cost', 'Optimized_Total_Cost'], 
                 barmode='group', title="Financial Comparison: Weekly Spend",
                 labels={'value': 'Cost ($)', 'variable': 'Strategy'},
                 color_discrete_sequence=['#E74C3C', '#27AE60'])
    st.plotly_chart(fig, use_container_width=True)

    # THE "SO WHAT" EXPLANATION (DIRECTLY UNDER CHART)
    st.markdown("---")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader(" Why This Matters (The 'So What')")
        st.info("""
        **1. Margin Recovery:** You are currently paying a 'Passive Premium.' For every CBM shipped LCL, you pay roughly 3x more than FCL. By aggregating at your dock, you reclaim that margin.
        
        **2. Velocity = Working Capital:** LCL shipments sit in 3rd party hubs (CFS) waiting for 'buddies.' Moving to FCL bypasses these hubs, reducing lead time by **~8 days**. Faster transit = less safety stock needed.
        
        **3. Direct Control:** FCLs are sealed at your plant. This eliminates 4-5 touchpoints where damage and theft typically occur in the LCL process.
        """)
    
    with col_b:
        st.subheader("💡 Key Takeaways")
        st.success(f"""
        * **Efficiency:** Your DC can successfully convert **{ (opt_df['FCL_Count'].sum() * 65) / raw_df['Vol_CBM'].sum() * 100 :.1f}%** of current LCL volume into FCL.
        * **Break-Even:** The simulation shows that holding cargo for just **4 days** at your dock yields a **{(total_savings/total_baseline)*100:.1f}%** reduction in total freight spend.
        * **Sustainability:** You are removing the equivalent of **{(total_savings/3800):.0f} full container trips** worth of fragmented handling and trucking.
        """)

with tab2:
    st.subheader("Market Rate Stress Test")
    st.write("How much can FCL rates spike before the strategy fails?")
    rates = np.arange(2500, 8500, 500)
    sens_data = [{'FCL_Rate': r, 'Savings': run_optimization(raw_df, r)['Savings'].sum()} for r in rates]
    sens_df = pd.DataFrame(sens_data)
    fig_sens = px.line(sens_df, x='FCL_Rate', y='Savings', markers=True, title="Savings Sensitivity")
    fig_sens.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even Point")
    st.plotly_chart(fig_sens, use_container_width=True)

with tab3:
    st.header("Executive Methodology")
    st.write("""
    **1. The Engine:** Uses a 'Time-Bucket' heuristic to group 1,000 shipments into weekly windows.
    **2. The W/M Rule:** Applies industry-standard 'Weight or Measure' math (1,000kg = 1 CBM).
    **3. Lead Time Friction:** Assigns an 8-day penalty to LCL for 3rd party hub processing.
    **4. Utilization Factor:** Assumes 15% air-loss in containers to maintain a realistic 'usable cube' of 65 CBM.
    """)
