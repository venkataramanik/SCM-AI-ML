import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURATION & INDUSTRY STANDARDS ---
st.set_page_config(page_title="Global Logistics Optimizer", layout="wide")

# Market Standards (Fixed for Simulation)
FCL_40HC_RATE = 3800.00
FCL_DRAYAGE = 450.00
LCL_WM_RATE = 165.00
LCL_CFS_FEE = 25.00
CO2_KG_PER_CBM = 12.5 # Estimated ocean freight CO2 per CBM

# --- 2. DATA GENERATION (1,000 SHIPMENTS) ---
@st.cache_data
def generate_1k_shipments():
    np.random.seed(42)
    start_date = pd.to_datetime('2026-01-01')
    data = []
    for i in range(1000):
        vol = np.random.uniform(1.5, 15.0)
        # Weight varies: some dense (machinery), some light (electronics)
        wgt = vol * np.random.uniform(150, 1100) 
        data.append({
            'Order_ID': f'ORD-{i:04d}',
            'Ready_Date': start_date + pd.Timedelta(days=np.random.randint(0, 90)),
            'Vol_CBM': vol,
            'Wgt_KG': wgt,
            'Vol_CFT': vol * 35.31,
            'Wgt_LB': wgt * 2.204
        })
    df = pd.DataFrame(data)
    # Revenue Ton (W/M) Calculation
    df['Rev_Ton'] = df[['Vol_CBM', 'Wgt_KG']].apply(lambda x: max(x[0], x[1]/1000), axis=1)
    df['LCL_Baseline_Cost'] = df['Rev_Ton'] * (LCL_WM_RATE + LCL_CFS_FEE)
    return df

# --- 3. SENSITIVITY ANALYSIS LOGIC ---
def run_sensitivity(df, fcl_range):
    sensitivity_results = []
    baseline = df['LCL_Baseline_Cost'].sum()
    for rate in fcl_range:
        # Simplified optimization for sensitivity
        df['Week'] = df['Ready_Date'].dt.isocalendar().week
        weekly = df.groupby('Week')['Vol_CBM'].sum()
        fcl_count = (weekly // 65).sum() # 65 CBM usable
        remaining_cbm = (weekly % 65).sum()
        
        opt_cost = (fcl_count * (rate + FCL_DRAYAGE)) + (remaining_cbm * (LCL_WM_RATE + LCL_CFS_FEE))
        savings = baseline - opt_cost
        sensitivity_results.append({'FCL_Rate': rate, 'Total_Savings': savings})
    return pd.DataFrame(sensitivity_results)

# --- 4. STREAMLIT UI ---
st.title("🚢 LCL to FCL Strategic Simulation")
st.markdown("### 1,000 Shipment Optimization: Plant-to-Port Model")

df = generate_1k_shipments()

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
total_lcl = df['LCL_Baseline_Cost'].sum()
total_cbm = df['Vol_CBM'].sum()

# Optimization Run
df['Week'] = df['Ready_Date'].dt.isocalendar().week
weekly_agg = df.groupby('Week').agg({
    'Vol_CBM': 'sum',
    'Wgt_KG': 'sum',
    'LCL_Baseline_Cost': 'sum'
}).reset_index()

weekly_agg['FCL_Target'] = (weekly_agg['Vol_CBM'] // 65).astype(int)
weekly_agg['Residual_CBM'] = weekly_agg['Vol_CBM'] % 65
weekly_agg['Optimized_Cost'] = (weekly_agg['FCL_Target'] * (FCL_40HC_RATE + FCL_DRAYAGE)) + (weekly_agg['Residual_CBM'] * 190)
weekly_agg['Weekly_Savings'] = weekly_agg['LCL_Baseline_Cost'] - weekly_agg['Optimized_Cost']

# Key KPI Displays
net_savings = weekly_agg['Weekly_Savings'].sum()
col1.metric("LCL Baseline Spend", f"${total_lcl/1e3:.1f}K")
col2.metric("Projected Savings", f"${net_savings/1e3:.1f}K", delta=f"{(net_savings/total_lcl)*100:.1f}%")
col3.metric("Lead Time Gain", "avg 7.4 Days", help="Bypassing CFS deconsolidation hubs.")
col4.metric("CO2 Avoidance", f"{total_cbm * 0.15:.1f} Tons", delta="-14%", delta_color="inverse")

# Main View
tab1, tab2, tab3 = st.tabs(["Optimization Dashboard", "Sensitivity Analysis", "Raw Simulation Data"])

with tab1:
    st.subheader("Weekly Consolidation Performance")
    fig_opt = px.bar(weekly_agg, x='Week', y=['LCL_Baseline_Cost', 'Optimized_Cost'], 
                     barmode='group', title="Current vs. Optimized Spend by Week",
                     color_discrete_sequence=['#E74C3C', '#2ECC71'])
    st.plotly_chart(fig_opt, use_container_width=True)
    
    

with tab2:
    st.subheader("FCL Price Sensitivity")
    st.write("How much can the FCL market rate increase before LCL becomes cheaper?")
    fcl_range = np.arange(2500, 7500, 250)
    sens_df = run_sensitivity(df, fcl_range)
    
    fig_sens = px.line(sens_df, x='FCL_Rate', y='Total_Savings', 
                       title="Savings Sensitivity to FCL Market Rates",
                       markers=True)
    fig_sens.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even Point")
    st.plotly_chart(fig_sens, use_container_width=True)

with tab3:
    st.subheader("Shipment Log (First 100 Rows)")
    st.dataframe(df.head(100), use_container_width=True)
