import streamlit as st
import pandas as pd
from pulp import *

st.set_page_config(page_title="Warehouse Labor Optimizer", layout="wide")

st.title("📦 Warehouse Labor Optimization Demo")
st.markdown("Optimize shift assignments based on hourly order volume.")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
hourly_wage = st.sidebar.number_input("Hourly Wage ($)", value=25)
uph = st.sidebar.slider("Units Per Hour (Efficiency)", 10, 100, 50)

# Mock Demand Data
data = {
    "Hour": ["08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00"],
    "Demand_Units": [450, 600, 800, 300, 200, 700, 900, 400]
}
df = pd.DataFrame(data)

st.subheader("Hourly Demand Forecast")
st.line_chart(df.set_index("Hour"))

# --- Optimization Engine ---
if st.button("Optimize Labor Allocation"):
    # Define the Problem
    prob = LpProblem("Labor_Staffing", LpMinimize)
    
    # Decision Variables: Number of workers per hour
    workers = LpVariable.dicts("Staff", df.index, lowBound=0, cat='Integer')
    
    # Objective Function: Minimize Cost
    prob += lpSum([workers[i] * hourly_wage for i in df.index])
    
    # Constraints: Staff capacity must meet hourly demand
    for i in df.index:
        prob += workers[i] * uph >= df.loc[i, "Demand_Units"]

    prob.solve()
    
    # Results Processing
    df["Required_Staff"] = [int(workers[i].varValue) for i in df.index]
    df["Total_Cost"] = df["Required_Staff"] * hourly_wage
    
    # --- Dashboard Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Labor Cost", f"${df['Total_Cost'].sum():,}")
    col2.metric("Total Workers Needed", f"{df['Required_Staff'].sum()}")
    col3.metric("Avg Utilization", "92%") # Simplified for demo

    st.subheader("Optimized Staffing Plan")
    st.bar_chart(df.set_index("Hour")["Required_Staff"])
    st.table(df)
