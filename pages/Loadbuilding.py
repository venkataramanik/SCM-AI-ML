import streamlit as st
import pandas as pd
from pulp import *
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="WLO Pro Optimizer", layout="wide")

st.title("🏭 Warehouse Labor Optimization Engine")
st.markdown("""
This tool uses **Mixed-Integer Linear Programming** to determine the minimum number of 8-hour shifts 
required to cover fluctuating hourly demand.
""")

# --- SIDEBAR: PARAMETERS ---
with st.sidebar:
    st.header("Cost & Efficiency Settings")
    hourly_wage = st.number_input("Standard Hourly Wage ($)", value=22)
    uph = st.slider("Units Per Hour (UPH) per Worker", 20, 150, 60)
    shift_length = 8  # Fixed shift length for this demo
    
    st.header("Demand Input")
    # Mock demand for a 16-hour warehouse operation
    demand_data = [400, 500, 700, 800, 600, 400, 300, 500, 800, 1000, 900, 600, 400, 300, 200, 150]
    hours = [f"{h}:00" for h in range(6, 22)]
    df_demand = pd.DataFrame({"Hour": hours, "Units": demand_data})

# --- OPTIMIZATION LOGIC ---
def solve_labor_optimization(demand, uph, wage, s_length):
    prob = LpProblem("Warehouse_Staffing", LpMinimize)
    
    # Decision Variables: Number of workers STARTING their shift at hour 'i'
    # We only allow starts that can finish within our 16-hour window
    start_indices = range(len(demand) - s_length + 1)
    x = LpVariable.dicts("ShiftStart", start_indices, lowBound=0, cat='Integer')
    
    # Objective: Minimize total wages paid
    # Each shift started costs (Wage * Shift Length)
    prob += lpSum([x[i] * (wage * s_length) for i in start_indices])
    
    # Constraints: For every hour 't', workers currently on shift >= demand/uph
    for t in range(len(demand)):
        # A worker is active at hour 't' if they started between (t - s_length + 1) and t
        active_workers = lpSum([x[i] for i in start_indices if i <= t < i + s_length])
        prob += active_workers * uph >= demand[t], f"Demand_Constraint_Hour_{t}"
        
    prob.solve(PULP_CBC_CMD(msg=0))
    
    # Extract results
    starts = [int(x[i].varValue) for i in start_indices]
    # Calculate total active workers per hour for plotting
    active_per_hour = []
    for t in range(len(demand)):
        active = sum([starts[i] for i in start_indices if i <= t < i + s_length])
        active_per_hour.append(active)
        
    return starts, active_per_hour, value(prob.objective)

# --- EXECUTION ---
if st.button("Run Optimization Model"):
    starts, active_schedule, total_cost = solve_labor_optimization(demand_data, uph, hourly_wage, shift_length)
    
    # Prepare Data for UI
    df_demand["Scheduled_Staff"] = active_schedule
    df_demand["Capacity_Units"] = [a * uph for a in active_schedule]
    df_demand["Utilization"] = (df_demand["Units"] / df_demand["Capacity_Units"] * 100).round(1)

    # --- KPI DASHBOARD ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Labor Cost", f"${total_cost:,.2f}")
    c2.metric("Total Shifts Scheduled", sum(starts))
    c3.metric("Avg Utilization", f"{df_demand['Utilization'].mean():.1f}%")

    # --- VISUALIZATION ---
    st.subheader("Demand vs. Capacity Coverage")
    fig = px.bar(df_demand, x="Hour", y=["Units", "Capacity_Units"], 
                 barmode="group", title="Units Demanded vs. Units Capacity Provided",
                 labels={"value": "Units", "variable": "Type"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Shift Start Schedule")
    # Show when managers should actually have people clock in
    start_df = pd.DataFrame({"Hour": [hours[i] for i in range(len(starts))], "New_Clock_Ins": starts})
    st.table(start_df[start_df["New_Clock_Ins"] > 0])

else:
    st.info("Click the button above to calculate the optimal staffing plan.")

# --- EDUCATIONAL FOOTER ---
with st.expander("How this works (The Math)"):
    st.write("""
    1. **Decision Variable:** $x_i$ is the number of workers starting a shift at hour $i$.
    2. **Objective:** Minimize $\sum x_i \times (\text{Wage} \times 8)$.
    3. **The 'Sliding Window':** The model ensures that for any hour $t$, the sum of all $x$ variables that started in the last 8 hours is sufficient to meet the demand.
    """)
