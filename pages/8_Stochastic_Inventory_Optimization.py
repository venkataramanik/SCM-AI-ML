import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Stochastic Inventory Optimization"
)

st.title("Stochastic Inventory Optimization")
st.write("---")

# -- Business Context Section --
st.subheader("Business Context")
st.write("""
In the real world, demand and lead times are rarely certain. Relying on deterministic models can lead to frequent stockouts or excessive inventory. **Stochastic Optimization** helps us build robust policies that perform well despite this uncertainty.
""")

# -- The Concept: Stochastic Optimization --
st.subheader("The Concept: Stochastic Optimization")
st.write("""
This project uses **Monte Carlo simulation** to model an inventory system over many scenarios. Instead of a single value, we use a **probability distribution** for demand. The model then helps find an optimal inventory policy (e.g., safety stock) that minimizes total costs on average.
""")

# -- Tools Used Section --
st.subheader("Tools Used")
st.write("""
- **Pandas** for data management.
- **Numpy** for simulating random demand.
- **Matplotlib** for visualization.
""")

# -- Code and Demonstration --
def run_simulation(safety_stock, simulation_id):
    np.random.seed(simulation_id)
    daily_demand = np.random.poisson(lam=50, size=365)
    
    total_inventory = 1000 + safety_stock
    total_cost = 0
    holding_cost_rate = 0.50
    stockout_cost = 5.00
    
    inventory_levels = []
    
    for demand in daily_demand:
        if total_inventory >= demand:
            total_inventory -= demand
            total_cost += total_inventory * holding_cost_rate
        else:
            stockout_quantity = demand - total_inventory
            total_inventory = 0
            total_cost += stockout_quantity * stockout_cost
            
        inventory_levels.append(total_inventory)
        
    simulation_data = pd.DataFrame({
        'Day': range(1, 366),
        'Daily Demand': daily_demand,
        'End of Day Inventory': inventory_levels
    })
    
    return total_cost, simulation_data

# Use a button to get new random data
if st.button("Run New Simulation"):
    st.session_state['simulation_id'] = np.random.randint(0, 1000000)

if 'simulation_id' not in st.session_state:
    st.session_state['simulation_id'] = 42 # Default seed

st.subheader("Interactive Stochastic Analysis")
st.info("Adjust the safety stock to see its effect on total costs and inventory levels over one year of uncertain demand.")

safety_stock = st.slider("Safety Stock Level", min_value=0, max_value=200, value=50)

total_cost, simulation_data = run_simulation(safety_stock, st.session_state['simulation_id'])

st.metric(label="Total Simulated Cost Over 1 Year", value=f"${total_cost:,.2f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(simulation_data['End of Day Inventory'])
ax.axhline(y=safety_stock, color='r', linestyle='--', label='Safety Stock Level')
ax.set_title("Inventory Level Over 1 Year with Stochastic Demand")
ax.set_xlabel("Day")
ax.set_ylabel("Inventory Level")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("Raw Simulation Data")
st.write("This table shows the daily demand and resulting inventory level for the simulated year.")
st.dataframe(simulation_data)
