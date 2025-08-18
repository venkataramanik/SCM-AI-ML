import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# from scipy.stats import beta, binom, norm, triangular  # ❌ Not needed; remove to avoid import errors

# --- Page Configuration ---
st.set_page_config(
    page_title="Supply Chain Risk Simulator",
    layout="wide",
    page_icon="🔗"
)

# --- Explanation Section ---
st.title("Inventory Management Under Uncertainty 📦")
st.markdown("""
This application demonstrates how to solve a supply chain problem using **Monte Carlo Simulation** and **Bayesian Inference**.
We'll find the optimal inventory reorder point by modeling uncertain demand and supplier lead times.

### The Problem
A company needs to manage the inventory of a product to minimize costs.
- **Holding Cost:** Cost to store an unsold item ($5/unit/day).
- **Stockout Cost:** Cost of a lost sale when out of stock ($50/unit).

The challenge is that **daily demand** and **supplier lead time** are uncertain.

### The Methodology

1.  **Initial Monte Carlo Simulation:** We'll start with our initial beliefs about demand and lead time to simulate thousands of scenarios and estimate the total costs.
2.  **Bayesian Update:** After observing some real-world data (e.g., actual lead times), we'll use Bayesian inference to update our beliefs.
3.  **Posterior Monte Carlo Simulation:** We'll run the simulation again with our more informed beliefs to get a more accurate estimate of the costs.
""")

st.markdown("---")

# --- Initial Assumptions ---
st.header("Initial Assumptions & Parameters")

with st.expander("Initial Model Parameters"):
    st.markdown("""
    - **Daily Demand (Prior):** Normal Distribution (Mean: 100 units, Std Dev: 15 units)
    - **Lead Time (Prior):** Triangular Distribution (Min: 5 days, Mode: 6 days, Max: 7 days)
    - **Reorder Point:** We will test a reorder point of 800 units.
    - **Reorder Quantity:** A fixed 1000 units.
    - **Holding Cost:** $5 per unit per day.
    - **Stockout Cost:** $50 per unit.
    """)

# --- Simulation Logic ---
@st.cache_data
def run_simulation(num_sims, demand_mean, demand_std, lead_time_params):
    """
    Runs the Monte Carlo simulation with given parameters.
    """
    total_costs = []
    holding_costs = []
    stockout_costs = []

    # Unpack lead time parameters
    lt_min, lt_mode, lt_max = lead_time_params

    for _ in range(num_sims):
        # 1. Simulate Lead Time (from triangular distribution)
        lead_time = int(np.round(np.random.triangular(lt_min, lt_mode, lt_max)))
        
        # 2. Simulate Total Demand during lead time
        total_demand = float(np.sum(np.random.normal(demand_mean, demand_std, lead_time)))
        
        # 3. Calculate Inventory Position at arrival
        reorder_point = 800
        reorder_quantity = 1000
        inventory_at_arrival = reorder_point + reorder_quantity - total_demand

        # 4. Calculate Costs
        if inventory_at_arrival > 0:
            h_cost = 5 * inventory_at_arrival
            s_cost = 0
        else:
            h_cost = 0
            s_cost = 50 * abs(inventory_at_arrival)
        
        total_costs.append(h_cost + s_cost)
        holding_costs.append(h_cost)
        stockout_costs.append(s_cost)

    return pd.DataFrame({
        'Total Cost': total_costs,
        'Holding Cost': holding_costs,
        'Stockout Cost': stockout_costs
    })

# --- Monte Carlo Simulation 1: Initial Beliefs ---
st.header("Step 1: Initial Monte Carlo Simulation")
st.markdown("Running 10,000 simulations based on our initial beliefs...")

initial_results = run_simulation(10000, 100, 15, (5, 6, 7))

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Average Total Cost (Initial)", value=f"${initial_results['Total Cost'].mean():,.2f}")
    st.metric(label="Probability of Stockout (Initial)", value=f"{(initial_results['Stockout Cost'] > 0).mean():.2%}")

with col2:
    fig_hist_initial = px.histogram(
        initial_results,
        x="Total Cost",
        nbins=50,
        title="Distribution of Initial Total Costs"
    )
    st.plotly_chart(fig_hist_initial, use_container_width=True)

st.markdown("---")

# --- Bayesian Update Section ---
st.header("Step 2: Bayesian Update with New Data 📈")
st.markdown("""
A new report arrives with real-world data. We observed **20 recent shipments** and the **average lead time was 6.5 days**.
We'll use Bayesian inference to update our belief about the lead time distribution.

- **Prior:** Our initial belief was a triangular distribution centered at 6 days.
- **Likelihood:** The new data (sample of 20 with a mean of 6.5 days).
- **Posterior:** Our new, updated belief. The posterior's mode will shift closer to 6.5 days, reflecting the new information.
""")

# Simplified Bayesian update for demonstration
# We will use a rule of thumb: The new mode is a weighted average
initial_mode = 6
new_data_mean = 6.5
new_data_count = 20
prior_count = 10  # A notional prior 'data count' to weight our initial belief
posterior_mode = (initial_mode * prior_count + new_data_mean * new_data_count) / (prior_count + new_data_count)
posterior_lead_time_params = (5, posterior_mode, 7)

st.info(f"The new, updated mode for the lead time distribution is **{posterior_mode:.2f} days**.")

st.markdown("---")

# --- Monte Carlo Simulation 2: Posterior Beliefs ---
st.header("Step 3: Posterior Monte Carlo Simulation")
st.markdown("Running 10,000 simulations using our **updated** belief about the lead time.")

posterior_results = run_simulation(10000, 100, 15, posterior_lead_time_params)

col3, col4 = st.columns(2)

with col3:
    st.metric(
        label="Average Total Cost (Updated)",
        value=f"${posterior_results['Total Cost'].mean():,.2f}",
        delta=f"${posterior_results['Total Cost'].mean() - initial_results['Total Cost'].mean():,.2f}"
    )
    st.metric(
        label="Pr
