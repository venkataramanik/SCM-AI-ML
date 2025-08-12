import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("The Bullwhip Effect Simulator")
st.markdown("Adjust the parameters to see how small changes in consumer demand get amplified up the supply chain.")

# --- Sidebar for user input ---
st.sidebar.header("Simulation Parameters")
weeks = st.sidebar.slider("Number of Weeks", 10, 50, 20)
retailer_buffer = st.sidebar.slider("Retailer's Safety Buffer", 0.0, 0.5, 0.1, 0.05)
wholesaler_buffer = st.sidebar.slider("Wholesaler's Safety Buffer", 0.0, 1.0, 0.25, 0.05)
manufacturer_buffer = st.sidebar.slider("Manufacturer's Safety Buffer", 0.0, 1.5, 0.5, 0.05)

if st.sidebar.button("Run Simulation"):
    # --- Simulation Logic ---
    data = {
        'Week': list(range(1, weeks + 1)),
        'Consumer Demand': np.random.randint(8, 12, weeks),
        'Retailer Orders': [0] * weeks,
        'Wholesaler Orders': [0] * weeks,
        'Manufacturer Orders': [0] * weeks,
    }

    df = pd.DataFrame(data)

    for i in range(weeks):
        # Retailer orders based on consumer demand + a safety buffer
        df.loc[i, 'Retailer Orders'] = df.loc[i, 'Consumer Demand'] * (1 + retailer_buffer)
        
        # Wholesaler orders based on retailer's order + a safety buffer
        # (Using i-1 to simulate lag in communication)
        if i > 0:
            df.loc[i, 'Wholesaler Orders'] = df.loc[i-1, 'Retailer Orders'] * (1 + wholesaler_buffer)
        else:
            df.loc[i, 'Wholesaler Orders'] = df.loc[i, 'Retailer Orders'] * (1 + wholesaler_buffer)

        # Manufacturer orders based on wholesaler's order + a safety buffer
        if i > 0:
            df.loc[i, 'Manufacturer Orders'] = df.loc[i-1, 'Wholesaler Orders'] * (1 + manufacturer_buffer)
        else:
             df.loc[i, 'Manufacturer Orders'] = df.loc[i, 'Wholesaler Orders'] * (1 + manufacturer_buffer)


    # --- Visualization ---
    st.subheader("Simulation Results")
    st.markdown("The chart below visualizes the **Bullwhip Effect**, where small changes in consumer demand lead to progressively larger orders up the supply chain.")
    
    fig = px.line(df, x='Week', y=['Consumer Demand', 'Retailer Orders', 'Wholesaler Orders', 'Manufacturer Orders'],
                  title='Order Volume Across Supply Chain Tiers',
                  labels={'value': 'Order Quantity', 'variable': 'Supply Chain Tier'})
    
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Table")
    st.dataframe(df)

st.sidebar.markdown("""
---
**What you're seeing:** The "bullwhip effect" is a supply chain phenomenon where orders to suppliers become more variable than demand from end customers. Each tier (Retailer, Wholesaler, Manufacturer) adds its own safety buffer, amplifying the initial demand signal.
""")
