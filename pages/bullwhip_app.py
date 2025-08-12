import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("The Bullwhip Effect Simulator 📊")
st.markdown("This app shows how small changes in consumer demand get dramatically amplified as orders move up the supply chain. Adjust the sliders on the left to see the effect!")

st.sidebar.header("Simulation Parameters")
weeks = st.sidebar.slider("Number of Weeks", 10, 50, 20)
retailer_buffer = st.sidebar.slider("Retailer's 'Fear Factor' (Safety Buffer)", 0.0, 0.5, 0.1, 0.05)
wholesaler_buffer = st.sidebar.slider("Wholesaler's 'Fear Factor'", 0.0, 1.0, 0.25, 0.05)
manufacturer_buffer = st.sidebar.slider("Manufacturer's 'Fear Factor'", 0.0, 1.5, 0.5, 0.05)

if st.sidebar.button("Run Simulation"):
    
    data = {
        'Week': list(range(1, weeks + 1)),
        'Consumer Demand': np.random.randint(8, 12, weeks),
        'Retailer Orders': [0.0] * weeks,
        'Wholesaler Orders': [0.0] * weeks,
        'Manufacturer Orders': [0.0] * weeks,
    }

    df = pd.DataFrame(data)

    for i in range(weeks):
        # Retailer orders based on demand + a buffer
        df.loc[i, 'Retailer Orders'] = df.loc[i, 'Consumer Demand'] * (1 + retailer_buffer)
        
        # Wholesaler's logic is based on the previous week's retailer order
        if i > 0:
            df.loc[i, 'Wholesaler Orders'] = df.loc[i-1, 'Retailer Orders'] * (1 + wholesaler_buffer)
        else:
            df.loc[i, 'Wholesaler Orders'] = df.loc[i, 'Retailer Orders'] * (1 + wholesaler_buffer)

        # Manufacturer's logic is based on the previous week's wholesaler order
        if i > 0:
            df.loc[i, 'Manufacturer Orders'] = df.loc[i-1, 'Wholesaler Orders'] * (1 + manufacturer_buffer)
        else:
             df.loc[i, 'Manufacturer Orders'] = df.loc[i, 'Wholesaler Orders'] * (1 + manufacturer_buffer)

    st.subheader("Order Volume Across Supply Chain Tiers")
    st.markdown("Notice how the small variations in **Consumer Demand** become huge swings in the **Manufacturer's Orders**!")
    
    fig = px.line(df, x='Week', y=['Consumer Demand', 'Retailer Orders', 'Wholesaler Orders', 'Manufacturer Orders'],
                  title='Order Amplification',
                  labels={'value': 'Order Quantity', 'variable': 'Supply Chain Tier'},
                  color_discrete_map={
                      "Consumer Demand": "black",
                      "Retailer Orders": "blue",
                      "Wholesaler Orders": "green",
                      "Manufacturer Orders": "red"
                  })
    
    fig.update_layout(height=500, width=1000)
    st.plotly_chart(fig, use_container_width=True)

    st.sidebar.markdown("""
---
**The 'Why' behind the whip:**
The bullwhip effect happens because each company in the chain makes its own decisions based on a limited view of the market. They often add a safety buffer to their orders just in case, and this small extra amount gets amplified at every single step, causing massive inefficiency for the manufacturer.
""")
