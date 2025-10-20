import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(layout="wide")
st.title("Data Transformation for Predictive Logistics: Feature Engineering ")

# --- 1. Synthesize Realistic Logistics Data (FIXED FOR PRECISION) ---
np.random.seed(42)
N = 1000

# Define the raw probabilities
raw_probs = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 
                      0.02, 0.02, 0.01, 0.01, 0.01, 
                      0.01, 0.01, 0.01, 0.01, 0.01])

# Corrective Step: Normalize the array to ensure the sum is exactly 1.0
p_normalized = raw_probs / raw_probs.sum()


# Generating data that represents common logistics scenarios
logistics_df = pd.DataFrame({
    # Trip length (miles)
    'distance_miles': np.random.uniform(50, 1500, N),
    # Number of stops/deliveries
    'stops_count': np.random.randint(0, 8, N),
    # Delivery region: High Cardinality (15 total regions for Frequency Encoding demo)
    'delivery_region': np.random.choice(
        ['East', 'West', 'Central', 'South', 'North'] + [f'Other_{i}' for i in range(10)], N, 
        # Pass the normalized array here
        p=p_normalized
    ),
    # Weather: Low Cardinality
    'weather_condition': np.random.choice(['Clear', 'Rain', 'Snow', 'Heavy Fog'], N, p=[0.7, 0.15, 0.1, 0.05]),
    # Time Data (Hour of Day)
    'trip_start_hour': np.random.randint(0, 24, N),
})

# --- The rest of the script continues below, using the logistics_df ---

st.header("Initial Data Overview: Raw Inputs")
st.markdown("""
This table shows the raw data received directly from our fleet's GPS and sensor logs. While simple, this data needs significant mathematical **transformation** (Feature Engineering) before it can be used by a predictive model to accurately calculate delivery times.
""")
st.dataframe(logistics_df.head(), use_container_width=True)

st.markdown("---")

# --- 2. Feature Engineering Demonstrations ---

st.header("Core Data Transformation Techniques (Feature Engineering)")

# A. Cyclic Features (trip_start_hour)
st.subheader("A. Cyclic Features: Solving the Time-of-Day Problem")
st.markdown("""
**Business Context:** Rush hour and overnight efficiency are cyclical. Standard numbering (0, 1, ..., 23) treats 11 PM as far from 1 AM, which is misleading for traffic.

**Transformation:** I convert the single hour number into **two new features (Sine and Cosine)**, mapping the time onto a circle. This correctly models the continuity of time, allowing our ETA model to anticipate cyclical congestion accurately.
""")
HOURS_IN_CYCLE = 24
logistics_df['hour_sin'] = np.sin(2 * np.pi * logistics_df['trip_start_hour'] / HOURS_IN_CYCLE)
logistics_df['hour_cos'] = np.cos(2 * np.pi * logistics_df['trip_start_hour'] / HOURS_IN_CYCLE)
st.dataframe(logistics_df[['trip_start_hour', 'hour_sin', 'hour_cos']].head(), use_container_width=True)
st.write("---")


# B. Interaction Features (distance_miles * weather_condition)
st.subheader("B. Interaction Features: Scaling Operational Risk")
st.markdown("""
**Business Context:** The penalty for a delay isn't fixed; it **amplifies** under certain conditions. Bad weather causes a minimal delay on a short trip but a massive delay on a long trip.

**Transformation:** I create a new feature by **multiplying** the distance by the weather's severity score. This explicitly teaches the model to apply a much larger penalty when both high distance and bad weather are present, improving **conditional risk assessment**.
""")
severity_map = {'Clear': 1, 'Rain': 2, 'Snow': 3, 'Heavy Fog': 4}
logistics_df['weather_severity'] = logistics_df['weather_condition'].map(severity_map)
logistics_df['risk_interaction'] = logistics_df['distance_miles'] * logistics_df['weather_severity']
st.dataframe(logistics_df[['distance_miles', 'weather_condition', 'risk_interaction']].head(), use_container_width=True)
st.write("---")


# C. Binning / Discretization (distance_miles)
st.subheader("C. Binning: Creating Clear Operational Categories")
st.markdown("""
**Business Context:** Operations are often categorized (e.g., local vs. long haul). Instead of modeling the unique pattern for every single mile, it's more robust to model the delay based on the **operational category**.

**Transformation:** I convert the continuous `distance_miles` into discrete, defined categories (`Local_Haul`, `Short_Haul`, etc.). This reduces noise and helps the predictive model find stable patterns for each haul type.
""")
bins = [0, 300, 800, 1200, np.inf]
labels = ['Local_Haul', 'Short_Haul', 'Medium_Haul', 'Long_Haul']
logistics_df['haul_category'] = pd.cut(
    logistics_df['distance_miles'], 
    bins=bins, 
    labels=labels, 
    right=False
)
st.dataframe(logistics_df[['distance_miles', 'haul_category']].head(), use_container_width=True)
st.write("---")


# D. One-Hot Encoding (weather_condition)
st.subheader("D. One-Hot Encoding (OHE): Quantifying Qualitative Data")
st.markdown("""
**Business Context:** The model needs numbers; it cannot process the text 'Snow'.

**Transformation:** I convert the text labels into **separate binary (0 or 1) columns**. This assigns a clear, quantifiable dimension to each weather type, allowing the model to learn the specific delay risk (the coefficient) associated with 'Snow'.
""")
weather_dummies = pd.get_dummies(logistics_df['weather_condition'], prefix='weather')
logistics_df = pd.concat([logistics_df, weather_dummies], axis=1)
st.dataframe(logistics_df[['weather_condition'] + [col for col in logistics_df.columns if 'weather_' in col]].head(), use_container_width=True)
st.write("---")


# E. Standardization (stops_count and distance_miles)
st.subheader("E. Standardization: Ensuring Fair Feature Influence")
st.markdown("""
**Business Context:** Raw features have wildly different scales: `Distance` (up to 1,500) and `Stops` (up to 7). The model might mistakenly assume Distance is 1,000 times more important just because its number is bigger.

**Transformation:** I scale both features to have a **mean of 0 and a standard deviation of 1**. This ensures that the predictive power of **Stops** has a fair and equal influence on the ETA calculation as **Distance**, regardless of their raw numerical values.
""")
scaler = StandardScaler()
cols_to_scale = ['distance_miles', 'stops_count']
logistics_df[['distance_scaled', 'stops_scaled']] = scaler.fit_transform(logistics_df[cols_to_scale])
st.dataframe(logistics_df[['distance_miles', 'distance_scaled', 'stops_count', 'stops_scaled']].head(), use_container_width=True)
st.write("---")


# F. Frequency Encoding (delivery_region - High Cardinality)
st.subheader("F. Frequency Encoding: Handling Rare Operational Segments")
st.markdown("""
**Business Context:** We have many delivery regions, but several are very rare (high **cardinality**). Creating an OHE column for every rare region leads to a chaotic, unstable model.

**Transformation:** I replace the region name with the **count of how often that region appears**. This helps the model implicitly learn that low-frequency regions carry a higher risk of ETA variance due to non-standard operations or remote locations, avoiding the problem of rare categories.
""")
region_counts = logistics_df['delivery_region'].value_counts().to_dict()
logistics_df['region_frequency'] = logistics_df['delivery_region'].map(region_counts)
st.dataframe(logistics_df[['delivery_region', 'region_frequency']].sort_values(by='region_frequency').head(7), use_container_width=True)
st.write("---")
