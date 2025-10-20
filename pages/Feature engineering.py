import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(layout="wide")
st.title("Data Transformation for Predictive Logistics: Feature Engineering Demo")

# --- 1. Synthesize Realistic Logistics Data ---
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
        p=p_normalized
    ),
    # Weather: Low Cardinality
    'weather_condition': np.random.choice(['Clear', 'Rain', 'Snow', 'Heavy Fog'], N, p=[0.7, 0.15, 0.1, 0.05]),
    # Time Data (Hour of Day)
    'trip_start_hour': np.random.randint(0, 24, N),
})

# --- Create a master copy for all transformations to ensure stability ---
df = logistics_df.copy()

st.header("Initial Data Overview: Raw Inputs")
st.markdown("""
This table shows the raw data received directly from the fleet's GPS and sensor logs. This data requires significant mathematical **transformation** (Feature Engineering) before it can be used by a predictive model to accurately calculate delivery times.
""")
st.dataframe(df.head(), use_container_width=True)

st.markdown("---")

# --- 2. Feature Engineering Demonstrations ---

st.header("Core Data Transformation Techniques (Feature Engineering)")

# A. Cyclic Features (trip_start_hour)
st.subheader("A. Cyclic Features: Solving the Time-of-Day Problem")
st.markdown("""
**Business Context:** Rush hour and overnight efficiency are cyclical. Standard numbering (0-23) is misleading because the end point (23) is conceptually next to the start point (0).

**Transformation:** The single hour number is converted into **two new features (Sine and Cosine)**, mapping the time onto a circle. This correctly models the continuity of time, allowing the ETA model to anticipate cyclical congestion accurately.
""")
HOURS_IN_CYCLE = 24
df['hour_sin'] = np.sin(2 * np.pi * df['trip_start_hour'] / HOURS_IN_CYCLE)
df['hour_cos'] = np.cos(2 * np.pi * df['trip_start_hour'] / HOURS_IN_CYCLE)
st.dataframe(df[['trip_start_hour', 'hour_sin', 'hour_cos']].head(), use_container_width=True)
st.write("---")


# B. Interaction Features (distance_miles * weather_condition)
st.subheader("B. Interaction Features: Scaling Operational Risk")
st.markdown("""
**Business Context:** The penalty for a delay is not fixed; it **amplifies** under certain conditions. Bad weather causes a minimal delay on a short trip but a massive delay on a long trip.

**Transformation:** A new feature is created by **multiplying** the distance by the weather's severity score. This explicitly teaches the model to apply a much larger penalty when both high distance and bad weather are present, improving **conditional risk assessment**.
""")
severity_map = {'Clear': 1, 'Rain': 2, 'Snow': 3, 'Heavy Fog': 4}
df['weather_severity'] = df['weather_condition'].map(severity_map)
df['risk_interaction'] = df['distance_miles'] * df['weather_severity']
st.dataframe(df[['distance_miles', 'weather_condition', 'risk_interaction']].head(), use_container_width=True)
st.write("---")


# C. Binning / Discretization (distance_miles)
st.subheader("C. Binning: Creating Clear Operational Categories")
st.markdown("""
**Business Context:** Operations are often categorized (e.g., local vs. long haul). Modeling the delay based on the **operational category** is more robust than relying on the pattern for every single mile.

**Transformation:** The continuous `distance_miles` is converted into discrete, defined categories (`Local_Haul`, `Short_Haul`, etc.). This reduces noise and helps the predictive model find stable patterns for each haul type.
""")
bins = [0, 300, 800, 1200, np.inf]
labels = ['Local_Haul', 'Short_Haul', 'Medium_Haul', 'Long_Haul']
df['haul_category'] = pd.cut(
    df['distance_miles'], 
    bins=bins, 
    labels=labels, 
    right=False
)
st.dataframe(df[['distance_miles', 'haul_category']].head(), use_container_width=True)
st.write("---")


# D. One-Hot Encoding (weather_condition)
st.subheader("D. One-Hot Encoding (OHE): Quantifying Qualitative Data")
st.markdown("""
**Business Context:** The predictive model needs numerical inputs; it cannot process the text labels 'Snow'.

**Transformation:** The text labels are converted into **separate binary (0 or 1) columns**. This assigns a clear, quantifiable dimension to each weather type, allowing the model to learn the specific delay risk associated with 'Snow'.
""")
# --- STABLE FIX: Create a temporary display DataFrame for OHE ---
df_ohe_display = pd.get_dummies(df[['weather_condition']], prefix='weather')
# Concatenate the original weather column with the new OHE columns just for display
df_ohe_final = pd.concat([df[['weather_condition']].reset_index(drop=True), df_ohe_display.reset_index(drop=True)], axis=1)

# Note: The main DataFrame 'df' is NOT altered here to prevent column duplication on rerun.
st.dataframe(df_ohe_final.head(), use_container_width=True) 
st.write("---")


# E. Standardization (stops_count and distance_miles)
st.subheader("E. Standardization: Ensuring Fair Feature Influence")
st.markdown("""
**Business Context:** Raw features have wildly different scales: `Distance` (up to 1,500) and `Stops` (up to 7). Unscaled, the model might mistakenly assume Distance is more important just because its raw numerical value is larger.

**Transformation:** Both features are scaled to have a **mean of 0 and a standard deviation of 1**. This ensures that the predictive power of **Stops** has a fair and equal influence on the ETA calculation as **Distance**, regardless of their raw numerical values.
""")
scaler = StandardScaler()
cols_to_scale = ['distance_miles', 'stops_count']
# Ensure we operate on the original columns and write to new scaled columns
df[['distance_scaled', 'stops_scaled']] = scaler.fit_transform(df[cols_to_scale]) 
st.dataframe(df[['distance_miles', 'distance_scaled', 'stops_count', 'stops_scaled']].head(), use_container_width=True)
st.write("---")


# F. Frequency Encoding (delivery_region - High Cardinality)
st.subheader("F. Frequency Encoding: Handling Rare Operational Segments")
st.markdown("""
**Business Context:** When a feature has many rare values (high **cardinality**), creating an OHE column for every rare region leads to an unstable model.

**Transformation:** The region name is replaced with the **count of how often that region appears**. This helps the model implicitly learn that low-frequency regions carry a higher risk of ETA variance due to non-standard operations or remote locations, avoiding the problem of rare categories.
""")
region_counts = df['delivery_region'].value_counts().to_dict()
df['region_frequency'] = df['delivery_region'].map(region_counts)
st.dataframe(df[['delivery_region', 'region_frequency']].sort_values(by='region_frequency').head(7), use_container_width=True)
st.write("---")
