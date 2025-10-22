import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import streamlit as st 

# Set seed for reproducibility
np.random.seed(42)

# --- 0. Simulate Logistics Data (The fix is here) ---
# FIX: Using np.maximum to ensure Repair_Cost is >= 1, avoiding the .clip() TypeError.
repair_costs = np.random.lognormal(mean=7.0, sigma=1.5, size=1000)
repair_costs = np.maximum(repair_costs, 1)

data = pd.DataFrame({
    'Repair_Cost': repair_costs,
    'Dwell_Time_Min': np.concatenate([np.random.normal(30, 5, 500), np.random.normal(120, 15, 500)]),
    'Stop_ID': [f'STP_{i:04d}' for i in np.random.randint(100, 950, size=1000)],
    'Load_Weight_KG': np.random.uniform(500, 20000, size=1000),
    'Driver_Experience_Yrs': np.random.uniform(1, 20, size=1000),
    'Mileage_Total': np.random.uniform(50000, 500000, size=1000)
})

# Add the dependent feature 
data['Load_Weight_LBS'] = data['Load_Weight_KG'] * 2.20462
data['Load_Weight_LBS'] = data['Load_Weight_LBS'].round(2)


# --- Start Streamlit App ---
st.title("Data Profile & Pre-Flight Audit")
st.markdown("---")

# ==============================================================================
## 1. Distribution Analysis (Skew and Multimodality Check)
# ==============================================================================

st.header("1. Distribution Analysis (Skew & Multimodality)")
st.write(f"**Repair_Cost Skew:** {data['Repair_Cost'].skew():.2f}")

fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(data['Dwell_Time_Min'], kde=True, bins=30, ax=ax)
ax.set_title('Dwell_Time_Min Distribution (Multimodality)')
st.pyplot(fig)
st.markdown("Observation: Dwell_Time_Min clearly shows two distinct peaks (bimodal), suggesting two different types of stops are being measured together.")

# ==============================================================================
## 2. Cardinality & Uniqueness Analysis
# ==============================================================================

st.header("2. Cardinality & Uniqueness Analysis")
unique_stop_count = data['Stop_ID'].nunique()
total_records = len(data)
duplicate_rate = (total_records - unique_stop_count) / total_records * 100

st.write(f"Total Stop_IDs: {total_records}")
st.write(f"Unique Stop_IDs: {unique_stop_count}")
st.write(f"**Duplicate Rate (of Stop_ID records): {duplicate_rate:.2f}%**")

# ==============================================================================
## 3. Correlation/Redundancy Check
# ==============================================================================

st.header("3. Correlation/Redundancy Check")
correlation = data['Load_Weight_KG'].corr(data['Load_Weight_LBS'])
st.write(f"Correlation between Load_Weight_KG and Load_Weight_LBS: {correlation:.4f}")
if correlation > 0.999:
    st.markdown("Observation: Near-perfect correlation! Load_Weight_LBS is redundant (a direct conversion) and should be dropped.")

# ==============================================================================
## 4. Outlier & Anomaly Detection (Using Z-Score)
# ==============================================================================

st.header("4. Outlier & Anomaly Detection")
data['Repair_Cost_Z'] = np.abs(stats.zscore(data['Repair_Cost']))
outliers = data[data['Repair_Cost_Z'] > 3]

st.write(f"Total Outliers in Repair_Cost (Z > 3): {len(outliers)}")
if len(outliers) > 0:
    st.write("Sample Outlier Repair Costs:", outliers['Repair_Cost'].head().values.round(2))
    st.markdown("Action: Investigate these extreme costs. If they are sensor/system errors, they must be capped or removed.")

# ==============================================================================
## 5. Feature Engineering Pre-Flight
# ==============================================================================

st.header("5. Feature Engineering Pre-Flight (Data Conditioning)")

### a) Log Transformation (Repair_Cost)
data['Repair_Cost_Log'] = np.log(data['Repair_Cost'])
st.write(f"Log Transformation Applied. New Skew: {data['Repair_Cost_Log'].skew():.2f}")

### b) Binning (Driver_Experience_Yrs)
bins = [0, 2, 10, data['Driver_Experience_Yrs'].max() + 1]
labels = ['Entry (0-2 Yrs)', 'Mid (3-10 Yrs)', 'Senior (10+ Yrs)']
data['Driver_Tier'] = pd.cut(data['Driver_Experience_Yrs'], bins=bins, labels=labels, right=False)
st.write("\nDriver Experience Binning:")
st.dataframe(data['Driver_Tier'].value_counts())

### c) Scaling (Mileage_Total)
scaler = MinMaxScaler()
data['Mileage_Scaled'] = scaler.fit_transform(data[['Mileage_Total']])
st.write(f"\nMileage Scaled (Min-Max) to: {data['Mileage_Scaled'].min():.2f} to {data['Mileage_Scaled'].max():.2f}")

# --- Final Output ---
st.header("Final Transformed Data Sample")
st.dataframe(data[['Repair_Cost', 'Repair_Cost_Log', 
                   'Driver_Experience_Yrs', 'Driver_Tier', 
                   'Mileage_Total', 'Mileage_Scaled']].head())
