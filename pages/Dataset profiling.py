import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# Set display options for cleaner output
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

# --- 0. Simulate Logistics Data ---
np.random.seed(42)
data = pd.DataFrame({
    # Positively skewed: Most costs are low, some are very high (for Log Analysis)
    'Repair_Cost': np.random.lognormal(mean=7.0, sigma=1.5, size=1000).clip(lower=1),
    # Bimodal/Multimodal: Driven by two distinct populations (for Distribution Analysis)
    'Dwell_Time_Min': np.concatenate([np.random.normal(30, 5, 500), np.random.normal(120, 15, 500)]),
    # High Cardinality with potential duplicates (for Cardinality/Uniqueness)
    'Stop_ID': [f'STP_{i:04d}' for i in np.random.randint(100, 950, size=1000)],
    # Perfect correlation/redundancy (for Correlation Check)
    'Load_Weight_KG': np.random.uniform(500, 20000, size=1000),
    'Load_Weight_LBS': lambda x: x['Load_Weight_KG'] * 2.20462,
    # Continuous experience: Years on the job (for Binning)
    'Driver_Experience_Yrs': np.random.uniform(1, 20, size=1000),
    # Feature for scaling (for Scaling)
    'Mileage_Total': np.random.uniform(50000, 500000, size=1000)
})
data['Load_Weight_LBS'] = data['Load_Weight_LBS'].round(2)

print("--- Data Profile Audit Starting ---")
print(f"Dataset Shape: {data.shape}")
print("-" * 50)

# ==============================================================================
## 1. Distribution Analysis (Skew and Multimodality Check)
# ==============================================================================

print("\n## 1. Distribution Analysis (Skew & Multimodality)")
# Check skew for Repair_Cost
print(f"Repair_Cost Skew: {data['Repair_Cost'].skew():.2f} (Highly skewed? >1 or <-1)")
# Visualize Dwell_Time_Min to check for multimodality
plt.figure(figsize=(10, 4))
sns.histplot(data['Dwell_Time_Min'], kde=True, bins=30)
plt.title('Dwell_Time_Min Distribution (Multimodality)')
plt.show()
print("Observation: Dwell_Time_Min clearly shows two distinct peaks (bimodal), suggesting two different types of stops are being measured together.")
print("-" * 50)

# ==============================================================================
## 2. Cardinality & Uniqueness Analysis
# ==============================================================================

print("\n## 2. Cardinality & Uniqueness Analysis")
unique_stop_count = data['Stop_ID'].nunique()
total_records = len(data)
duplicate_rate = (total_records - unique_stop_count) / total_records * 100

print(f"Total Stop_IDs: {total_records}")
print(f"Unique Stop_IDs: {unique_stop_count}")
print(f"Duplicate Rate (of Stop_ID records): {duplicate_rate:.2f}%")
print("Observation: Stop_ID has a significant duplicate rate. Need to confirm if these are intentional shared IDs or unintended data duplication.")
print("-" * 50)

# ==============================================================================
## 3. Correlation/Redundancy Check
# ==============================================================================

print("\n## 3. Correlation/Redundancy Check")
correlation = data['Load_Weight_KG'].corr(data['Load_Weight_LBS'])
print(f"Correlation between Load_Weight_KG and Load_Weight_LBS: {correlation:.4f}")

if correlation > 0.999:
    print("Observation: Near-perfect correlation! Load_Weight_LBS is redundant (a direct conversion) and should be dropped to simplify the model.")
    data.drop(columns=['Load_Weight_LBS'], inplace=True)
print("-" * 50)

# ==============================================================================
## 4. Outlier & Anomaly Detection (Using Z-Score)
# ==============================================================================

print("\n## 4. Outlier & Anomaly Detection")
# Calculate Z-scores for Repair_Cost (assuming anything > 3 standard deviations is an outlier)
data['Repair_Cost_Z'] = np.abs(stats.zscore(data['Repair_Cost']))
outliers = data[data['Repair_Cost_Z'] > 3]

print(f"Total Outliers in Repair_Cost (Z > 3): {len(outliers)}")
print("Sample Outlier Repair Costs:")
print(outliers['Repair_Cost'].head().values)
print("Action: Investigate these extreme costs. If they are sensor/system errors, they must be capped or removed.")
data.drop(columns=['Repair_Cost_Z'], inplace=True) # Clean up temp column
print("-" * 50)

# ==============================================================================
## 5. Feature Engineering Pre-Flight
# ==============================================================================

print("\n## 5. Feature Engineering Pre-Flight (Data Conditioning)")

### a) Log Transformation (Repair_Cost)
data['Repair_Cost_Log'] = np.log(data['Repair_Cost'])
print(f"Log Transformation Applied. New Skew: {data['Repair_Cost_Log'].skew():.2f}")

### b) Binning (Driver_Experience_Yrs)
bins = [0, 2, 10, data['Driver_Experience_Yrs'].max() + 1]
labels = ['Entry (0-2 Yrs)', 'Mid (3-10 Yrs)', 'Senior (10+ Yrs)']
data['Driver_Tier'] = pd.cut(data['Driver_Experience_Yrs'], bins=bins, labels=labels, right=False)
print("\nDriver Experience Binning:")
print(data['Driver_Tier'].value_counts())

### c) Scaling (Mileage_Total)
scaler = MinMaxScaler()
data['Mileage_Scaled'] = scaler.fit_transform(data[['Mileage_Total']])
print(f"\nMileage Scaled (Min-Max) to: {data['Mileage_Scaled'].min():.2f} to {data['Mileage_Scaled'].max():.2f}")
print("-" * 50)

# --- Final Output ---
print("\n--- FINAL TRANSFORMED DATA SAMPLE ---")
print(data[['Repair_Cost', 'Repair_Cost_Log', 
            'Driver_Experience_Yrs', 'Driver_Tier', 
            'Mileage_Total', 'Mileage_Scaled']].head())
