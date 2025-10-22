import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer

# Set seed for reproducibility
np.random.seed(42)

# --- 1. Simulate Rich Logistics Data with Missingness ---
N_SAMPLES = 1000

data = pd.DataFrame({
    # Numerical, Skewed (Target for Median Imputation)
    'Repair_Cost': np.random.lognormal(mean=8.0, sigma=1.0, size=N_SAMPLES).round(2),
    # Numerical, Time-Series (Target for KNN Imputation)
    'Last_GPS_Lag_Hours': np.random.normal(0.5, 0.3, size=N_SAMPLES).clip(min=0),
    # Categorical/Discrete (Target for Constant Imputation)
    'Trailer_Type': np.random.choice(['Reefer', 'Flatbed', 'Dry Van', np.nan], size=N_SAMPLES, p=[0.2, 0.1, 0.6, 0.1]),
    # Numerical Feature (Used as a predictor for KNN imputation)
    'Mileage_Total': np.random.randint(50000, 500000, size=N_SAMPLES)
})

# Introduce Missing Values strategically
data.loc[data.sample(frac=0.2).index, 'Repair_Cost'] = np.nan
data.loc[data.sample(frac=0.15).index, 'Last_GPS_Lag_Hours'] = np.nan

# Ensure Repair_Cost is fixed using the stable np.maximum method for safety against Python 3.13/NumPy issue
repair_costs = np.maximum(data['Repair_Cost'].values, 1)
data['Repair_Cost'] = repair_costs

st.title("Data Imputation: Filling the Gaps for Model Readiness")
st.markdown("---")

st.header("Initial Data Audit (Missing Values)")
st.write(f"Total Records: {N_SAMPLES}")
st.dataframe(data.isnull().sum().rename("Missing Count"))
st.dataframe(data.head())


# Create a copy for imputation results
imputed_data = data.copy()

# ==============================================================================
## 2. Imputation Method 1: Median/Constant (Simple Imputer)
# ==============================================================================

st.header("2. Method 1: Simple Imputation (Median & Constant)")

st.subheader("A. Repair Cost (Median Imputation)")
st.markdown("Median imputation is used for skewed numerical data like repair costs, as the median is less affected by extreme outliers than the mean.")

# Median Imputation for Skewed Numerical Data
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputed_data['Repair_Cost_Median'] = median_imputer.fit_transform(imputed_data[['Repair_Cost']])

st.write(f"Original Repair Cost Median: ${data['Repair_Cost'].median():,.2f}")
st.write(f"Imputed Repair Cost Median: ${imputed_data['Repair_Cost_Median'].median():,.2f}")


st.subheader("B. Trailer Type (Constant Imputation) - FIX APPLIED")
st.markdown("Constant imputation marks missing categorical data with a specific label ('Unknown') to retain the information that the original data was missing.")

# Constant Imputation for Categorical Data
constant_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Unknown')

# *** FIX FOR VALUEERROR: Use .ravel() to flatten the 2D output array to 1D ***
imputed_data['Trailer_Type_Constant'] = constant_imputer.fit_transform(imputed_data[['Trailer_Type']]).ravel()

st.write("Imputed Trailer Type Counts (Note the 'Unknown' category):")
st.dataframe(imputed_data['Trailer_Type_Constant'].value_counts(dropna=False))


# ==============================================================================
## 3. Imputation Method 2: KNN Imputer (Model-Based)
# ==============================================================================

st.header("3. Method 2: KNN Imputer (Model-Based)")

st.markdown("KNN Imputation uses the other features (like Mileage) to find the 5 most similar trucks and guesses the missing GPS lag based on their values. This provides a more realistic estimate.")

# We use the two numerical columns: GPS Lag (with NaNs) and Mileage (complete)
knn_data = imputed_data[['Last_GPS_Lag_Hours', 'Mileage_Total']].copy()

# KNN Imputation with K=5
knn_imputer = KNNImputer(n_neighbors=5)
knn_imputed_array = knn_imputer.fit_transform(knn_data)

imputed_data['Last_GPS_Lag_KNN'] = knn_imputed_array[:, 0] # First column is the imputed one

# Visualize the effect of KNN vs. Original data distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(data['Last_GPS_Lag_Hours'].dropna(), kde=True, ax=ax[0])
ax[0].set_title('Original GPS Lag Distribution (Without Missing)')

sns.histplot(imputed_data['Last_GPS_Lag_KNN'], kde=True, ax=ax[1])
ax[1].set_title('KNN Imputed GPS Lag Distribution (Filled)')
st.pyplot(fig)

st.markdown("Observation: Model-based imputation (KNN) preserves the shape and variance of the original data's distribution.")
st.markdown("---")

st.header("Final Comparison of Imputed vs. Original Data")
st.dataframe(imputed_data[['Repair_Cost', 'Repair_Cost_Median', 
                          'Trailer_Type', 'Trailer_Type_Constant',
                          'Last_GPS_Lag_Hours', 'Last_GPS_Lag_KNN']].head(10))
