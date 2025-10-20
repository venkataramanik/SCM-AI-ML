import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. Synthesize Realistic Logistics Data ---
np.random.seed(42)
N = 1000

logistics_df = pd.DataFrame({
    # Numerical Continuous (High Range)
    'distance_miles': np.random.uniform(50, 1500, N),
    # Numerical Discrete (Low Range)
    'stops_count': np.random.randint(0, 8, N),
    # Categorical Nominal (High Cardinality) - simulates rare regions
    'delivery_region': np.random.choice(['East', 'West', 'Central', 'South', 'North'] + [f'Other_{i}' for i in range(10)], N, 
                                        p=[0.25, 0.25, 0.2, 0.1, 0.1, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
    # Categorical Nominal (Low Cardinality)
    'weather_condition': np.random.choice(['Clear', 'Rain', 'Snow', 'Heavy Fog'], N, p=[0.7, 0.15, 0.1, 0.05]),
    # Time Data (Hour of Day)
    'trip_start_hour': np.random.randint(0, 24, N),
})

print("--- Initial Data Snapshot (First 5 Rows) ---")
print(logistics_df.head())
print("-" * 50)
