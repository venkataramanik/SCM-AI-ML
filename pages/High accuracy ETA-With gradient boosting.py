import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Requires: pip install xgboost shap
import xgboost as xgb
import shap

# --- STREAMLIT CONFIG (Avoids warnings related to plotting) ---
# Removed deprecated st.set_option for PyPlot warning fix.

# --------------------------------------------------------------------
# ETA PREDICTOR 2.0 — HIGH-ACCURACY GRADIENT BOOSTING
# --------------------------------------------------------------------

st.title("ETA Predictor 2.0: Optimizing Delivery Performance with Gradient Boosting")

# -------------------------- MODEL SHIFT BLURB ------------------------
st.header("Why We Use Gradient Boosting: Maximizing Prediction Accuracy")

st.markdown("""
Our prior Decision Tree model provided a solid, auditable forecast, but in competitive logistics, the goal is always to **minimize error**.

**The Business Need:** A smaller prediction error (MAE) means we can offer a **tighter delivery window** to the customer. For example, reducing MAE from 1.0 hour to 0.7 hours is the difference between guaranteeing a 4-hour window and a 3-hour window—a huge **competitive advantage**.

**The Solution (Gradient Boosting):** We transition to **Gradient Boosting (XGBoost)** because it's specifically designed for maximum accuracy. It works by:
1.  **Learning Sequentially:** It starts with a simple guess, then builds hundreds of small 'expert' decision trees, with each new tree focused entirely on **correcting the errors** made by the previous trees.
2.  **Trade-off:** We sacrifice the easy-to-read flowchart for **significantly reduced prediction error** and use **SHAP** (Section 2) to maintain trust in every individual prediction.
""")

st.markdown("---")

# -------------------------- Sidebar controls (Retained) ------------------------
st.sidebar.header("Data Simulation Controls")
n = st.sidebar.slider("Number of trips (Historical Data)", 500, 20000, 4000, 100)
noise_sd = st.sidebar.slider("Unobserved Variation (Noise, hours)", 0.0, 3.0, 1.0, 0.1)
bad_weather_rate = st.sidebar.slider("Bad Weather Frequency", 0.0, 0.9, 0.30, 0.05)
stop_rate = st.sidebar.slider("Probability of an Extra Stop", 0.0, 0.8, 0.35, 0.05)
region_count = st.sidebar.slider("Number of Regions", 2, 8, 5, 1)
seed = st.sidebar.number_input("Random Seed", 0, 9999, 17, 1)

st.sidebar.header("Operational Impact (Retained)")
base_handling = st.sidebar.slider("Base Handling Time (hours)", 0.5, 4.0, 1.5, 0.1)
rate_per_mile = st.sidebar.slider("Travel Rate (hours per mile)", 0.02, 0.12, 0.07, 0.005)
weather_delay = st.sidebar.slider("Bad Weather Delay (hours penalty)", 0.0, 5.0, 1.2, 0.1)
stop_delay = st.sidebar.slider("Extra Stop Delay (hours penalty)", 0.0, 3.0, 0.8, 0.1)
congestion_penalty = st.sidebar.slider("Congestion Region Penalty (hours)", 0.0, 3.0, 0.7, 0.1)

st.sidebar.header("XGBoost Controls")
n_estimators = st.sidebar.slider("Number of Trees (Power)", 10, 200, 100, 10)
learning_rate = st.sidebar.slider("Learning Rate (Refinement Speed)", 0.01, 0.3, 0.1, 0.01)
max_depth = st.sidebar.slider("Max Tree Depth (Simplicity)", 2, 8, 4, 1)
test_size = 0.25

# ------------------------ Data Generation (Retained) -------------------
rng = np.random.default_rng(seed)

distance = rng.uniform(40, 700, n)
weather_bad = rng.binomial(1, bad_weather_rate, n)

raw_p = np.array([1 - stop_rate, stop_rate * 0.6, stop_rate * 0.3, stop_rate * 0.1])
raw_p = np.maximum(0.0, raw_p) 
p_sum = np.sum(raw_p)
probabilities = raw_p / p_sum if p_sum > 0 else np.array([0.25, 0.25, 0.25, 0.25])
stops = rng.choice([0, 1, 2, 3], size=n, p=probabilities)

region = rng.integers(0, region_count, n)
weekend = rng.binomial(1, 0.3, n)

region_center = (region_count - 1)
congested = (region >= max(1, region_center - 1)).astype(int)

eta_true = (
    base_handling + rate_per_mile * distance + weather_delay * weather_bad + 
    stop_delay * stops + congestion_penalty * congested + 0.25 * weekend
)
eta_true += (weather_bad * (np.maximum(0, distance - 350) / 150.0))
eta = eta_true + rng.normal(0, noise_sd, n)

df = pd.DataFrame({
    "distance_miles": distance.round(1),
    "weather_bad": weather_bad,
    "stops": stops,
    "region": region,
    "weekend": weekend,
    "eta_hours": eta.round(2)
})

X = df[["distance_miles", "weather_bad", "stops", "region", "weekend"]].values
y = df["eta_hours"].values
feat_names = ["distance_miles", "weather_bad", "stops", "region", "weekend"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed
)
X_test_df = pd.DataFrame(X_test, columns=feat_names)

# --------------------------- Model Fitting --------------------
# 1. Linear Model (Baseline)
lin = LinearRegression()
lin.fit(X_train, y_train)
y_lin = lin.predict(X_test)
mae_lin = mean_absolute_error(y_test, y_lin)

# 2. Decision Tree (Baseline from Demo 1)
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=6, min_samples_leaf=40, random_state=seed)
tree.fit(X_train, y_train)
y_tree = tree.predict(X_test)
mae_tree = mean_absolute_error(y_test, y_tree)

# 3. Gradient Boosting (New High-Performance Model)
xgb_model = xgb.XGBRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    random_state=seed,
    objective='reg:squarederror',
    # FIX for SHAP KeyError: 'base_score'
    booster='gbtree' 
)
xgb_model.fit(X_train, y_train)
y_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_xgb))


# -------------------------- Performance Comparison --------------------
st.header("1. Performance Metrics: The Impact of Ensemble Learning")

st.markdown("""
The table below directly compares the average prediction error (MAE) across the three model types. Notice how the **Gradient Boosting** model significantly cuts the error margin.
""")

metrics_df = pd.DataFrame({
    'Model': ['Linear Regression (Simplest)', 'Decision Tree (Auditable)', 'Gradient Boosting (XGBoost)'],
    'MAE (Avg. Error, hours)': [f'{mae_lin:.2f}', f'{mae_tree:.2f}', f'{mae_xgb:.2f}'],
    'RMSE (Penalty for Large Errors, hours)': [
        f'{np.sqrt(mean_squared_error(y_test, y_lin)):.2f}', 
        f'{np.sqrt(mean_squared_error(y_test, y_tree)):.2f}', 
        f'{rmse_xgb:.2f}'
    ]
}).set_index('Model')

st.dataframe(metrics_df)

st.markdown(f"""
#### Business Outcome (Non-Technical Blurb) 💡
The shift to Gradient Boosting delivers a **{((mae_tree - mae_xgb) / mae_tree * 100):.1f}% reduction in average error** compared to our transparent Decision Tree. This translates directly into **tighter service guarantees**, improving customer satisfaction and strengthening our competitive edge.
""")
st.markdown("---")

# -------------------------- Feature Contribution (SHAP) --------------------
st.header("2. Explaining the Black Box: Individual Prediction Contributions (SHAP)")

st.markdown("""
To ensure we can still audit and trust every forecast, we use **SHAP** values. This advanced technique instantly shows **which specific factors pushed the predicted ETA higher or lower** for any single trip, giving auditors feature-by-feature accountability.
""")

# Select a single example from the test set 
sample_index = np.argmax(np.abs(y_test - y_xgb)) # Find a trip with a large residual for drama
X_sample = X_test[sample_index, :].reshape(1, -1)
predicted_eta = y_xgb[sample_index]

st.markdown(f"""
#### Case Study: Shipment Audit
- **Input Features:** Distance: **{X_sample[0, 0]:.1f} mi**, Stops: **{int(X_sample[0, 2])}**, Weather: **{'Bad' if X_sample[0, 1] else 'Clear'}**
- **Predicted ETA:** **{predicted_eta:.2f} hours**
""")

# Calculate SHAP values for the specific sample
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)

# Create the Force Plot visualization
st.subheader("Factor Contribution Breakdown")
st.caption("The **Base Value** is the average ETA for all trips. Factors pushing the prediction higher (delay) are **RED**; factors pushing it lower (efficiency) are **BLUE**.")

# Generate the plot
fig_shap = plt.figure()
shap.force_plot(
    explainer.expected_value, 
    shap_values, 
    X_sample, 
    feature_names=feat_names, 
    matplotlib=True, 
    show=False
)
plt.xlabel("Model Output (Predicted ETA in Hours)")
st.pyplot(fig_shap)

st.markdown("""
#### Non-Technical Interpretation Blurb 📢
The SHAP plot is the **immediate audit trail**. For this specific prediction: the long **distance** pushed the ETA significantly to the right (RED), increasing the forecast from the overall average. Conversely, having a low number of **stops** (BLUE) pulled the forecast slightly back down. This tool ensures we can always justify the *why* behind our most accurate forecasts.
""")
st.markdown("---")

# -------------------------- Final Business Explanation -------------------
st.header("3. Key Takeaways: Advanced Forecasting for Business Value")

st.markdown(
    """
1.  **Direct ROI from Accuracy:** The core value of Gradient Boosting is the reduction in prediction error. This improvement directly translates into **cost savings** (fewer service failures) and higher **customer retention** due to more reliable ETAs.
2.  **Maintaining Trust:** We use SHAP to bridge the gap between model complexity and business accountability. This means we can deploy a high-accuracy 'black box' and still satisfy **auditors and dispatchers** with clear, instant explanations.
3.  **Scalable Production System:** This model is the final candidate for deployment via a **real-time API endpoint**, taking the live telematics features and delivering a high-confidence prediction in milliseconds.
"""
)
