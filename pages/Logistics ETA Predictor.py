import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------------------------
# ETA PREDICTOR — REGRESSION TREE (Business-Focused Demo)
# --------------------------------------------------------------------

st.title("Logistics ETA Predictor: Transparent Forecasting with Regression Trees")

# -------------------------- Data Origin Blurb ------------------------
st.header("Data Origin: From Telematics Fire Hose")

st.markdown("""
In modern logistics, the raw data for ETA predictions comes from a continuous **"fire hose"** of real-time signals.

1.  **The Source (Telematics):** Every vehicle is equipped with a telematics unit that constantly streams **GPS coordinates, speed, accelerometer readings,** and other sensor data.
2.  **The Stream (Event Bus):** This raw, high-volume data is immediately pushed into a high-throughput, low-latency system like **Apache Kafka** or a similar **event bus/stream processing platform**.
3.  **The Processing:** This data is then **cleaned, filtered, and aggregated** in real-time. For instance, thousands of raw pings are transformed into the aggregated operational features (like `stops` and `weather_bad`) consumed by our model.
""")

st.markdown("---")

# -------------------------- Model Goal ------------------------
st.header("Model Goal: Accurate and Trustworthy Delivery Estimates")

st.markdown("""
This tool demonstrates how a **Regression Tree** can predict Estimated Time of Arrival (ETA) in hours. Unlike opaque models, the tree provides **interpretable, rule-based logic** that operations teams can understand, audit, and trust.

The Core Problem: Transportation times are **nonlinear**. A simple "hours per mile" calculation (Linear Regression) fails when unexpected factors like bad weather, extra stops, or highly congested regions introduce step-changes in delay. The Regression Tree is designed to capture these operational **"tipping points."**
""")

# -------------------------- Sidebar controls ------------------------
st.sidebar.header("Data Simulation Controls")
n = st.sidebar.slider("Number of trips (Historical Data)", 500, 20000, 4000, 100)
noise_sd = st.sidebar.slider("Unobserved Variation (Noise, hours)", 0.0, 3.0, 1.0, 0.1, help="Represents non-modeled factors like unexpected detours or minor loading delays.")
bad_weather_rate = st.sidebar.slider("Bad Weather Frequency", 0.0, 0.9, 0.30, 0.05)
stop_rate = st.sidebar.slider("Probability of an Extra Stop", 0.0, 0.8, 0.35, 0.05)
region_count = st.sidebar.slider("Number of Regions", 2, 8, 5, 1)
seed = st.sidebar.number_input("Random Seed", 0, 9999, 17, 1)

st.sidebar.header("Operational Impact (How Factors Shape ETA)")
base_handling = st.sidebar.slider("Base Handling Time (hours)", 0.5, 4.0, 1.5, 0.1)
rate_per_mile = st.sidebar.slider("Travel Rate (hours per mile)", 0.02, 0.12, 0.07, 0.005)
weather_delay = st.sidebar.slider("Bad Weather Delay (hours penalty)", 0.0, 5.0, 1.2, 0.1)
stop_delay = st.sidebar.slider("Extra Stop Delay (hours penalty)", 0.0, 3.0, 0.8, 0.1)
congestion_penalty = st.sidebar.slider("Congestion Region Penalty (hours)", 0.0, 3.0, 0.7, 0.1)

st.sidebar.header("Model Controls")
max_depth = st.sidebar.slider("Max Tree Depth (Complexity)", 2, 16, 6, 1, help="Limits rule specificity to prevent overfitting.")
min_samples_leaf = st.sidebar.slider("Min Samples per Leaf (Rule Stability)", 1, 500, 40, 1, help="Ensures rules are based on a large, stable group of historical trips.")
test_size = st.sidebar.slider("Test Split Size", 0.1, 0.5, 0.25, 0.05)
show_linear_baseline = st.sidebar.checkbox("Compare against Linear Regression Baseline", value=True)

# ------------------------ Generate synthetic data -------------------
rng = np.random.default_rng(seed)

# Features
distance = rng.uniform(40, 700, n)
weather_bad = rng.binomial(1, bad_weather_rate, n)

# CORRECTED: Normalized probability calculation for 'stops'
raw_p = np.array([
    1 - stop_rate,  
    stop_rate * 0.6, 
    stop_rate * 0.3, 
    stop_rate * 0.1 
])

raw_p = np.maximum(0.0, raw_p) 
p_sum = np.sum(raw_p)

if p_sum > 0:
    probabilities = raw_p / p_sum
else:
    probabilities = np.array([0.25, 0.25, 0.25, 0.25])

stops = rng.choice([0, 1, 2, 3], size=n, p=probabilities)
# END CORRECTION


region = rng.integers(0, region_count, n)
weekend = rng.binomial(1, 0.3, n)

# Region congestion index
region_center = (region_count - 1)
congested = (region >= max(1, region_center - 1)).astype(int)

# Ground-truth ETA with nonlinearities and interactions
eta_true = (
    base_handling
    + rate_per_mile * distance
    + weather_delay * weather_bad
    + stop_delay * stops
    + congestion_penalty * congested
    + 0.25 * weekend
)

# Crucial Interaction: weather penalty grows after long distance (nonlinear)
eta_true += (weather_bad * (np.maximum(0, distance - 350) / 150.0))

# Add noise
eta = eta_true + rng.normal(0, noise_sd, n)

df = pd.DataFrame({
    "distance_miles": distance.round(1),
    "weather_bad": weather_bad,
    "stops": stops,
    "region": region,
    "weekend": weekend,
    "eta_hours": eta.round(2)
})

st.markdown("#### Historical Data Sample (Aggregated Telematics Features)")
st.dataframe(df.head(12))

# --------------------------- Train / test split ---------------------
X = df[["distance_miles", "weather_bad", "stops", "region", "weekend"]].values
y = df["eta_hours"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed
)

# --------------------------- Regression Tree fit --------------------
tree = DecisionTreeRegressor(
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    random_state=seed
)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
# FIXED: Calculate RMSE manually for compatibility with older sklearn versions
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) 
r2 = r2_score(y_test, y_pred)

# ------------------------ Linear baseline (optional) ----------------
if show_linear_baseline:
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_lin = lin.predict(X_test)
    
    mae_lin = mean_absolute_error(y_test, y_lin)
    # FIXED: Calculate RMSE manually for compatibility with older sklearn versions
    mse_lin = mean_squared_error(y_test, y_lin)
    rmse_lin = np.sqrt(mse_lin)
    r2_lin = r2_score(y_test, y_lin)

# -------------------------- Metrics & Comparison --------------------
st.header("1. Model Performance and Accuracy")

col_tree, col_lin = st.columns(2)

with col_tree:
    st.subheader("Decision Tree Model Accuracy")
    st.metric(label="MAE (Mean Absolute Error)", value=f"{mae:.2f} hours", help="The average time difference between our prediction and the actual arrival time.")
    st.metric(label="RMSE (Root Mean Squared Error)", value=f"{rmse:.2f} hours", help="Penalizes large errors more heavily. Useful for setting service guarantees.")
    st.metric(label="R² (Variance Explained)", value=f"{r2:.3f}", help="Fraction of the total variability in ETA explained by the model.")

if show_linear_baseline:
    with col_lin:
        st.subheader("Linear Regression Baseline")
        st.metric(label="MAE (Mean Absolute Error)", value=f"{mae_lin:.2f} hours")
        st.metric(label="RMSE (Root Mean Squared Error)", value=f"{rmse_lin:.2f} hours")
        st.metric(label="R² (Variance Explained)", value=f"{r2_lin:.3f}")

st.markdown(f"""
#### Business Insight: Setting the Service Buffer
If your **Mean Absolute Error (MAE)** is **{mae:.2f} hours**, you know, on average, your prediction is off by this amount. To guarantee 95% **On-Time Performance (OTP)**, you must set your published ETA to be **Predicted ETA + (Noise SD $\\times$ safety factor)**. A low RMSE means you can offer a tighter, more competitive ETA window.
""")
st.markdown("---")

# ------------------- Partial dependence style slices ---------------
st.header("2. Model Behavior: Capturing Tipping Points")
st.markdown("The Decision Tree captures **nonlinear relationships** where a simple linear model cannot. Note how the line for the Tree is step-wise, reflecting the discrete rules it has learned.")
colA, colB = st.columns(2)

# Slice over distance holding other inputs
with colA:
    st.write("Predicted ETA vs. Distance (The Step-Function)")
    dist_grid = np.linspace(df["distance_miles"].min(), df["distance_miles"].max(), 250)
    w_hold = 0
    stops_hold = 1
    region_hold = int(region_count // 2)
    weekend_hold = 0
    X_slice = np.c_[dist_grid, np.full_like(dist_grid, w_hold), np.full_like(dist_grid, stops_hold),
                    np.full_like(dist_grid, region_hold), np.full_like(dist_grid, weekend_hold)]
    y_slice_tree = tree.predict(X_slice)
    
    # Linear baseline slice for direct comparison
    X_slice_lin = np.c_[dist_grid, np.full_like(dist_grid, w_hold), np.full_like(dist_grid, stops_hold),
                        np.full_like(dist_grid, region_hold), np.full_like(dist_grid, weekend_hold)]
    y_slice_lin = lin.predict(X_slice_lin) if show_linear_baseline else np.zeros_like(dist_grid)

    fig_slice1, ax_slice1 = plt.subplots(figsize=(6, 4))
    ax_slice1.plot(dist_grid, y_slice_tree, label="Regression Tree", linewidth=3)
    if show_linear_baseline:
        ax_slice1.plot(dist_grid, y_slice_lin, label="Linear Model", linestyle='--', color='gray')

    ax_slice1.set_xlabel("Distance (miles)")
    ax_slice1.set_ylabel("Predicted ETA (hours)")
    ax_slice1.set_title("Tree captures stable ETA 'brackets'")
    ax_slice1.legend()
    fig_slice1.tight_layout()
    st.pyplot(fig_slice1)

# Slice over weather showing distance interaction
with colB:
    st.write("Interaction: Weather Delay vs. Distance")
    dist_grid2 = np.linspace(80, 650, 180)
    
    X_slice_clear = np.c_[dist_grid2, np.zeros_like(dist_grid2), np.ones_like(dist_grid2),
                          np.full_like(dist_grid2, region_hold), np.zeros_like(dist_grid2)]
    X_slice_bad = np.c_[dist_grid2, np.ones_like(dist_grid2), np.ones_like(dist_grid2),
                        np.full_like(dist_grid2, region_hold), np.zeros_like(dist_grid2)]
    y_clear = tree.predict(X_slice_clear)
    y_bad = tree.predict(X_slice_bad)
    
    fig_slice2, ax_slice2 = plt.subplots(figsize=(6, 4))
    ax_slice2.plot(dist_grid2, y_clear, label="Clear Weather ETA")
    ax_slice2.plot(dist_grid2, y_bad, label="Bad Weather ETA", color='red')
    ax_slice2.fill_between(dist_grid2, y_clear, y_bad, alpha=0.1, color='red', label="Weather Delay Gap")

    ax_slice2.set_xlabel("Distance (miles)")
    ax_slice2.set_ylabel("Predicted ETA (hours)")
    ax_slice2.set_title("The Bad Weather Penalty is NOT constant")
    ax_slice2.legend()
    fig_slice2.tight_layout()
    st.pyplot(fig_slice2)

st.markdown("---")

# ------------------------- Feature importance ----------------------
st.header("3. Auditability: Which Factors Matter Most?")
st.markdown("The feature importance helps identify the most effective levers for reducing or stabilizing ETA.")

feat_names = ["distance_miles", "weather_bad", "stops", "region", "weekend"]
importances = tree.feature_importances_
fi_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
st.dataframe(fi_df, use_container_width=True)

# -------------------------- Tree visualization ---------------------
st.header("4. Operational Rules: The Decision Flowchart")
st.caption("Each split is a discovered rule. The final value in the box is the **average predicted ETA (in hours)** for all trips that follow that path.")
fig_tree, ax_tree = plt.subplots(figsize=(11, 8))
plot_tree(
    tree,
    feature_names=feat_names,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax_tree,
    # Customize for business clarity
    label="none",
    precision=1
)
fig_tree.tight_layout()
st.pyplot(fig_tree)

# -------------------------- Scenario estimation --------------------
st.header("5. Scenario Analysis: Predict and Explain")
st.markdown("Test a specific shipment's parameters to get its predicted ETA based on the learned rules.")

c1, c2, c3 = st.columns(3)
with c1:
    s_distance = st.slider("Distance (miles)", 40, 700, 320, 5)
    s_stops = st.slider("Stops (count)", 0, 3, 1, 1)
with c2:
    s_weather = st.selectbox("Weather", ["clear (0)", "bad (1)"])
    s_weekend = st.selectbox("Weekend", ["weekday (0)", "weekend (1)"])
with c3:
    s_region = st.slider("Region", 0, max(0, region_count - 1), min(1, region_count // 2), 1)

x_user = np.array([[
    float(s_distance),
    1 if s_weather.startswith("bad") else 0,
    int(s_stops),
    int(s_region),
    1 if s_weekend.startswith("weekend") else 0
]])

# Prediction and Rule tracing
eta_user = float(tree.predict(x_user))

# Simplified feature map for explanation
feat_map = {0: "distance_miles", 1: "weather_bad", 2: "stops", 3: "region", 4: "weekend"}
node_indicator = tree.decision_path(x_user).toarray()[0]
path_rules = []
for i, indicator in enumerate(node_indicator):
    if indicator and tree.tree_.children_left[i] != tree.tree_.children_right[i]: # It's a non-leaf node
        feature = tree.tree_.feature[i]
        threshold = tree.tree_.threshold[i]
        is_left = x_user[0, feature] <= threshold
        
        rule = f"{feat_map[feature]} {'<=' if is_left else '>'} {threshold:.2f}"
        path_rules.append(rule)

st.write(f"- Predicted ETA: **{eta_user:.2f} hours**")
if path_rules:
    st.write(f"- **Rule Path Used (for audit):** The prediction was determined by the sequence of rules: `{' AND '.join(path_rules[:max_depth])}`")
else:
    st.write("- **Rule Path Used (for audit):** Prediction based on the root node average.")

st.markdown("---")

# -------------------------- Business explanation -------------------
st.header("Key Takeaways for Logistics Operations")

st.markdown(
    """
1.  **Trust Through Transparency:** The Regression Tree's strength is that it's a **white-box model**. Every ETA is traceable back to a small, finite set of if/then rules visible in the flowchart. This is vital for **dispute resolution** and building trust with planning teams.
2.  **Validating Operational Knowledge:** Review the **Regression Tree Diagram** with your dispatchers. If a split (e.g., Distance > 400 miles) aligns with their real-world experience, the model is incorporating true operational knowledge.
3.  **Tighter Service Levels:** By capturing the **nonlinear interactions** (like how bad weather affects long hauls disproportionately, Section 2), the tree produces a much more accurate ETA than a linear model, allowing the business to offer **tighter, more competitive service windows** without jeopardizing on-time delivery guarantees.
"""
)
