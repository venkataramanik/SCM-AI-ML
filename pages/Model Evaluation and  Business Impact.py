# =========================
# 1. Model Evaluation & Business Impact
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

st.header("Model Evaluation & Business Impact")

# --------- Business levers (you can tune live) ----------
with st.sidebar:
    st.subheader("Business Levers")
    cost_per_minute = st.number_input("Ops cost per delay minute ($/min)", 0.10, 20.0, 1.50, 0.10)
    shipments_per_day = st.number_input("Shipments per day", 50, 50000, 3500, 50)
    penalty_per_miss = st.number_input("Avg penalty per SLA miss ($)", 0.0, 500.0, 50.0, 5.0)
    promised_window_hr = st.number_input("Promised delivery window (hours)", 1.0, 12.0, 4.0, 0.5)

# --------- Safe access to data/models from earlier ----------
FEATURES = ["distance_miles", "weather_bad", "stops", "region", "weekend"]
if "df" in globals():
    base_df = df.copy()
else:
    # fall back: if df not found, rebuild quickly from earlier variables
    base_df = pd.DataFrame(X, columns=FEATURES)
    base_df["eta_hours"] = y

# Ensure we have a test split and predictions
def ensure_predictions():
    global X_train, X_test, y_train, y_test, y_lin, y_tree, y_xgb, lin, tree, xgb_model
    try:
        # if values exist, just return
        _ = y_test, y_lin, y_tree, y_xgb
        return
    except Exception:
        pass

    X_all = base_df[FEATURES].values.astype(np.float32)
    y_all = base_df["eta_hours"].values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=17)

    # Train quick baselines if missing
    lin = LinearRegression().fit(X_train, y_train)
    tree = DecisionTreeRegressor(max_depth=6, min_samples_leaf=40, random_state=17).fit(X_train, y_train)

    # Fall back XGB if missing in scope
    try:
        xgb_model
    except NameError:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=4, random_state=17,
            objective="reg:squarederror", booster="gbtree", n_jobs=-1
        ).fit(X_train, y_train)

    y_lin  = lin.predict(X_test)
    y_tree = tree.predict(X_test)
    y_xgb  = xgb_model.predict(X_test)

ensure_predictions()

# --------- Metrics table ----------
def metrics_for(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

mae_lin,  rmse_lin,  r2_lin  = metrics_for(y_test, y_lin)
mae_tree, rmse_tree, r2_tree = metrics_for(y_test, y_tree)
mae_xgb,  rmse_xgb,  r2_xgb  = metrics_for(y_test, y_xgb)

metrics_df = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Gradient Boosting (XGBoost)"],
    "MAE (h)": [mae_lin, mae_tree, mae_xgb],
    "RMSE (h)": [rmse_lin, rmse_tree, rmse_xgb],
    "R²": [r2_lin, r2_tree, r2_xgb],
})
metrics_df["MAE (min)"]  = (metrics_df["MAE (h)"] * 60).round(1)
metrics_df["RMSE (min)"] = (metrics_df["RMSE (h)"] * 60).round(1)

st.subheader("Performance Metrics")
st.dataframe(metrics_df.style.format({"MAE (h)":"{:.2f}","RMSE (h)":"{:.2f}","R²":"{:.3f}"}))

# --------- Business impact: minutes → dollars ----------
improve_vs_tree_min = (mae_tree - mae_xgb) * 60.0
daily_minutes_saved = improve_vs_tree_min * shipments_per_day
daily_savings = daily_minutes_saved * cost_per_minute

# Simple SLA miss approximation: error beyond half-window counts as a miss
half_window = promised_window_hr / 2.0
sla_miss_rate_tree = float(np.mean(np.abs(y_test - y_tree) > half_window))
sla_miss_rate_xgb  = float(np.mean(np.abs(y_test - y_xgb)  > half_window))
misses_avoided_per_day = (sla_miss_rate_tree - sla_miss_rate_xgb) * shipments_per_day
penalty_savings_per_day = max(0.0, misses_avoided_per_day) * penalty_per_miss

st.subheader("Business Impact")
col1, col2, col3 = st.columns(3)
col1.metric("MAE reduction vs Tree (min)", f"{improve_vs_tree_min:.1f}")
col2.metric("Daily time saved (min)", f"{daily_minutes_saved:,.0f}")
col3.metric("Daily cost savings ($)", f"{daily_savings:,.0f}")

col4, col5, col6 = st.columns(3)
col4.metric("SLA miss rate - Tree", f"{sla_miss_rate_tree*100:.1f}%")
col5.metric("SLA miss rate - XGB", f"{sla_miss_rate_xgb*100:.1f}%")
col6.metric("Penalty savings/day ($)", f"{penalty_savings_per_day:,.0f}")

st.caption(
    "We translate model accuracy into dollars using your ops levers: cost per delay minute, shipments/day, "
    "and average penalty per miss. Lower MAE tightens delivery windows and reduces SLA misses."
)

st.markdown("---")

# --------- Visuals: residuals + error by distance ----------
st.subheader("Where errors happen")

# Residuals
residuals = pd.DataFrame({
    "Residual (h)": y_test - y_xgb
})
fig1 = plt.figure()
plt.hist(residuals["Residual (h)"], bins=40)
plt.title("Residuals (XGBoost)")
plt.xlabel("Prediction error (hours)")
plt.ylabel("Count")
st.pyplot(fig1)

# Error by distance
if "distance_miles" in base_df.columns:
    X_test_df = pd.DataFrame(X_test, columns=FEATURES)
    err_abs = np.abs(y_test - y_xgb)
    plot_df = pd.DataFrame({
        "distance_miles": X_test_df["distance_miles"],
        "abs_error_hr": err_abs
    })
    fig2 = plt.figure()
    plt.scatter(plot_df["distance_miles"], plot_df["abs_error_hr"], s=10, alpha=0.5)
    plt.title("Absolute Error by Distance")
    plt.xlabel("Distance (miles)")
    plt.ylabel("Absolute error (hours)")
    st.pyplot(fig2)

st.markdown("---")

# --------- Drift guardrail (simple demo) ----------
st.subheader("Drift Guardrail (Demo)")
wo_mae = mae_xgb  # pretend "this week" MAE
prev_mae = mae_xgb * 0.9  # pretend "last week" MAE
drift_pct = ((wo_mae - prev_mae) / max(prev_mae, 1e-6)) * 100

alert_threshold = 10.0  # %
if drift_pct > alert_threshold:
    st.error(f"MAE drifted by {drift_pct:.1f}% vs. last week — trigger retraining.")
else:
    st.success(f"MAE drift is {drift_pct:.1f}% vs. last week — within tolerance.")
st.caption("In production, this compares weekly MAE and triggers an automated retrain if drift exceeds a threshold.")
