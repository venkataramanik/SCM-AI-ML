# ETA Predictor 2.0 — Gradient Boosting (Clean Rewrite)
# -----------------------------------------------------
# Requires:
#   pip install streamlit numpy==1.24.4 pandas matplotlib==3.8.4
#   pip install scikit-learn==1.3.2 xgboost==1.6.2 shap==0.40.0
# -----------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb
import shap

# -------------- Page config --------------
st.set_page_config(
    page_title="ETA Predictor 2.0 — Gradient Boosting",
    layout="wide"
)

# -------------- Title & intro --------------
st.title("ETA Predictor 2.0: Optimizing Delivery Performance with Gradient Boosting")

st.header("Why Gradient Boosting")
st.markdown(
    """
A decision tree is simple and auditable, but competitive operations demand the **lowest possible error**.
Gradient Boosting (XGBoost) builds many small trees **sequentially**, each correcting the prior error, to
deliver **tighter ETA windows**. We keep trust using **SHAP** to audit individual predictions.
"""
)

st.header("What is SHAP")
st.markdown(
    """
**SHAP (SHapley Additive exPlanations)** helps us understand **why a machine-learning model made a specific prediction**.  

When a model gives an ETA, SHAP breaks that number into **individual feature contributions** — showing how distance, weather, or number of stops each pushed the ETA **up** or **down**.  

Even though Gradient Boosting is a complex model, SHAP lets us **see and explain its reasoning** in a transparent, human way — building trust with dispatchers, planners, and customers who rely on its predictions.
"""
)

st.markdown("---")

# -------------- Sidebar controls --------------
with st.sidebar:
    st.header("Data Simulation Controls")
    n = st.slider("Number of trips (historical)", 500, 20000, 4000, 100)
    noise_sd = st.slider("Unobserved variation (noise, hours)", 0.0, 3.0, 1.0, 0.1)
    bad_weather_rate = st.slider("Bad weather frequency", 0.0, 0.9, 0.30, 0.05)
    stop_rate = st.slider("Probability of an extra stop", 0.0, 0.8, 0.35, 0.05)
    region_count = st.slider("Number of regions", 2, 8, 5, 1)
    seed = st.number_input("Random seed", 0, 9999, 17, 1)

    st.header("Operational Impact")
    base_handling = st.slider("Base handling time (hours)", 0.5, 4.0, 1.5, 0.1)
    rate_per_mile = st.slider("Travel rate (hours per mile)", 0.02, 0.12, 0.07, 0.005)
    weather_delay = st.slider("Bad weather delay (hours)", 0.0, 5.0, 1.2, 0.1)
    stop_delay = st.slider("Extra stop delay (hours)", 0.0, 3.0, 0.8, 0.1)
    congestion_penalty = st.slider("Congestion penalty (hours)", 0.0, 3.0, 0.7, 0.1)

    st.header("XGBoost Controls")
    n_estimators = st.slider("Number of trees (power)", 10, 200, 100, 10)
    learning_rate = st.slider("Learning rate (refinement)", 0.01, 0.3, 0.1, 0.01)
    max_depth = st.slider("Max tree depth (simplicity)", 2, 8, 4, 1)

TEST_SIZE = 0.25
FEATURES = ["distance_miles", "weather_bad", "stops", "region", "weekend"]


# -------------- Helpers --------------
@st.cache_data(show_spinner=False)
def simulate_data(
    n: int,
    noise_sd: float,
    bad_weather_rate: float,
    stop_rate: float,
    region_count: int,
    seed: int,
    base_handling: float,
    rate_per_mile: float,
    weather_delay: float,
    stop_delay: float,
    congestion_penalty: float
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    distance = rng.uniform(40, 700, n)
    weather_bad = rng.binomial(1, bad_weather_rate, n)

    raw_p = np.array([1 - stop_rate, stop_rate * 0.6, stop_rate * 0.3, stop_rate * 0.1])
    raw_p = np.maximum(0.0, raw_p)
    p_sum = raw_p.sum()
    probabilities = raw_p / p_sum if p_sum > 0 else np.array([0.25, 0.25, 0.25, 0.25])

    stops = rng.choice([0, 1, 2, 3], size=n, p=probabilities)
    region = rng.integers(0, region_count, n)
    weekend = rng.binomial(1, 0.3, n)

    # Congestion heuristic: upper-middle and top regions are congested
    region_center = (region_count - 1)
    congested = (region >= max(1, region_center - 1)).astype(int)

    # Ground-truth ETA
    eta_true = (
        base_handling
        + rate_per_mile * distance
        + weather_delay * weather_bad
        + stop_delay * stops
        + congestion_penalty * congested
        + 0.25 * weekend
    )
    # Distance-weather interaction: longer trips in bad weather get penalized more
    eta_true += (weather_bad * (np.maximum(0, distance - 350) / 150.0))

    eta = eta_true + rng.normal(0, noise_sd, n)

    df = pd.DataFrame(
        {
            "distance_miles": distance.round(1),
            "weather_bad": weather_bad,
            "stops": stops,
            "region": region,
            "weekend": weekend,
            "eta_hours": eta.round(2),
        }
    )
    return df


@st.cache_resource(show_spinner=False)
def train_models(X_train, y_train, seed, n_estimators, learning_rate, max_depth):
    # Baselines
    lin = LinearRegression().fit(X_train, y_train)
    tree = DecisionTreeRegressor(max_depth=6, min_samples_leaf=40, random_state=seed).fit(X_train, y_train)

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=seed,
        objective="reg:squarederror",
        booster="gbtree",
        n_jobs=-1
    ).fit(X_train, y_train)

    return lin, tree, xgb_model


def evaluate_models(models, X_test, y_test):
    lin, tree, xgb_model = models

    preds = {
        "Linear Regression (Simplest)": lin.predict(X_test),
        "Decision Tree (Auditable)": tree.predict(X_test),
        "Gradient Boosting (XGBoost)": xgb_model.predict(X_test),
    }

    rows = []
    for name, yhat in preds.items():
        mae = mean_absolute_error(y_test, yhat)
        rmse = np.sqrt(mean_squared_error(y_test, yhat))
        rows.append((name, mae, rmse))

    metrics = pd.DataFrame(rows, columns=["Model", "MAE (hours)", "RMSE (hours)"]).set_index("Model")
    return metrics, preds["Gradient Boosting (XGBoost)"]


def pick_case_index(y_true, y_pred):
    # choose the case with largest absolute error to make SHAP interesting
    return int(np.argmax(np.abs(y_true - y_pred)))


def plot_feature_importance(bst, feature_names):
    # Gain-based importance from XGBoost booster
    try:
        fmap = bst.get_score(importance_type="gain")
        imp_df = pd.DataFrame(
            [(feature_names[int(k.replace("f", ""))], v) for k, v in fmap.items()],
            columns=["Feature", "Gain"]
        ).sort_values("Gain", ascending=False)
        fig = plt.figure()
        plt.barh(imp_df["Feature"], imp_df["Gain"])
        plt.gca().invert_yaxis()
        plt.title("XGBoost Feature Importance (Gain)")
        plt.xlabel("Gain")
        return fig
    except Exception:
        return None


# -------------- Data --------------
df = simulate_data(
    n=n,
    noise_sd=noise_sd,
    bad_weather_rate=bad_weather_rate,
    stop_rate=stop_rate,
    region_count=region_count,
    seed=seed,
    base_handling=base_handling,
    rate_per_mile=rate_per_mile,
    weather_delay=weather_delay,
    stop_delay=stop_delay,
    congestion_penalty=congestion_penalty
)

X = df[FEATURES].values
y = df["eta_hours"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
X_test_df = pd.DataFrame(X_test, columns=FEATURES)

# -------------- Train & evaluate --------------
models = train_models(X_train, y_train, seed, n_estimators, learning_rate, max_depth)
metrics, y_xgb = evaluate_models(models, X_test, y_test)
lin, tree, xgb_model = models

# -------------- Show metrics --------------
st.header("1. Performance Metrics")
st.markdown("Gradient Boosting typically reduces error vs. the baselines, enabling tighter ETA windows.")
st.dataframe(metrics.style.format({"MAE (hours)": "{:.2f}", "RMSE (hours)": "{:.2f}"}))

mae_tree = metrics.loc["Decision Tree (Auditable)", "MAE (hours)"]
mae_xgb = metrics.loc["Gradient Boosting (XGBoost)", "MAE (hours)"]
improve_pct = ((mae_tree - mae_xgb) / mae_tree * 100) if mae_tree > 0 else 0.0

st.markdown(
    f"**Business Outcome** — Estimated **{improve_pct:.1f}% reduction** in average error versus the auditable Decision Tree."
)
st.markdown("---")

# -------------- Feature importance (model-level) --------------
st.subheader("Model-Level Drivers (XGBoost Feature Importance)")
bst = xgb_model.get_booster()
fig_imp = plot_feature_importance(bst, FEATURES)
if fig_imp:
    st.pyplot(fig_imp, clear_figure=True)
else:
    st.info("Feature importance not available in this environment.")

# -------------------- Per-shipment explanation (XGBoost native TreeSHAP) --------------------
st.header("2. Why this ETA? (Per-feature contributions)")

# Choose an interesting test case (largest absolute error)
case_idx = int(np.argmax(np.abs(y_test - y_xgb)))
X_sample = X_test[case_idx:case_idx+1, :].astype(np.float32)
predicted_eta = float(y_xgb[case_idx])

st.markdown(
    f"""
**Shipment audit example**  
- Distance: **{X_sample[0, 0]:.1f} mi**  
- Stops: **{int(X_sample[0, 2])}**  
- Weather: **{'Bad' if int(X_sample[0, 1]) else 'Clear'}**  
- Predicted ETA: **{predicted_eta:.2f} hours**
"""
)

try:
    # Use XGBoost's native SHAP (TreeSHAP) for rock-solid contributions
    booster = xgb_model.get_booster()

    # Ensure feature names flow through (important for readable bars)
    dtest_single = xgb.DMatrix(X_sample, feature_names=FEATURES)
    contribs = booster.predict(dtest_single, pred_contribs=True)  # last column = bias / expected value

    expected_value = float(contribs[0, -1])
    shap_vals = contribs[0, :-1]  # per-feature contributions

    # Build a tidy series for plotting
    contrib = pd.Series(shap_vals, index=FEATURES).sort_values()

    # Fresh, single-use figure (avoid reuse/clear issues)
    fig, ax = plt.subplots()
    ax.barh(contrib.index, contrib.values)
    ax.set_title("Per-feature contribution to ETA (hours)")
    ax.set_xlabel("Contribution (+ delays / – saves time)")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "Bars to the right **increase** ETA (delays). Bars to the left **decrease** ETA (efficiencies). "
        f"Baseline (expected) ETA ≈ **{expected_value:.2f} h**; baseline + contributions ≈ **{predicted_eta:.2f} h**."
    )

except Exception as e2:
    st.error(f"Per-shipment explanation failed: {e2}")
    st.warning("If you must use `shap` visuals, keep float32 casts, avoid `clear_figure=True`, and create a new figure each render.")



st.markdown("---")

# -------------- Data preview & download --------------
with st.expander("Sample of training data"):
    st.dataframe(df.head(20))

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download simulated dataset as CSV",
    data=csv,
    file_name="eta_simulated_data.csv",
    mime="text/csv"
)

# -------------- Final business summary --------------
st.header("3. Key Takeaways")
st.markdown(
    """
1. **Accuracy → Business Value**: Lower MAE means tighter delivery windows and fewer service misses.  
2. **Trust Preserved**: SHAP explains individual predictions, satisfying auditor and dispatcher needs.  
3. **Ready for Production**: This model can be deployed behind a real-time API and scored in milliseconds.
"""
)
