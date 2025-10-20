import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# scikit-learn is used for a robust, well-tested Decision Tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

# --------------------------------------------------------------------
# LOAD DELAY RISK CLASSIFIER — DECISION TREE 
# --------------------------------------------------------------------

st.title("Load Delay Risk Classifier: The Auditable Decision Tree")

st.markdown("""
#### Business Application: Rule Discovery and Risk Management
This tool demonstrates how a **Decision Tree** model can be used to identify high-risk shipments and, crucially, provide the **exact, transparent rules** (the "why") behind every prediction. We use this to move from simple tracking to **proactive risk mitigation**.
""")

# -------------------------- Sidebar controls ------------------------
st.sidebar.header("Simulation Controls (Historical Data)")
n = st.sidebar.slider("Shipment Volume (Historical Data)", 500, 10000, 2000, 100)
late_rate_base = st.sidebar.slider("Base Late Rate (Overall Difficulty)", 0.02, 0.5, 0.15, 0.01)
bad_weather_rate = st.sidebar.slider("Bad Weather Frequency", 0.0, 0.8, 0.30, 0.05)
region_count = st.sidebar.slider("Number of Regions", 2, 6, 4, 1)
random_seed = st.sidebar.number_input("Random Seed", 0, 9999, 42, 1)

st.sidebar.header("Signal Strength (Levers of Delay)")
distance_effect = st.sidebar.slider("Distance Effect", 0.0, 1.0, 0.55, 0.05)
weather_effect = st.sidebar.slider("Weather Effect", 0.0, 1.0, 0.70, 0.05)
experience_effect = st.sidebar.slider("Low Experience Effect", 0.0, 1.0, 0.35, 0.05)
customer_tight_sla_effect = st.sidebar.slider("Tight SLA Effect", 0.0, 1.0, 0.45, 0.05)
region_effect = st.sidebar.slider("Region Variability Effect", 0.0, 1.0, 0.25, 0.05)

st.sidebar.header("Model & Alert Controls")
criterion = st.sidebar.selectbox("Impurity Measure", ["gini", "entropy"])
max_depth = st.sidebar.slider("Max Tree Depth (Model Complexity)", 2, 12, 5, 1)
min_samples_leaf = st.sidebar.slider("Min Samples per Leaf", 1, 200, 30, 1)
threshold = st.sidebar.slider("Alert Threshold (Probability of Late)", 0.1, 0.9, 0.5, 0.05)
test_size = st.sidebar.slider("Test Split Size", 0.1, 0.5, 0.25, 0.05)

st.sidebar.header("Cost of Errors (To Guide Threshold)")
cost_fp = st.sidebar.number_input("Cost of False Alert (FP)", min_value=0.0, value=1.0, step=0.5, help="Cost of unnecessary intervention/tracking.")
cost_fn = st.sidebar.number_input("Cost of Missed Delay (FN)", min_value=0.0, value=5.0, step=0.5, help="Cost of customer complaint/service failure.")

# ------------------------ Generate synthetic data -------------------
rng = np.random.default_rng(random_seed)

# Features:
distance = rng.normal(300, 110, n).clip(30, 800)
weather_bad = rng.binomial(1, bad_weather_rate, n)
driver_exp = rng.uniform(0, 12, n)
customer_tight_sla = rng.binomial(1, 0.4, n)
region = rng.integers(0, region_count, n)

# Baseline log-odds for delay
eps = 1e-9
base_logit = np.log((late_rate_base + eps) / (1 - late_rate_base + eps))

# Latent score calculation
score = (
    base_logit
    + distance_effect * ((distance - 350) / 100.0)
    + weather_effect * (weather_bad * 1.2)
    + experience_effect * ((3 - driver_exp).clip(-5, 3) / 3.0)
    + customer_tight_sla_effect * (customer_tight_sla * 0.9)
    + region_effect * ((region - region_count/2) / max(1, region_count/2))
)

# Convert to probability and realized labels
p_late = 1 / (1 + np.exp(-score))
y = (rng.uniform(0, 1, n) < p_late).astype(int)

# Build DataFrame
df = pd.DataFrame({
    "distance_miles": distance.round(1),
    "weather_bad": weather_bad,
    "driver_experience_years": driver_exp.round(1),
    "customer_tight_sla": customer_tight_sla,
    "region": region,
    "late": y
})

# --------------------------- Train / test split ---------------------
X = df[["distance_miles", "weather_bad", "driver_experience_years", "customer_tight_sla", "region"]].values
y_vec = df["late"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y_vec, test_size=test_size, random_state=random_seed, stratify=y_vec
)

# --------------------------- Decision Tree fit ----------------------
clf = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    random_state=random_seed
)
clf.fit(X_train, y_train)

# Predicted probabilities and labels using chosen threshold
probs_test = clf.predict_proba(X_test)[:, 1]
y_pred = (probs_test >= threshold).astype(int)

# ------------------------------ Metrics -----------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# ---------------------- Business-Focused Results --------------------
st.header("4. Model Performance and Operational Impact")
col_perf, col_cost = st.columns([1.5, 2])

with col_perf:
    st.subheader("Key Performance Metrics")
    col_met_val = st.columns(4)
    col_met_val[0].metric(label="Accuracy", value=f"{acc:.2f}")
    col_met_val[1].metric(label="Precision", value=f"{prec:.2f}", 
                          help="Of all flagged loads, how many were actually late. High Precision minimizes wasted effort (False Alerts).")
    col_met_val[2].metric(label="Recall", value=f"{rec:.2f}",
                          help="Of all actual late loads, how many we caught. High Recall minimizes service failures (Missed Delays).")
    col_met_val[3].metric(label="F1-Score", value=f"{f1:.2f}",
                          help="A balanced metric of Precision and Recall.")

with col_cost:
    st.subheader("Operational Cost Analysis")
    expected_cost = fp * cost_fp + fn * cost_fn
    st.metric(
        label=f"Total Estimated Error Cost for Test Set (Threshold: {threshold:.2f})",
        value=f"${expected_cost:.2f}",
        delta=f"Based on {fp} False Alerts ($ {cost_fp:.1f} each) and {fn} Missed Delays ($ {cost_fn:.1f} each)",
        delta_color="off"
    )

st.markdown("---")

# --------------------------- Confusion matrix ----------------------
st.subheader("Confusion Matrix: Where Errors Occur")
cm_df = pd.DataFrame(
    cm,
    index=pd.Index(["Actual On-Time (0)", "Actual Late (1)"], name="Reality"),
    columns=pd.Index(["Pred On-Time (0)", "Pred Late (1)"], name="Prediction")
)
st.dataframe(cm_df)

# ------------------------- Feature importance ----------------------
st.header("5. Feature Importance: Identifying the Key Levers")
st.markdown("This shows which factors the model relies on most often to successfully separate late vs. on-time loads. **These are your operational levers.**")

col_fi_chart, col_fi_table = st.columns([2, 1])

feat_names = ["distance_miles", "weather_bad", "driver_experience_years", "customer_tight_sla", "region"]
importances = clf.feature_importances_
fi_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)

with col_fi_chart:
    fig_fi, ax_fi = plt.subplots(figsize=(7, 4))
    colors = plt.cm.viridis(fi_df["importance"] / fi_df["importance"].max())
    ax_fi.barh(fi_df["feature"], fi_df["importance"], color=colors)
    ax_fi.set_xlabel("Relative Importance (%)")
    ax_fi.set_title("Which Inputs Drive the Decision Rules")
    fig_fi.tight_layout()
    st.pyplot(fig_fi)

with col_fi_table:
    st.dataframe(fi_df, use_container_width=True)

st.markdown("---")

# -------------------------- Tree visualization ---------------------
st.header("6. Model Audit: The Decision Flowchart")
st.caption("The Decision Tree is the model itself—an auditable set of if/then rules. Each split is a decision (a 'Tipping Point') the model discovered.")
fig_tree, ax_tree = plt.subplots(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=feat_names,
    class_names=["On-Time", "Late"],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax_tree,
    impurity=False, # Hides Gini/Entropy for cleaner business view
    label="none" # Removes the large label for better visualization
)
fig_tree.tight_layout()
st.pyplot(fig_tree)

# -------------------------- Scenario analysis ----------------------
st.header("7. Scenario Estimation: Predict and Explain (No Black Box)")

# Function to get the leaf node index for a given input
def get_leaf_index(tree, X_input):
    return tree.apply(X_input.reshape(1, -1))[0]

col_scen1, col_scen2, col_scen3 = st.columns(3)
with col_scen1:
    s_distance = st.slider("Distance (miles)", 30, 800, 320, 5)
    s_experience = st.slider("Driver Experience (years)", 0, 12, 4, 1)
with col_scen2:
    s_weather = st.selectbox("Weather", ["clear (0)", "bad (1)"])
    s_customer_sla = st.selectbox("Customer SLA", ["normal (0)", "tight (1)"])
with col_scen3:
    s_region = st.slider("Region", 0, max(0, region_count - 1), min(1, region_count // 2), 1)

# User input vector
x_user = np.array([
    [float(s_distance),
     1 if s_weather.startswith("bad") else 0,
     float(s_experience),
     1 if s_customer_sla.startswith("tight") else 0,
     int(s_region)]
])

# Prediction
p_user = float(clf.predict_proba(x_user)[:, 1])
leaf_index = get_leaf_index(clf, x_user)
leaf_value = clf.tree_.value[leaf_index]
late_count = leaf_value[0, 1]
total_count = leaf_value.sum()


st.markdown("""
<style>
.metric-box {
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #ccc;
    background-color: #f0f2f6;
}
.alert-risk {
    background-color: #ffe5e5; /* Light red */
    border-left: 5px solid red;
}
.ok-risk {
    background-color: #e5ffe5; /* Light green */
    border-left: 5px solid green;
}
</style>
""", unsafe_allow_html=True)


is_alert = p_user >= threshold
risk_class = "alert-risk" if is_alert else "ok-risk"

st.markdown(
    f"""
    <div class='metric-box {risk_class}'>
        <p style='font-size: 1.2em; margin-bottom: 5px;'>Predicted Probability of Late: <strong>{p_user:.2f}</strong></p>
        <p style='margin-bottom: 5px;'><strong>Decision: {'ALERT - PROACTIVE INTERVENTION NEEDED' if is_alert else 'OK - NO ACTION REQUIRED'}</strong> (Threshold: {threshold:.2f})</p>
        <p style='font-size: 0.9em; color: #555;'>
            <i>Explanation: This scenario landed in a terminal node (leaf) where {late_count:.0f} out of {total_count:.0f} historical samples were late.</i>
        </p>
    </div>
    """, unsafe_allow_html=True
)

# -------------------------- Final Business Summary -------------------
st.markdown("---")
st.header("Business Interpretation: The Value of the Decision Tree")

st.markdown(
    """
    The Decision Tree is not just a predictor; it's a **diagnostic tool** and a **policy discovery engine**.

    * **Proactive Intervention (Threshold):** By setting the **Alert Threshold**, the business decides its acceptable level of risk. A lower threshold prioritizes catching every potential delay (**High Recall**) but increases unnecessary work (**High FP Cost**). A higher threshold reduces overhead (**Low FP Cost**) but risks service failures (**High FN Cost**).
    * **Actionable Rules:** The **Decision Tree Diagram** provides the exact operational rule that triggered the alert (e.g., "Distance > 450 miles **AND** Driver Experience < 3 years"). This level of transparency builds **trust** with operations teams.
    * **Strategic Investment (Feature Importance):** The **Feature Importance Chart** tells management where to invest long-term capital. If *Weather Effect* is consistently the top lever, the investment should go into weather-proofing the fleet or dynamic rerouting tools.
    """
)
