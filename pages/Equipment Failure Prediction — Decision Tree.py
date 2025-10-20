import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import _tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

# --------------------------------------------------------------------
# EQUIPMENT FAILURE PREDICTION — DECISION TREE (Business-Focused Demo)
# --------------------------------------------------------------------

st.title("Equipment Failure Prediction — Decision Tree")

st.markdown("""
#### Purpose
- Predict which vehicles are likely to experience a mechanical failure in the next 7 days.
- Provide clear, auditable rules for maintenance planning and dispatch.

#### What this shows
- How a Decision Tree converts sensor and usage data into if/then rules.
- How to handle rare events by using thresholds and cost trade-offs.
- How to explain alerts to maintenance teams with plain-language rule paths.

#### How IoT data feeds this model
- Vehicles stream sensor readings over OBD-II or CAN bus into a fleet platform at frequent intervals.
- Data is aggregated hourly or daily into features: maximums, minimums, averages, and recent alert counts.
- Each record becomes a labeled example: inputs are sensor features, target is whether a failure occurred within 7 days.
- The tree learns threshold rules, such as: if engine_temp_max > 230 and oil_pressure_min < 20 and vibration_rms > 4.5, then failure risk is high.

#### Tech focus in plain terms
- The model tries questions that best separate failures from healthy vehicles.
- It measures how mixed a group is and chooses questions that make groups purer.
- Depth and minimum leaf size control how specific the rules get to avoid overfitting.
""")

# -------------------------- Sidebar controls ------------------------
st.sidebar.header("Simulation controls")
n = st.sidebar.slider("Number of vehicle days", 1000, 40000, 8000, 500)
base_fail_rate = st.sidebar.slider("Base failure rate (rare event rate)", 0.002, 0.10, 0.02, 0.002)
region_count = st.sidebar.slider("Number of regions", 2, 10, 6, 1)
seed = st.sidebar.number_input("Random seed", 0, 9999, 202, 1)

st.sidebar.header("Signal strength (how much each factor contributes)")
engine_temp_effect = st.sidebar.slider("Engine temperature effect", 0.0, 2.0, 1.2, 0.1)
oil_pressure_effect = st.sidebar.slider("Low oil pressure effect", 0.0, 2.0, 1.0, 0.1)
vibration_effect = st.sidebar.slider("Vibration effect", 0.0, 2.0, 0.9, 0.1)
mileage_effect = st.sidebar.slider("Mileage since service effect", 0.0, 2.0, 0.8, 0.1)
coolant_alerts_effect = st.sidebar.slider("Recent coolant alerts effect", 0.0, 2.0, 1.1, 0.1)
battery_low_effect = st.sidebar.slider("Low battery voltage effect", 0.0, 2.0, 0.5, 0.1)
tire_pressure_effect = st.sidebar.slider("Low tire pressure effect", 0.0, 2.0, 0.4, 0.1)
ambient_heat_effect = st.sidebar.slider("Ambient heat effect", 0.0, 2.0, 0.6, 0.1)
region_effect = st.sidebar.slider("Region variability effect", 0.0, 2.0, 0.3, 0.1)

st.sidebar.header("Model controls")
criterion = st.sidebar.selectbox("Impurity measure", ["gini", "entropy"])
max_depth = st.sidebar.slider("Max tree depth", 2, 16, 6, 1)
min_samples_leaf = st.sidebar.slider("Min samples per leaf", 1, 1000, 80, 1)
test_size = st.sidebar.slider("Test split size", 0.1, 0.5, 0.25, 0.05)
threshold = st.sidebar.slider("Alert threshold (probability of failure)", 0.05, 0.95, 0.40, 0.05)

st.sidebar.header("Cost of errors")
cost_fp = st.sidebar.number_input(
    "Cost of false alert (unnecessary shop visit)",
    min_value=0.0, value=1.0, step=0.5
)
cost_fn = st.sidebar.number_input(
    "Cost of missed failure (roadside breakdown)",
    min_value=0.0, value=10.0, step=0.5
)

# ------------------------ Generate synthetic data -------------------
rng = np.random.default_rng(seed)

# Features (daily aggregates per vehicle)
# engine_temp_max (°F), oil_pressure_min (psi), vibration_rms, mileage_since_service (miles),
# coolant_alerts_7d (count), battery_voltage_min (V), tire_pressure_avg (psi),
# ambient_temp (°F), region_id (0..region_count-1)
engine_temp_max = rng.normal(205, 18, n)            # higher increases risk
oil_pressure_min = rng.normal(36, 6, n)             # lower increases risk
vibration_rms = np.abs(rng.normal(2.5, 1.0, n))     # higher increases risk
mileage_since_service = np.abs(rng.normal(12000, 5000, n))  # higher increases risk
coolant_alerts_7d = rng.poisson(0.15, n)            # more alerts increases risk
battery_voltage_min = rng.normal(12.2, 0.6, n)      # lower increases risk
tire_pressure_avg = rng.normal(34, 3.0, n)          # lower increases risk
ambient_temp = rng.normal(72, 18, n)                # higher can increase risk
region_id = rng.integers(0, region_count, n)        # some regions tougher duty cycles

# Convert to a latent failure score (logit), centered around base rate
# Base log-odds for the rare event
eps = 1e-9
base_logit = np.log((base_fail_rate + eps) / (1 - base_fail_rate + eps))

# Normalize key ranges to make the sliders meaningful
score = (
    base_logit
    + engine_temp_effect * ((engine_temp_max - 220) / 15.0)         # >220F risky
    + oil_pressure_effect * ((30 - oil_pressure_min) / 8.0)         # <30 psi risky
    + vibration_effect * ((vibration_rms - 3.5) / 1.2)              # >3.5 g risky
    + mileage_effect * ((mileage_since_service - 15000) / 6000.0)   # >15k since service risky
    + coolant_alerts_effect * (coolant_alerts_7d * 0.8)             # frequent coolant alerts risky
    + battery_low_effect * ((12.0 - battery_voltage_min) / 0.6)     # low voltage risky
    + tire_pressure_effect * ((32.0 - tire_pressure_avg) / 2.5)     # underinflation risky
    + ambient_heat_effect * ((ambient_temp - 90) / 10.0)            # very hot days risky
    + region_effect * ((region_id - (region_count - 1) / 2) / max(1, (region_count - 1) / 2))
)

p_fail = 1 / (1 + np.exp(-score))
y = (rng.uniform(0, 1, n) < p_fail).astype(int)  # 1 = Fail in next 7 days, 0 = Healthy

df = pd.DataFrame({
    "engine_temp_max_f": engine_temp_max.round(1),
    "oil_pressure_min_psi": oil_pressure_min.round(1),
    "vibration_rms": vibration_rms.round(2),
    "mileage_since_service_mi": mileage_since_service.astype(int),
    "coolant_alerts_7d": coolant_alerts_7d,
    "battery_voltage_min_v": battery_voltage_min.round(2),
    "tire_pressure_avg_psi": tire_pressure_avg.round(1),
    "ambient_temp_f": ambient_temp.round(1),
    "region_id": region_id,
    "fail_7d": y
})

st.markdown("#### Sample data")
st.dataframe(df.head(12))

# --------------------------- Train / test split ---------------------
feature_cols = [
    "engine_temp_max_f", "oil_pressure_min_psi", "vibration_rms",
    "mileage_since_service_mi", "coolant_alerts_7d", "battery_voltage_min_v",
    "tire_pressure_avg_psi", "ambient_temp_f", "region_id"
]
X = df[feature_cols].values
y_vec = df["fail_7d"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y_vec, test_size=test_size, random_state=seed, stratify=y_vec
)

# --------------------------- Decision Tree fit ----------------------
clf = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    random_state=seed
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

st.markdown("#### Results overview")
st.write(f"- Accuracy: {acc:.3f}")
st.write(f"- Precision (of flagged vehicles, how many truly fail): {prec:.3f}")
st.write(f"- Recall (of all true failures, how many we catch): {rec:.3f}")
st.write(f"- F1 (balance of precision and recall): {f1:.3f}")

expected_cost = fp * cost_fp + fn * cost_fn
st.write(f"- Estimated cost at this threshold = FP × {cost_fp:.1f} + FN × {cost_fn:.1f} = {expected_cost:.1f}")

# --------------------------- Confusion matrix ----------------------
st.markdown("#### Confusion matrix")
cm_df = pd.DataFrame(
    cm,
    index=pd.Index(["Actual Healthy (0)", "Actual Fail (1)"], name="Reality"),
    columns=pd.Index(["Pred Healthy (0)", "Pred Fail (1)"], name="Prediction")
)
st.dataframe(cm_df)

# ----------------------------- ROC curve ---------------------------
st.markdown("#### ROC curve and AUC")
fpr, tpr, _ = roc_curve(y_test, probs_test)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax_roc.plot([0, 1], [0, 1], linestyle="--", label="No-skill")
ax_roc.set_xlabel("False positive rate")
ax_roc.set_ylabel("True positive rate")
ax_roc.set_title("ROC")
ax_roc.legend(loc="lower right")
fig_roc.tight_layout()
st.pyplot(fig_roc)

# ---------------------- Precision-Recall curve ---------------------
st.markdown("#### Precision–Recall curve")
prec_curve, rec_curve, thr_curve = precision_recall_curve(y_test, probs_test)
fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
ax_pr.plot(rec_curve, prec_curve)
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision–Recall")
fig_pr.tight_layout()
st.pyplot(fig_pr)

# ------------------------- Feature importance ----------------------
st.markdown("#### Feature importance")
importances = clf.feature_importances_
fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
st.dataframe(fi_df)

fig_fi, ax_fi = plt.subplots(figsize=(7, 4))
ax_fi.bar(fi_df["feature"], fi_df["importance"])
ax_fi.set_xlabel("Feature")
ax_fi.set_ylabel("Relative importance")
ax_fi.set_title("Which inputs drive the alert rules")
fig_fi.tight_layout()
st.pyplot(fig_fi)

# ---------------------- Rule path for a test sample ----------------
st.markdown("#### Explain a specific alert (rule path)")
st.caption("Pick a vehicle-day from the test set to see the exact if/then path that led to the alert.")
idx_choice = st.slider("Select test index to explain", 0, max(0, len(X_test)-1), 0, 1)

def explain_path(tree: DecisionTreeClassifier, feature_names, x_row):
    node_indicator = tree.decision_path(x_row.reshape(1, -1))
    leaf_id = tree.apply(x_row.reshape(1, -1))[0]
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    lines = []
    for node_id in node_indicator.indices:
        if leaf_id == node_id:
            continue
        fname = feature_names[feature[node_id]]
        thresh = threshold[node_id]
        if x_row[feature[node_id]] <= thresh:
            decision = f"{fname} <= {thresh:.3f}"
        else:
            decision = f"{fname} > {thresh:.3f}"
        lines.append(decision)
    return lines

x_row = X_test[idx_choice]
path_lines = explain_path(clf, feature_cols, x_row)
st.write("- Rule path")
for line in path_lines:
    st.write(f"  - {line}")
p_row = float(probs_test[idx_choice])
st.write(f"- Predicted probability of failure: {p_row:.2f}")
st.write(f"- Action at threshold {threshold:.2f}: {'Alert' if p_row >= threshold else 'OK'}")

# -------------------------- Tree visualization ---------------------
st.markdown("#### Decision tree diagram")
st.caption("Each box shows a rule, how many samples it covers, and class mix. Depth is limited by the max depth control.")
fig_tree, ax_tree = plt.subplots(figsize=(12, 7))
plot_tree(
    clf,
    feature_names=feature_cols,
    class_names=["Healthy", "Fail"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax_tree
)
fig_tree.tight_layout()
st.pyplot(fig_tree)

# -------------------------- What-if analysis -----------------------
st.markdown("#### What-if analysis")
st.caption("Adjust sensor values to see how risk changes. Use this to simulate preventive actions before dispatch.")
c1, c2, c3 = st.columns(3)
with c1:
    w_engine = st.slider("Engine temp max (°F)", 170, 260, 230, 1)
    w_oil = st.slider("Oil pressure min (psi)", 10, 60, 28, 1)
    w_vib = st.slider("Vibration RMS", 0.0, 8.0, 4.2, 0.1)
with c2:
    w_miles = st.slider("Mileage since service (mi)", 0, 40000, 18000, 500)
    w_cool = st.slider("Coolant alerts last 7d", 0, 10, 2, 1)
    w_batt = st.slider("Battery voltage min (V)", 9.0, 14.0, 11.8, 0.1)
with c3:
    w_tire = st.slider("Tire pressure avg (psi)", 20.0, 45.0, 31.0, 0.5)
    w_amb = st.slider("Ambient temp (°F)", -20.0, 120.0, 95.0, 1.0)
    w_region = st.slider("Region id", 0, max(0, region_count - 1), min(1, region_count // 2), 1)

x_whatif = np.array([[
    float(w_engine), float(w_oil), float(w_vib),
    int(w_miles), int(w_cool), float(w_batt),
    float(w_tire), float(w_amb), int(w_region)
]])
p_whatif = float(clf.predict_proba(x_whatif)[:, 1])
st.write(f"- Predicted probability of failure: {p_whatif:.2f}")
st.write(f"- Action at threshold {threshold:.2f}: {'Alert' if p_whatif >= threshold else 'OK'}")

# -------------------------- Business explanation -------------------
st.markdown("""
### Business interpretation
- The model learns clear rules. Examples:
  - If engine temperature is high and oil pressure is low, risk rises sharply.
  - Frequent coolant alerts or high vibration after long mileage since service raises risk.
- Feature importance ranks the strongest drivers of failure, guiding maintenance priorities.
- Threshold and cost sliders help select an operating point that minimizes expected business impact.

### How to use this operationally
- Sort vehicles by predicted risk and schedule preventive checks for the top band.
- Review rule paths with maintenance leads so alerts are trustworthy and actionable.
- Track precision and recall monthly; raise or lower the alert threshold as seasons and routes change.
- Feed real-time sensor streams into the model to generate early warnings for dispatch.

### How the model decides, in plain terms
- The tree tests questions like:
  - Is engine temperature greater than a certain number.
  - Is oil pressure below a certain number.
  - Has mileage since service exceeded a limit.
- It chooses questions that best separate likely failures from healthy vehicles.
- It repeats until further questions do not improve separation or until we hit depth limits.
""")
