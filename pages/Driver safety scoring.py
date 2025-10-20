import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

# --------------------------------------------------------------------
# DRIVER SAFETY SCORING — DECISION TREE (Business-Focused Demo)
# --------------------------------------------------------------------

st.title("Driver Safety Scoring — Decision Tree")

st.markdown("""
#### Purpose
- Identify drivers at higher risk of a safety incident in the next 7 days.
- Provide clear, explainable rules for coaching and policy interventions.

#### What this shows
- How a Decision Tree turns telematics and schedule context into if/then rules.
- How to tune alert thresholds based on cost trade-offs.
- How to read metrics that matter to safety managers: precision, recall, and F1.

#### Model focus
- The model asks simple questions that split the data into clearer groups.
- Splits are chosen to reduce mixing of outcomes (incident vs no incident).
- Depth and minimum leaf size limit complexity to avoid overfitting.
""")

# -------------------------- Sidebar controls ------------------------
st.sidebar.header("Simulation controls")
n = st.sidebar.slider("Number of driver-weeks", 500, 40000, 8000, 500)
base_incident_rate = st.sidebar.slider("Base incident rate (rare event rate)", 0.002, 0.10, 0.03, 0.002)
region_count = st.sidebar.slider("Number of regions", 2, 10, 6, 1)
vehicle_classes = st.sidebar.slider("Number of vehicle classes", 1, 6, 3, 1)
seed = st.sidebar.number_input("Random seed", 0, 9999, 77, 1)

st.sidebar.header("Signal strength (how much each factor contributes)")
speeding_effect = st.sidebar.slider("Speeding events effect", 0.0, 2.0, 1.0, 0.1)
braking_effect = st.sidebar.slider("Harsh braking effect", 0.0, 2.0, 0.9, 0.1)
cornering_effect = st.sidebar.slider("Harsh cornering effect", 0.0, 2.0, 0.7, 0.1)
fatigue_effect = st.sidebar.slider("Fatigue score effect", 0.0, 2.0, 0.8, 0.1)
night_hours_effect = st.sidebar.slider("Night hours effect", 0.0, 2.0, 0.6, 0.1)
weather_effect = st.sidebar.slider("Bad weather exposure effect", 0.0, 2.0, 0.5, 0.1)
urban_effect = st.sidebar.slider("Urban driving exposure effect", 0.0, 2.0, 0.4, 0.1)
region_effect = st.sidebar.slider("Region variability effect", 0.0, 2.0, 0.3, 0.1)
vehicle_class_effect = st.sidebar.slider("Vehicle class effect", 0.0, 2.0, 0.3, 0.1)

st.sidebar.header("Model controls")
criterion = st.sidebar.selectbox("Impurity measure", ["gini", "entropy"])
max_depth = st.sidebar.slider("Max tree depth", 2, 16, 6, 1)
min_samples_leaf = st.sidebar.slider("Min samples per leaf", 1, 1000, 80, 1)
test_size = st.sidebar.slider("Test split size", 0.1, 0.5, 0.25, 0.05)
threshold = st.sidebar.slider("Alert threshold (probability of incident)", 0.05, 0.95, 0.40, 0.05)

st.sidebar.header("Cost of errors")
cost_fp = st.sidebar.number_input(
    "Cost of false alert (unnecessary coaching/escalation)",
    min_value=0.0, value=1.0, step=0.5
)
cost_fn = st.sidebar.number_input(
    "Cost of missed incident (accident/violation)",
    min_value=0.0, value=15.0, step=0.5
)

# ------------------------ Generate synthetic data -------------------
rng = np.random.default_rng(seed)

# Features per driver-week (telematics + context)
# speeding_events_7d, harsh_brake_events_7d, cornering_g_events_7d,
# fatigue_score (0..100), night_hours_7d (0..60), bad_weather_rate (0..1),
# urban_pct (0..1), vehicle_class (0..vehicle_classes-1), region_id (0..region_count-1)

speeding_ev = rng.poisson(2.0, n)
harsh_brake_ev = rng.poisson(1.6, n)
cornering_ev = rng.poisson(1.2, n)

fatigue_score = np.clip(rng.normal(45, 15, n), 0, 100)      # higher worse
night_hours = np.clip(rng.normal(12, 8, n), 0, 60)
bad_weather_rate = np.clip(rng.beta(2, 6, n), 0, 1)         # fraction of hours in bad weather
urban_pct = np.clip(rng.beta(3, 3, n), 0, 1)

vehicle_class = rng.integers(0, vehicle_classes, n)
region_id = rng.integers(0, region_count, n)

# Base log-odds for incident
eps = 1e-9
base_logit = np.log((base_incident_rate + eps) / (1 - base_incident_rate + eps))

# Latent score from factors
score = (
    base_logit
    + speeding_effect * ((speeding_ev - 3.0) / 2.5)
    + braking_effect * ((harsh_brake_ev - 2.0) / 2.0)
    + cornering_effect * ((cornering_ev - 1.8) / 1.5)
    + fatigue_effect * ((fatigue_score - 60.0) / 20.0)
    + night_hours_effect * ((night_hours - 20.0) / 10.0)
    + weather_effect * ((bad_weather_rate - 0.35) / 0.2)
    + urban_effect * ((urban_pct - 0.6) / 0.2)
    + region_effect * ((region_id - (region_count - 1) / 2) / max(1, (region_count - 1) / 2))
    + vehicle_class_effect * ((vehicle_class - (vehicle_classes - 1) / 2) / max(1, (vehicle_classes - 1) / 2))
)

p_incident = 1 / (1 + np.exp(-score))
y = (rng.uniform(0, 1, n) < p_incident).astype(int)  # 1 = Incident within 7 days

df = pd.DataFrame({
    "speeding_events_7d": speeding_ev,
    "harsh_brake_events_7d": harsh_brake_ev,
    "cornering_g_events_7d": cornering_ev,
    "fatigue_score": fatigue_score.round(1),
    "night_hours_7d": night_hours.round(1),
    "bad_weather_rate": bad_weather_rate.round(2),
    "urban_pct": urban_pct.round(2),
    "vehicle_class": vehicle_class,
    "region_id": region_id,
    "incident_7d": y
})

st.markdown("#### Sample data")
st.dataframe(df.head(12))

# --------------------------- Train / test split ---------------------
feature_cols = [
    "speeding_events_7d", "harsh_brake_events_7d", "cornering_g_events_7d",
    "fatigue_score", "night_hours_7d", "bad_weather_rate", "urban_pct",
    "vehicle_class", "region_id"
]
X = df[feature_cols].values
y_vec = df["incident_7d"].values

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
st.write(f"- Precision (of flagged drivers, how many had incidents): {prec:.3f}")
st.write(f"- Recall (of all incidents, how many we caught): {rec:.3f}")
st.write(f"- F1 (balance of precision and recall): {f1:.3f}")

expected_cost = fp * cost_fp + fn * cost_fn
st.write(f"- Estimated cost at this threshold = FP × {cost_fp:.1f} + FN × {cost_fn:.1f} = {expected_cost:.1f}")

# --------------------------- Confusion matrix ----------------------
st.markdown("#### Confusion matrix")
cm_df = pd.DataFrame(
    cm,
    index=pd.Index(["Actual Safe (0)", "Actual Incident (1)"], name="Reality"),
    columns=pd.Index(["Pred Safe (0)", "Pred Incident (1)"], name="Prediction")
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
ax_fi.set_title("Which inputs drive the safety rules")
fig_fi.tight_layout()
st.pyplot(fig_fi)

# ------------------- Behavior slice: fatigue vs night hours --------
st.markdown("#### Risk slice: fatigue and night hours")
fat_grid = np.linspace(10, 90, 120)
night_grid = np.linspace(0, 50, 120)
hold = {
    "speed": 2, "brake": 2, "corner": 2,
    "weather": 0.2, "urban": 0.5,
    "vclass": int(max(0, vehicle_classes // 2)),
    "region": int(max(0, region_count // 2))
}
X_slice = np.c_[
    np.full_like(fat_grid, hold["speed"]),
    np.full_like(fat_grid, hold["brake"]),
    np.full_like(fat_grid, hold["corner"]),
    fat_grid,
    night_grid,
    np.full_like(fat_grid, hold["weather"]),
    np.full_like(fat_grid, hold["urban"]),
    np.full_like(fat_grid, hold["vclass"]),
    np.full_like(fat_grid, hold["region"])
]
p_slice = clf.predict_proba(X_slice)[:, 1]
fig_slice, ax_slice = plt.subplots(figsize=(6, 4))
ax_slice.plot(fat_grid, p_slice, label="risk vs fatigue (night hrs increases along x)")
ax_slice.axhline(threshold, linestyle="--")
ax_slice.set_xlabel("Fatigue score")
ax_slice.set_ylabel("Predicted probability of incident")
ax_slice.set_title("Model behavior along fatigue/night dimension (others fixed)")
fig_slice.tight_layout()
st.pyplot(fig_slice)

# -------------------------- Tree visualization ---------------------
st.markdown("#### Decision tree diagram")
st.caption("Each box shows a rule, how many samples it covers, and class mix. Depth is limited by the max depth control.")
fig_tree, ax_tree = plt.subplots(figsize=(12, 7))
plot_tree(
    clf,
    feature_names=feature_cols,
    class_names=["Safe", "Incident"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax_tree
)
fig_tree.tight_layout()
st.pyplot(fig_tree)

# -------------------------- Scenario estimation --------------------
st.markdown("#### Scenario estimation")
c1, c2, c3 = st.columns(3)
with c1:
    s_speed = st.slider("Speeding events (7d)", 0, 20, 3, 1)
    s_brake = st.slider("Harsh brake events (7d)", 0, 20, 2, 1)
    s_corner = st.slider("Harsh cornering events (7d)", 0, 20, 2, 1)
with c2:
    s_fatigue = st.slider("Fatigue score", 0, 100, 55, 1)
    s_night = st.slider("Night hours (7d)", 0, 60, 18, 1)
    s_weather = st.slider("Bad weather rate", 0.0, 1.0, 0.25, 0.05)
with c3:
    s_urban = st.slider("Urban driving percent", 0.0, 1.0, 0.6, 0.05)
    s_vclass = st.slider("Vehicle class", 0, max(0, vehicle_classes - 1), min(1, vehicle_classes // 2), 1)
    s_region = st.slider("Region", 0, max(0, region_count - 1), min(1, region_count // 2), 1)

x_user = np.array([[
    int(s_speed), int(s_brake), int(s_corner),
    float(s_fatigue), float(s_night), float(s_weather),
    float(s_urban), int(s_vclass), int(s_region)
]])
p_user = float(clf.predict_proba(x_user)[:, 1])
st.write(f"- Predicted probability of incident: {p_user:.2f}")
st.write(f"- Action at threshold {threshold:.2f}: {'Alert' if p_user >= threshold else 'OK'}")

# -------------------------- Business explanation -------------------
st.markdown("""
### Business interpretation
- The model learns clear, reviewable rules. Examples:
  - If fatigue is high and night hours are high, risk increases sharply.
  - Frequent speeding and harsh braking combine to raise risk.
- Feature importance ranks the strongest drivers; this supports targeted coaching.
- Threshold and cost sliders let you choose an operating point that matches policy.

### How to use this operationally
- Sort driver-weeks by predicted risk and prioritize coaching or rest scheduling.
- Review rules with safety managers; align alerts with policy guidelines.
- Track precision and recall over time and by region; adjust the threshold seasonally.
- Combine with policy actions (training refresh, shift rotation, lane changes).

### How the model decides
- The tree tries questions like:
  - Are speeding events above a certain count.
  - Is the fatigue score above a certain level.
  - Are night hours unusually high.
- It picks questions that best separate incident vs no incident and repeats until further splits do not help or hit limits.
""")
