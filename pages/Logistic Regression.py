import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# LOGISTIC REGRESSION – BUSINESS DEMO (On-Time Probability)
# Includes: MLE/log-loss tracking, Accuracy/Precision/Recall/F1, Calibration, ROC
# ============================================================================

st.title("Operational Risk Scoring with Logistic Regression")

st.markdown("""
### 1) Business Question
Which shipments are **at risk of being late**, and by **how much** (probability)?  
We want a transparent risk score that dispatch and customer service can act on.

### 2) Modeling Approach
**Logistic Regression** estimates the **probability of on-time arrival (0–1)** from factors like distance,
weather, and driver experience. You choose a **threshold** (e.g., 0.60) to convert probabilities into actions.
""")

# ------------------------------- Sidebar -----------------------------------
st.sidebar.header("Simulation Controls")
n = st.sidebar.slider("Number of shipments", 200, 10000, 1500, 100)
noise_scale = st.sidebar.slider("Outcome randomness (logit noise)", 0.0, 2.0, 0.6, 0.1)
bad_weather_rate = st.sidebar.slider("Bad weather frequency", 0.0, 0.9, 0.35, 0.05)
exp_max = st.sidebar.slider("Max driver experience (years)", 5, 25, 10, 1)

st.sidebar.header("Training (Gradient Descent)")
lr = st.sidebar.slider("Learning rate", 0.0005, 0.5, 0.05, 0.0005)
epochs = st.sidebar.slider("Iterations", 200, 20000, 3000, 100)
fit_intercept = st.sidebar.checkbox("Fit intercept", value=True)
show_loss = st.sidebar.checkbox("Show log-loss convergence", value=True)

st.sidebar.header("Scoring & Display")
threshold = st.sidebar.slider("Action threshold (p on-time)", 0.1, 0.9, 0.6, 0.05)
show_data = st.sidebar.checkbox("Show sample data", value=False)
show_coeffs = st.sidebar.checkbox("Show coefficients", value=True)
show_calibration = st.sidebar.checkbox("Show calibration (binned)", value=True)
show_roc = st.sidebar.checkbox("Show ROC", value=True)
show_pr = st.sidebar.checkbox("Show Precision-Recall curve", value=False)

st.sidebar.header("Cost of Errors (optional)")
cost_fp = st.sidebar.number_input("Cost of False Positive (alert when OK)", min_value=0.0, value=1.0, step=0.5)
cost_fn = st.sidebar.number_input("Cost of False Negative (miss a late load)", min_value=0.0, value=5.0, step=0.5)

# --------------------------- Synthetic Data --------------------------------
rng = np.random.default_rng(42)
distance = rng.normal(300, 100, n)                     # miles
weather = rng.binomial(1, bad_weather_rate, n)         # 1=bad, 0=clear
experience = rng.uniform(0, exp_max, n)                # years

# "True" relationship (unknown to the learner)
# Intercept=+2.0, distance=-0.006, bad weather=-1.0, experience=+0.25
z = 2.0 + (-0.006)*distance + (-1.0)*weather + (0.25)*experience
z_noisy = z + rng.normal(0, noise_scale, n)            # unobserved factors
p_on_time_true = 1 / (1 + np.exp(-z_noisy))
y = (rng.uniform(0, 1, n) < p_on_time_true).astype(int)  # 1=On-Time, 0=Late

df = pd.DataFrame({
    "distance_miles": distance,
    "weather_bad": weather,
    "driver_experience_years": experience,
    "on_time": y,
    "p_on_time_true": p_on_time_true
})
if show_data:
    st.subheader("Sample Data")
    st.dataframe(df.head(10))

# --------------------------- Logistic Regression ---------------------------
def sigmoid(a):
    return 1 / (1 + np.exp(-np.clip(a, -500, 500)))

def prepare_X(X, add_intercept=True):
    return np.hstack([np.ones((X.shape[0], 1)), X]) if add_intercept else X

X = np.c_[distance, weather, experience]
Xb = prepare_X(X, add_intercept=fit_intercept)
y_vec = y.reshape(-1, 1)

# Initialize weights small random
w = rng.normal(0.0, 0.01, (Xb.shape[1], 1))

loss_history = []
for _ in range(epochs):
    logits = Xb @ w
    probs = sigmoid(logits)
    # Gradient of negative log-likelihood (cross-entropy)
    grad = (Xb.T @ (probs - y_vec)) / Xb.shape[0]
    w -= lr * grad
    # Log-loss (negative log-likelihood) tracking
    eps = 1e-10
    loss = -np.mean(y_vec*np.log(probs + eps) + (1 - y_vec)*np.log(1 - probs + eps))
    loss_history.append(loss)

probs_hat = sigmoid(Xb @ w).ravel()
preds = (probs_hat >= threshold).astype(int)

# ------------------------------- Metrics -----------------------------------
acc = (preds == y).mean()
tp = int(((preds==1) & (y==1)).sum())
tn = int(((preds==0) & (y==0)).sum())
fp = int(((preds==1) & (y==0)).sum())
fn = int(((preds==0) & (y==1)).sum())

# Extended metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

st.subheader("Quick Results")
st.write(f"**Accuracy:** {acc:.3f}  |  **Precision:** {precision:.3f}  |  **Recall:** {recall:.3f}  |  **F1:** {f1:.3f}")
st.write(f"Threshold: {threshold:.2f}  |  **Final Log-Loss:** {loss_history[-1]:.4f}")

cm = pd.DataFrame(
    [[tp, fp],
     [fn, tn]],
    index=pd.Index(["Pred On-Time=1 (Alerts)", "Pred On-Time=0 (No Alerts)"], name="Prediction"),
    columns=pd.Index(["Actual On-Time=1", "Actual On-Time=0"], name="Reality")
)
st.write("**Confusion Matrix**")
st.dataframe(cm)

# Cost of errors
expected_cost = fp * cost_fp + fn * cost_fn
st.write(f"**Estimated Cost** (given current threshold):  FP×{cost_fp:.1f} + FN×{cost_fn:.1f} = **{expected_cost:.1f}**")

# ------------------------------- Loss Plot ---------------------------------
if show_loss:
    st.subheader("Training Error (Log-Loss over Iterations)")
    fig_loss, ax_loss = plt.subplots(figsize=(6,4))
    ax_loss.plot(loss_history, label="Log-Loss")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss (negative log-likelihood)")
    ax_loss.set_title("Model Convergence (MLE via Log-Loss Minimization)")
    ax_loss.legend()
    st.pyplot(fig_loss)

# ------------------------------- Plots -------------------------------------
# 1) Probability vs Distance (holding other factors)
st.subheader("Probability Curve (holding weather and experience fixed)")
fig1, ax1 = plt.subplots(figsize=(8,5))
dist_grid = np.linspace(max(30, distance.min()), distance.max()+30, 300)
weather_hold = 0  # clear
exp_hold = max(1.0, exp_max/2)
X_grid = np.c_[dist_grid, np.full_like(dist_grid, weather_hold), np.full_like(dist_grid, exp_hold)]
Xb_grid = prepare_X(X_grid, add_intercept=fit_intercept)
p_grid = sigmoid(Xb_grid @ w).ravel()
ax1.plot(dist_grid, p_grid, label=f"p(On-Time) | weather=clear, exp≈{exp_hold:.1f}y")
ax1.axhline(threshold, linestyle="--", label="action threshold")
ax1.set_xlabel("Distance (miles)")
ax1.set_ylabel("Predicted Probability of On-Time")
ax1.set_title("Learned Logistic Relationship")
ax1.legend(loc="best")
fig1.tight_layout()
st.pyplot(fig1)

# 2) Histogram of predicted probabilities
st.subheader("Risk Distribution (Predicted Probabilities)")
fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.hist(probs_hat, bins=30, alpha=0.9)
ax2.axvline(threshold, linestyle="--")
ax2.set_xlabel("Predicted p(On-Time)")
ax2.set_ylabel("Shipments")
ax2.set_title("Distribution of Risk Scores")
fig2.tight_layout()
st.pyplot(fig2)

# 3) Calibration: Do 0.8 scores happen ~80% of the time?
if show_calibration:
    st.subheader("Calibration (Do 0.8 scores happen ~80% of the time?)")
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(probs_hat, bins) - 1
    cal_rows = []
    for b in range(10):
        mask = idx == b
        if mask.sum() > 0:
            cal_rows.append([0.5*(bins[b]+bins[b+1]), y[mask].mean(), mask.sum()])
    if cal_rows:
        cal = np.array(cal_rows)
        fig3, ax3 = plt.subplots(figsize=(6,5))
        ax3.plot([0,1],[0,1],"--", label="perfect")
        ax3.scatter(cal[:,0], cal[:,1], s=10+2*cal[:,2], label="binned")
        ax3.set_xlabel("Predicted Probability (bin center)")
        ax3.set_ylabel("Actual On-Time Rate")
        ax3.set_title("Calibration Plot")
        ax3.legend(loc="best")
        fig3.tight_layout()
        st.pyplot(fig3)
    else:
        st.info("Not enough data to compute calibration bins.")

# 4) ROC curve (no sklearn)
def compute_roc(y_true, score):
    thr = np.unique(np.sort(score))[::-1]
    tpr_list, fpr_list = [], []
    P = (y_true==1).sum()
    N = (y_true==0).sum()
    tp_cum = fp_cum = 0
    order = np.argsort(-score)
    y_sorted = y_true[order]
    s_sorted = score[order]
    i = 0
    for t in thr:
        while i < len(s_sorted) and s_sorted[i] >= t:
            if y_sorted[i]==1: tp_cum += 1
            else: fp_cum += 1
            i += 1
        tpr_list.append(tp_cum / P if P>0 else 0)
        fpr_list.append(fp_cum / N if N>0 else 0)
    return np.array(fpr_list), np.array(tpr_list)

if show_roc:
    st.subheader("ROC Curve (Ranking Quality)")
    fpr, tpr = compute_roc(y, probs_hat)
    auc = np.trapz(tpr, fpr)  # trapezoid AUC
    fig4, ax4 = plt.subplots(figsize=(6,5))
    ax4.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    ax4.plot([0,1],[0,1], "--", label="random")
    ax4.set_xlabel("False Positive Rate")
    ax4.set_ylabel("True Positive Rate")
    ax4.set_title("Receiver Operating Characteristic")
    ax4.legend(loc="lower right")
    fig4.tight_layout()
    st.pyplot(fig4)

# 5) Precision-Recall curve (optional, no sklearn)
def compute_pr(y_true, score):
    # thresholds high->low; compute precision, recall at each cut
    order = np.argsort(-score)
    y_sorted = y_true[order]
    tp, fp = 0, 0
    P = (y_true==1).sum()
    precisions, recalls = [], []
    for i in range(len(score)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / P if P > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
    return np.array(recalls), np.array(precisions)

if show_pr:
    st.subheader("Precision-Recall Curve (Class-Imbalance Friendly)")
    rec, prec = compute_pr(y, probs_hat)
    fig5, ax5 = plt.subplots(figsize=(6,5))
    ax5.plot(rec, prec, label="PR curve")
    ax5.set_xlabel("Recall")
    ax5.set_ylabel("Precision")
    ax5.set_title("Precision-Recall")
    ax5.legend(loc="best")
    fig5.tight_layout()
    st.pyplot(fig5)

# --------------------------- Coefficients ----------------------------------
if show_coeffs:
    st.subheader("Model Coefficients (direction & strength)")
    names = (["intercept"] if fit_intercept else []) + ["distance_miles", "weather_bad", "driver_experience_years"]
    coeffs = w.ravel()
    coef_df = pd.DataFrame({"feature": names, "weight": coeffs})
    st.dataframe(coef_df)

# ----------------------------- Scenario Tool -------------------------------
st.markdown("---")
st.subheader("Scenario Estimation")
user_distance = st.slider("Enter distance (miles)", 50, 600, 250)
user_weather = st.selectbox("Weather", ["clear (0)", "bad (1)"])
user_weather_val = 0 if user_weather.startswith("clear") else 1
user_exp = st.slider("Driver experience (years)", 0.0, float(exp_max), float(max(1.0, exp_max/2.0)), 0.5)
X_user = np.array([[user_distance, user_weather_val, user_exp]])
Xb_user = (np.hstack([np.ones((1,1)), X_user]) if fit_intercept else X_user)
p_user = float(sigmoid(Xb_user @ w))
st.write(f"Predicted **p(On-Time)**: **{p_user:.2f}**")
st.write(f"Action at threshold {threshold:.2f}: **{'OK' if p_user>=threshold else 'Pre-alert / Mitigate'}**")

# ------------------- What Measures Mean & How to Use Them -------------------
st.markdown("""
## What These Measures Mean — and How to Use Them

**Accuracy**  
> Share of **all** predictions that were correct.  
Use when classes are balanced and the cost of FP and FN is similar.

**Precision (for On-Time=1 alerts)**  
> Of the loads we **flagged as on-time**, how many actually were?  
High precision = **few false alarms**. Useful when alerts trigger costly actions.

**Recall (Sensitivity, for On-Time=1)**  
> Of all the **actual on-time** loads, how many did we **correctly** mark on-time?  
If you flip the class focus (Late=1), recall then measures how many **late** loads you **caught**.  
High recall = **few misses**. Useful when missing a problem is very costly.

**F1-Score**  
> Harmonic mean of precision and recall.  
Best when classes are **imbalanced** or when you must balance **false alarms vs. misses**.

**Log-Loss (Negative Log-Likelihood)**  
> Penalizes **confidently wrong** predictions more than mildly wrong ones.  
Minimizing log-loss is equivalent to **Maximum Likelihood Estimation (MLE)**.

### Threshold Tuning (Business)
- Slide the threshold to trade off **precision vs. recall**.
- Use the **Cost of Errors** box to align with business impact:
  - **False Positive (FP)**: unnecessary alert / customer worry / ops work.
  - **False Negative (FN)**: missed late load / service failure / penalties.
- Choose the threshold that **minimizes expected cost** and meets SLA targets.
""")

