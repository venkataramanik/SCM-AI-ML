
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# LINEAR REGRESSION – BUSINESS DEMO (distance vs. delivery time)
# --------------------------------------------------------------------

st.title("Operational Insight with Linear Regression")

st.markdown("""
### 1. Business Question  
How does delivery time scale with route distance?  
I want a simple, transparent relationship that dispatch, pricing, and planning teams can interpret and apply.

### 2. Modeling Approach  
Linear Regression gives a first-order view. It estimates:
- a baseline handling time (loading/unloading, paperwork),
- an incremental travel rate per mile.

This is the baseline model I build first before moving to more complex, non-linear approaches.
""")

# -------------------------- Sidebar controls ------------------------
st.sidebar.header("Simulation Controls")
n = st.sidebar.slider("Number of trips", 50, 500, 100, 50)
noise_level = st.sidebar.slider("Data variation (noise)", 0.0, 5.0, 2.0, 0.5)
lr = st.sidebar.slider("Learning rate (Gradient Descent)", 0.000001, 0.00001, 0.000002, format="%.8f")
iterations = st.sidebar.slider("Iterations (Gradient Descent)", 500, 5000, 2000, 500)
model_choice = st.sidebar.multiselect("Show models", ["Least Squares", "Gradient Descent"], default=["Least Squares", "Gradient Descent"])
show_data = st.sidebar.checkbox("Show sample data", value=False)
show_errors_to_line = st.sidebar.checkbox("Show errors to line (residual whiskers)", value=False)

# ------------------------ Generate sample data ----------------------
np.random.seed(42)
distance = np.random.uniform(50, 500, n)  # miles
base_time_true = 1.5                       # hours (fixed handling)
rate_per_mile_true = 0.07                  # hours per mile
noise = np.random.normal(0, noise_level, n)
delivery_time = base_time_true + rate_per_mile_true * distance + noise

data = pd.DataFrame({"Distance (miles)": distance, "Delivery Time (hours)": delivery_time})
if show_data:
    st.subheader("Sample Data")
    st.dataframe(data.head(10))

# --------------------- Least Squares (analytical) -------------------
X = np.c_[np.ones((n, 1)), distance]  # intercept column
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(delivery_time)
a_lse, b_lse = float(beta[0]), float(beta[1])

# --------------------- Gradient Descent (iterative) -----------------
a_gd, b_gd = 0.0, 0.0
for _ in range(iterations):
    pred = a_gd + b_gd * distance
    error = delivery_time - pred
    a_grad = -2 * np.sum(error) / n
    b_grad = -2 * np.sum(distance * error) / n
    a_gd -= lr * a_grad
    b_gd -= lr * b_grad

# ------------------------------ Plot --------------------------------
x_line = np.linspace(50, 500, 100)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(distance, delivery_time, color="steelblue", label="Actual Trips", alpha=0.6)

if "Least Squares" in model_choice:
    ax.plot(x_line, a_lse + b_lse * x_line, "r-", label="Least Squares")
if "Gradient Descent" in model_choice:
    ax.plot(x_line, a_gd + b_gd * x_line, "g--", label="Gradient Descent")

# optional: show residual lines to the chosen reference line (prefer LSE if present)
if show_errors_to_line:
    if "Least Squares" in model_choice:
        ref_a, ref_b = a_lse, b_lse
    elif "Gradient Descent" in model_choice:
        ref_a, ref_b = a_gd, b_gd
    else:
        ref_a, ref_b = a_lse, b_lse  # default
    # draw thin vertical whiskers from each point to the fitted line
    y_on_line = ref_a + ref_b * distance
    for x, y, yhat in zip(distance, delivery_time, y_on_line):
        ax.plot([x, x], [y, yhat], color="gray", alpha=0.25, linewidth=0.8)

ax.set_xlabel("Distance (miles)")
ax.set_ylabel("Delivery Time (hours)")
ax.set_title("Fitted Relationship")
ax.legend()
fig.tight_layout()
st.pyplot(fig)

# ------------------------- Coefficients ------------------------------
st.subheader("Model Coefficients")
if "Least Squares" in model_choice:
    st.write(f"Least Squares: base = {a_lse:.3f} hrs, rate = {b_lse:.3f} hrs/mile")
if "Gradient Descent" in model_choice:
    st.write(f"Gradient Descent: base = {a_gd:.3f} hrs, rate = {b_gd:.3f} hrs/mile")

# -------------------------- Scenario tool ---------------------------
st.markdown("---")
st.subheader("Scenario Estimation")
user_distance = st.slider("Enter trip distance (miles)", 50, 500, 250)
pred_time = a_lse + b_lse * user_distance
st.write(f"Estimated delivery time: **{pred_time:.2f} hours**")

# ----------------------- Plain-English error story -------------------
st.markdown("""
### Understanding Model Error  
Each point is a trip. The fitted line won’t pass through every point.  
The **error** is the vertical gap between a point (actual time) and the line (predicted time).  
Both methods aim to make the total of those squared gaps as small as possible:

- **Least Squares** solves for the exact line that minimizes the total squared gaps in one calculation.  
- **Gradient Descent** starts with a rough line and nudges it in small steps, each time checking if the total error got smaller, until it can’t improve further.

Either way, the result should be a stable, explainable relationship that teams can use.
""")

# -------------------- Interpretation and takeaway --------------------
st.markdown(f"""
### 3. Interpretation  
- Intercept (fixed overhead): about **{a_lse:.1f} hours** per trip.  
- Slope (incremental rate): about **{b_lse:.2f} hours per mile**.  
- Even with noise, the relationship stays stable and easy to interpret.

### 4. Business Takeaway  
Use this baseline to forecast delivery time for new routes, plan driver hours, and evaluate the impact of distance on cost.  
If non-linear effects or interactions matter (traffic cadence, seasonality, stop density), graduate to polynomial or tree-based models; keep this linear view as the quick, auditable reference.
""")

# -------------------- Perspective: regression to the mean ------------
st.markdown("""
### Perspective: Regression to the Mean (useful caution)  
**Francis Galton** popularized “regression to the mean” after observing that very tall parents tend to have children closer to average height on the next measurement.  
The point: extreme outcomes are often followed by more ordinary ones simply because of natural variation.

In operations, the same caution applies: a week with a severe delay spike is often followed by a calmer week even if nothing major changed in the process.  
Don’t over-correct on one abnormal period. Look for persistence before shifting policy.

A related note from thinkers like **Taleb** and **Mandelbrot**: some environments have **fat tails** — rare but outsized events matter more than the average suggests.  
If your lanes or seasons produce occasional extreme delays, a pure linear average can be misleading.  
That’s where **percentile targets (P90 ETAs), quantile regression, stress tests,** or **buffering rules** help you plan for the tails rather than just the mean.
""")
