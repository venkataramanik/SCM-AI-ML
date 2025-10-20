"""
Linear Regression – Explanation and Demonstration
-------------------------------------------------
Linear regression is one of the simplest and most useful predictive models.
It assumes a roughly proportional relationship between variables – for example,
as distance increases, delivery time also increases.

We use it when we want an interpretable, explainable relationship between inputs
and outputs — to quantify how strongly one factor affects another.

Typical business applications:
- Forecasting sales, cost, or demand
- Understanding drivers of variation (e.g., how distance affects delivery time)
- Estimating trends like fuel usage vs. load, maintenance vs. mileage
- Providing baselines before moving to more complex models

Internally, it fits a straight line that minimizes total prediction error.
There are two ways to arrive at that line:
1. Least Squares Estimation (solves the optimal line directly)
2. Gradient Descent (starts with guesses and improves step by step)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Narrative section ---
st.title("Linear Regression – Concept and Demonstration")

st.markdown("""
Linear Regression assumes a simple proportional relationship between inputs and outputs.  
It helps us answer questions like *"How much does delivery time increase with distance?"*

It's transparent and quick — ideal for operational analytics or early-stage modeling.
""")

# --- Create sample data ---
np.random.seed(42)
n = 100
distance = np.random.uniform(50, 500, n)            # miles
base_time = 1.5                                     # fixed handling time (hours)
rate_per_mile = 0.07                                # hours per mile
noise = np.random.normal(0, 2, n)                   # real-world variation
delivery_time = base_time + rate_per_mile * distance + noise

# --- Least Squares Estimation ---
X = np.c_[np.ones((n, 1)), distance]                # add intercept column
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(delivery_time)
a_lse, b_lse = beta[0], beta[1]

# --- Gradient Descent (iterative optimization) ---
a_gd, b_gd = 0.0, 0.0
lr = 0.000001
iterations = 2000

for _ in range(iterations):
    pred = a_gd + b_gd * distance
    error = delivery_time - pred
    a_grad = -2 * np.sum(error) / n
    b_grad = -2 * np.sum(distance * error) / n
    a_gd -= lr * a_grad
    b_gd -= lr * b_grad

# --- Output summary ---
st.subheader("Model Results")
st.write(f"**Least Squares:** base time = {a_lse:.3f} hrs, rate = {b_lse:.3f} hrs/mile")
st.write(f"**Gradient Descent:** base time = {a_gd:.3f} hrs, rate = {b_gd:.3f} hrs/mile")

# --- Visualization ---
x_line = np.linspace(50, 500, 100)
plt.figure(figsize=(8,5))
plt.scatter(distance, delivery_time, color='blue', label='Actual Data')
plt.plot(x_line, a_lse + b_lse * x_line, 'r-', label='Least Squares')
plt.plot(x_line, a_gd + b_gd * x_line, 'g--', label='Gradient Descent')
plt.xlabel("Distance (miles)")
plt.ylabel("Delivery Time (hours)")
plt.title("Linear Regression: Delivery Time vs Distance")
plt.legend()
st.pyplot(plt)

# --- Interpretation ---
st.subheader("Interpretation and Business Context")
st.markdown(f"""
The model finds a simple linear relationship between distance and delivery time.

- Fixed overhead per trip ≈ **{a_lse:.1f} hours**  
  (represents loading/unloading or handling time)
- Incremental travel rate ≈ **{b_lse:.2f} hours per mile**  
  (how much time increases for every additional mile)

Both methods — Least Squares and Gradient Descent — converge on nearly the same
relationship, confirming that the pattern is stable and consistent.

This kind of model is often used as a baseline in logistics, finance, and operations
analytics before introducing more complex non-linear or machine learning methods.
""")
