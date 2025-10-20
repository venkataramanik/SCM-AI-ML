import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SET UP THE STREAMLIT PAGE ---
st.set_page_config(
    page_title="Delivery Predictor: Logistic Regression",
    layout="wide"
)

# --- 2. HEADER AND NON-TECHNICAL EXPLANATION ---
st.title("The Power of Probability: Predicting On-Time Delivery with Logistic Regression")
st.markdown("---")

st.header("What is Logistic Regression?")
st.markdown(
    """
    Think of Logistic Regression not as guessing the future, but as calculating the **likelihood** (probability) of a specific event—in our case, an **On-Time Delivery**.

    * It takes in various factors (**features**) like a driver's historical performance, route complexity, or weather forecasts.
    * It then outputs a **clear probability** between 0% and 100% that the delivery will be on time.

    **Our Goal:** Move beyond simple 'Yes/No' answers to understand the *certainty* behind our delivery schedules.
    """
)

st.markdown("---")

# --- 3. CREATE MOCK DATA FOR A LOGISTICS EXAMPLE ---
def create_logistics_data(n_samples=150):
    """Generates synthetic data for on-time delivery prediction."""
    np.random.seed(42)
    # Feature 1: Route Difficulty (e.g., historical delays, traffic-prone) - Higher is harder
    route_difficulty = np.random.uniform(1, 10, n_samples)
    # Feature 2: Weather Impact (e.g., rain, snow, heat) - Higher is worse
    weather_impact = np.random.uniform(1, 10, n_samples)

    # Simplified probability model (the 'core' of our logistic thinking)
    linear_score = -0.6 * route_difficulty - 0.4 * weather_impact + 8
    prob_on_time = 1 / (1 + np.exp(-linear_score)) # Sigmoid function

    # Classify as 1 (On-Time) or 0 (Late) based on a probability threshold (with some noise)
    y_on_time = (prob_on_time + np.random.normal(0, 0.1, n_samples) > 0.52).astype(int) 

    data = pd.DataFrame({
        'Route Difficulty (1-10)': route_difficulty.round(1),
        'Weather Impact (1-10)': weather_impact.round(1),
        'On-Time Delivery': y_on_time
    })
    
    # Replace 0/1 with business terms for clarity
    data['Delivery Status'] = data['On-Time Delivery'].replace({1: 'On-Time ✅', 0: 'Late ❌'})
    return data, y_on_time

df, y = create_logistics_data()
X = df[['Route Difficulty (1-10)', 'Weather Impact (1-10)']]

# --- 4. MODEL TRAINING AND DISPLAY ---

# Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)


# --- 5. LOG LOSS AND MLE SECTION ---

st.header("1. The Engine Room: Maximum Likelihood and Log Loss")

st.subheader("Maximum Likelihood Estimation (MLE)")
st.markdown(
    """
    How does the model *learn*? It uses a principle called **Maximum Likelihood Estimation (MLE)**.
    
    * **Analogy:** Imagine we have two weather predictors. Predictor A says a storm is **95%** likely; Predictor B says it's **55%** likely. If the storm *actually happens*, Predictor A is the one whose prediction was **most likely** to be true.
    * **Model Goal:** Logistic Regression searches for the set of weights (for Route Difficulty, Weather Impact, etc.) that makes the actual historical outcomes (On-Time or Late) **most likely** to have occurred.
    """
)

st.subheader("Log Loss: The Model's Scorecard")
st.markdown(
    """
    To achieve MLE, the model minimizes a metric called **Log Loss** (or Logistic Loss). Log Loss is simply the negative of the likelihood.
    
    * **Goal:** The closer the predicted probability is to the actual outcome, the **lower** the Log Loss.
    * **Penalty:** Log Loss heavily **penalizes** predictions that are confident and wrong. For example, if a delivery was **Late (0)** but the model confidently predicted **99% On-Time**, the Log Loss would be extremely high.
    """
)

# Plotting the Log Loss Function
probs = np.linspace(0.01, 0.99, 100)
# Loss when y=1 (True outcome is On-Time)
loss_y1 = -np.log(probs)
# Loss when y=0 (True outcome is Late)
loss_y0 = -np.log(1 - probs)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(probs, loss_y1, label='Actual Outcome: On-Time (y=1)', color='blue')
ax.plot(probs, loss_y0, label='Actual Outcome: Late (y=0)', color='red')
ax.set_title("Log Loss (Cost) vs. Predicted Probability")
ax.set_xlabel("Predicted Probability of On-Time (p)")
ax.set_ylabel("Loss")
ax.set_ylim(0, 5)
ax.axhline(0, color='gray', linestyle='--')
ax.legend()
st.pyplot(fig)

st.markdown(
    """
    * **Blue Line (Actual On-Time):** Loss approaches zero as predicted probability approaches 1 (perfect prediction). It spikes if we predict 0 when the outcome was 1.
    * **Red Line (Actual Late):** Loss approaches zero as predicted probability approaches 0 (perfect prediction). It spikes if we predict 1 when the outcome was 0.

    The model constantly tweaks its parameters to find the minimum point across all data points, achieving the best possible set of probability estimates.
    """
)

st.markdown("---")

# --- 6. MODEL PERFORMANCE METRICS ---
st.header("2. Model Evaluation: Key Performance Metrics")

col_data, col_metrics = st.columns(2)

with col_data:
    st.subheader("Sample of Training Data")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("This historical data teaches the model the relationship between inputs and outcomes.")
    
with col_metrics:
    st.subheader("Performance on Test Data")
    
    col_met_values = st.columns(4)
    col_met_values[0].metric(label="Overall Accuracy", value=f"{accuracy * 100:.2f}%")
    col_met_values[1].metric(label="Precision", value=f"{precision * 100:.2f}%")
    col_met_values[2].metric(label="Recall", value=f"{recall * 100:.2f}%")
    col_met_values[3].metric(label="F1-Score", value=f"{f1 * 100:.2f}%")


st.markdown("---")

# --- 7. METRICS EXPLANATION (Business Focus) ---
st.header("3. Interpreting the Metrics for Business Decisions")
st.markdown("We focus on **On-Time Delivery (1)** as the 'Positive' event we want to predict.")

col_biz_metrics = st.columns(4)

with col_biz_metrics[0]:
    st.subheader("Accuracy")
    st.markdown(
        """
        **What it is:** The percentage of *all* predictions (On-Time and Late) that were correct.
        
        **Business Use:** General measure of how often our model is right overall.
        """
    )
with col_biz_metrics[1]:
    st.subheader("Precision")
    st.markdown(
        """
        **What it is:** When the model *predicts* **On-Time**, how often is it actually On-Time?
        
        **Business Use:** Avoids false confidence. Low Precision means we're over-promising to customers.
        """
    )

with col_biz_metrics[2]:
    st.subheader("Recall")
    st.markdown(
        """
        **What it is:** Of all runs that *actually* finished **On-Time**, how many did the model correctly identify?
        
        **Business Use:** Avoids missed opportunities. Low Recall means we're classifying too many successful runs as late risks.
        """
    )

with col_biz_metrics[3]:
    st.subheader("F1-Score")
    st.markdown(
        """
        **What it is:** Balances Precision and Recall. Our best single measure for a robust prediction capability.
        
        **Business Use:** High F1 means we're both reliable and comprehensive in identifying successful deliveries.
        """
    )

st.markdown("---")

# --- 8. INTERACTIVE DEMO AND INTERPRETATION ---
st.header("4. Interactive Scenario: Probability vs. Prediction")
st.markdown(
    """
    Use the sliders below to see how changes in route and weather impact the **probability**
    of an on-time delivery.
    """
)

# User inputs
col_input1, col_input2 = st.columns(2)
new_route_difficulty = col_input1.slider("Set New Route Difficulty (1=Easy, 10=Hard)", 1.0, 10.0, 5.0)
new_weather_impact = col_input2.slider("Set New Weather Impact (1=Clear, 10=Severe)", 1.0, 10.0, 5.0)

# Prepare new input for the model
new_X = pd.DataFrame({
    'Route Difficulty (1-10)': [new_route_difficulty],
    'Weather Impact (1-10)': [new_weather_impact]
})

# Get probability and prediction
probability_on_time = model.predict_proba(new_X)[:, 1][0] # Probability for class 1 (On-Time)
final_prediction = model.predict(new_X)[0]
prediction_text = 'On-Time ✅' if final_prediction == 1 else 'Late ❌'

# Display results
st.subheader(f"Results for Scenario: **Route {new_route_difficulty:.1f}**, **Weather {new_weather_impact:.1f}**")

col_result1, col_result2 = st.columns(2)

with col_result1:
    st.metric(
        label="Calculated On-Time Probability",
        value=f"{probability_on_time * 100:.1f}%",
        delta="The likelihood this specific run will be successful."
    )

with col_result2:
    st.metric(
        label="Model's Final Prediction (Threshold > 50%)",
        value=prediction_text,
        delta="What the model predicts based on the 50% cut-off."
    )

st.markdown("""
### **Key Takeaway for Business Leaders (Interpretation):**
The two numbers above are why Logistic Regression is so powerful.

* If the probability is **51%** and the prediction is **On-Time**, we know it's a tight call. We should assign a backup driver or pre-warn the client.
* If the probability is **99%** and the prediction is **On-Time**, we can be highly confident and optimize resources elsewhere.

This is the shift from simple classification to **quantifying risk and certainty**.
""")

# --- 9. DECISION BOUNDARY VISUALIZATION ---
st.markdown("---")
st.subheader("5. Visualizing the 'Decision Line'")

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    x='Route Difficulty (1-10)',
    y='Weather Impact (1-10)',
    hue='Delivery Status',
    style='Delivery Status',
    data=df,
    palette={'On-Time ✅': 'blue', 'Late ❌': 'red'},
    s=100,
    ax=ax
)

# Plot the decision boundary (where P = 50%)
w0, w1, w2 = model.intercept_[0], model.coef_[0][0], model.coef_[0][1]
x_min, x_max = X['Route Difficulty (1-10)'].min(), X['Route Difficulty (1-10)'].max()
line_x = np.array([x_min, x_max])
line_y = (-w0 - w1 * line_x) / w2

ax.plot(line_x, line_y, color='k', linestyle='--', label='50% Probability Line')
ax.set_title("How the Model Separates On-Time vs. Late Deliveries")
ax.legend(title='Outcome')

st.pyplot(fig)

st.markdown(
    """
    The dashed black line represents the **50% probability threshold**—our model's line in the sand.
    * Any delivery scenario falling on the **blue side** is predicted **On-Time**.
    * Any scenario falling on the **red side** is predicted **Late**.

    This visualization shows that Logistic Regression finds the *best straight line* to separate two groups, providing a clear, **interpretable** basis for every decision it makes.
    """
)

