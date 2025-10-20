import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SET UP THE STREAMLIT PAGE ---
st.set_page_config(
    page_title="Data-Driven Focus: Quantifying the ROI of Operational Levers",
    layout="wide"
)

# --- 2. CREATE MOCK DATA WITH REALISTIC LOGISTICS FEATURES ---
@st.cache_data
def create_logistics_data(n_samples=250): # Increased sample size for better tree performance
    """Generates synthetic data using realistic logistics features."""
    np.random.seed(42)
    
    # Realistic Features (Scores 0-10 or specific ranges)
    # High Influence Factors
    route_risk_level = np.random.uniform(1, 10, n_samples)          # Complexity, historical delays, etc.
    traffic_congestion_score = np.random.uniform(1, 10, n_samples)  # Real-time traffic data
    
    # Medium Influence Factors
    driver_behavior_score = np.random.uniform(5, 10, n_samples)     # 10 is perfect driving, 5 is average
    time_of_day_risk = np.random.choice([1, 5, 8], n_samples, p=[0.5, 0.3, 0.2]) # 1=low (mid-day), 8=high (rush hour)
    
    # Low Influence Factors
    customs_doc_score = np.random.uniform(7, 10, n_samples)         # 10 is perfect documentation
    vehicle_age_years = np.random.uniform(1, 10, n_samples)         # Older vehicles have slightly higher risk
    
    # Define outcome probability based on realistic influence
    # Negative coefficients = higher feature value increases LATE probability (decreases ON-TIME prob)
    linear_score = (
        -0.70 * route_risk_level            # Highest negative impact
        -0.60 * traffic_congestion_score    # High negative impact
        + 0.35 * driver_behavior_score      # Positive impact (better score -> on-time)
        -0.40 * time_of_day_risk            # Medium negative impact
        + 0.10 * customs_doc_score          # Low positive impact
        -0.05 * vehicle_age_years           # Very low negative impact
        + 10 # Bias to center the probability
    )
    prob_on_time = 1 / (1 + np.exp(-linear_score)) 
    y_on_time = (prob_on_time + np.random.normal(0, 0.05, n_samples) > 0.6).astype(int) # 60% threshold

    data = pd.DataFrame({
        'Route Risk Level (1-10)': route_risk_level.round(1),
        'Traffic Congestion Score (1-10)': traffic_congestion_score.round(1),
        'Driver Behavior Score (5-10)': driver_behavior_score.round(1),
        'Time of Day Risk (1-8)': time_of_day_risk,
        'Customs Doc Score (7-10)': customs_doc_score.round(1),
        'Vehicle Age (Years)': vehicle_age_years.round(1),
        'On-Time Delivery': y_on_time
    })
    data['Delivery Status'] = data['On-Time Delivery'].replace({1: 'On-Time ✅', 0: 'Late ❌'})
    return data, y_on_time

df, y = create_logistics_data()
X = df.drop(['On-Time Delivery', 'Delivery Status'], axis=1)

# --- 3. MODEL TRAINING AND FEATURE IMPORTANCE EXTRACTION ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
dt_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42) 
dt_model.fit(X_train, y_train)

# Extract feature importance
feature_names = X.columns
importances = dt_model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# Filter out features with 0 importance, then sort
importance_df = importance_df[importance_df['Importance'] > 0].sort_values(by='Importance', ascending=True)


# --- 4. STREAMLIT DASHBOARD LAYOUT ---

st.title("Decision Tree: Identifying Key Business Levers from Real-World Data")
st.markdown("---")

# --- 5. DATASET PREVIEW ---
st.header("1. The Business Data Input (Sample)")
st.markdown("This historical data includes factors that influence delivery outcomes, from road conditions to driver performance.")
st.dataframe(df.head(10), use_container_width=True)
st.markdown("---")


# --- 6. EXPLANATION OF KEY LEVERS CONCEPT ---
st.header("2. Articulating the Concept: Finding the Key Levers")

st.markdown(
    """
    Decision Trees help leaders by finding the **Tipping Points** in business data. They answer: **Which operational factor causes the biggest drop in reliability?**
    
    ### The Business Logic: Measuring 'Influence'
    The Decision Tree algorithm measures influence by how effectively a factor can create **pure** groups of outcomes (e.g., a group that is $95\%$ On-Time).
    
    * **High Influence (Key Lever):** If the model finds that a rule based on **Route Risk** is the most powerful divider for the entire dataset, that feature is awarded the highest importance score. It's the primary **lever** you can pull to manage risk.
    * **Low Influence (Minor Factor):** If a factor like **Customs Doc Score** is only used deep in the flowchart for minor corrections, its score is low. Its variation doesn't strongly drive the overall Late/On-Time outcome.
    
    The chart below quantifies this influence, directing our resources to the highest-impact areas.
    """
)
st.markdown("---")


# --- 7. FEATURE IMPORTANCE DASHBOARD ---
st.header("3. Key Levers Dashboard: Importance Ranking")

# Create Bar Chart
if not importance_df.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    # Using more appropriate colors for data visualization
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    ax.set_xlabel("Relative Influence on Delivery Outcome (%)")
    ax.set_title("Operational Levers: Importance Ranking (Decision Tree Output)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
    st.pyplot(fig)
else:
    st.warning("Feature importance could not be calculated. Please check the data generation.")


# --- 8. BUSINESS ACTION ITEMS ---
if not importance_df.empty:
    st.subheader("Business Action Items: Allocating Resources Strategically")

    top_feature = importance_df.iloc[-1]['Feature']
    second_feature = importance_df.iloc[-2]['Feature']
    
    st.markdown(
        f"""
        1.  **Highest Priority (Max ROI):** The factor **{top_feature}** is the single greatest driver of success or failure. Our highest **capital investment**—in dynamic rerouting or preventative action plans—must target this area.
        2.  **Secondary Focus:** The next most impactful lever is **{second_feature}**. Optimizing resources here, such as real-time congestion alerts, will deliver the next largest **gain in reliability**.
        3.  **Efficiency Gain:** Factors with low importance (e.g., **Vehicle Age**) require only baseline monitoring. We can free up manager time and reduce operational reporting on these low-influence items.
        """
    )

st.markdown("---")
st.success("Decision Trees provide clear, quantifiable direction for maximizing operational control and reliability.")
