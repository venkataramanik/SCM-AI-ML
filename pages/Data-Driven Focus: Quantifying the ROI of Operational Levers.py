import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SET UP THE STREAMLIT PAGE ---
st.set_page_config(
    page_title="Dynamic Levers: Decision Tree Simulation",
    layout="wide"
)

st.title("Dynamic Levers Dashboard: Continuous Improvement in Logistics")
st.markdown("---")


# --- 2. CONTROL SLIDERS (USER INPUT) ---
st.header("1. Input Controls: Simulate Business Conditions")
st.markdown("Adjust these sliders to simulate different operational environments and see how the model's focus changes.")

col_a, col_b = st.columns(2)

with col_a:
    # Control for the size of the historical data
    N_SAMPLES = st.slider(
        "Historical Data Volume (Total Shipments)", 
        min_value=100, max_value=1000, value=300, step=50
    )
    # Control for the impact of the highest lever (Route Risk)
    ROUTE_IMPACT = st.slider(
        "Route Risk Volatility (High value = more critical routes)", 
        min_value=0.5, max_value=1.5, value=1.0, step=0.1
    )

with col_b:
    # Control for the impact of a medium lever (Driver Behavior)
    DRIVER_IMPACT = st.slider(
        "Driver Behavior Influence (High value = more reliance on driver skill)", 
        min_value=0.2, max_value=1.0, value=0.5, step=0.1
    )
    # Control for the base level of noise/randomness
    RANDOM_SEED = st.number_input(
        "Random Seed (Change this to re-run the simulation)", 
        min_value=1, value=int(np.random.randint(1, 1000)), step=1
    )
st.markdown("---")


# --- 3. CREATE MOCK DATA WITH DYNAMIC INFLUENCE ---
@st.cache_data(show_spinner="Generating Dynamic Logistics Data...")
def create_logistics_data(n_samples, route_impact, driver_impact, seed):
    """Generates synthetic data with influence based on slider values."""
    np.random.seed(seed)
    
    # Features (High Influence Factors are dynamically weighted)
    route_risk_level = np.random.uniform(1, 10, n_samples)
    traffic_congestion_score = np.random.uniform(1, 10, n_samples)
    driver_behavior_score = np.random.uniform(5, 10, n_samples)
    time_of_day_risk = np.random.choice([1, 5, 8], n_samples, p=[0.5, 0.3, 0.2])
    customs_doc_score = np.random.uniform(7, 10, n_samples)
    vehicle_age_years = np.random.uniform(1, 10, n_samples)
    
    # Dynamic Influence Logic: Coefficients are multiplied by user sliders
    linear_score = (
        (-0.70 * route_impact) * route_risk_level          
        + (-0.60 * 1.0) * traffic_congestion_score          
        + (0.35 * driver_impact) * driver_behavior_score      
        - 0.40 * time_of_day_risk            
        + 0.10 * customs_doc_score          
        - 0.05 * vehicle_age_years          
        + 10 
    )
    prob_on_time = 1 / (1 + np.exp(-linear_score)) 
    y_on_time = (prob_on_time + np.random.normal(0, 0.05, n_samples) > 0.6).astype(int)

    data = pd.DataFrame({
        'Route Risk Level': route_risk_level.round(1),
        'Traffic Congestion Score': traffic_congestion_score.round(1),
        'Driver Behavior Score': driver_behavior_score.round(1),
        'Time of Day Risk': time_of_day_risk,
        'Customs Doc Score': customs_doc_score.round(1),
        'Vehicle Age (Years)': vehicle_age_years.round(1),
        'On-Time Delivery': y_on_time
    })
    return data, y_on_time

# Generate Data
df, y = create_logistics_data(N_SAMPLES, ROUTE_IMPACT, DRIVER_IMPACT, RANDOM_SEED)
X = df.drop(['On-Time Delivery'], axis=1)

# --- 4. MODEL TRAINING AND FEATURE IMPORTANCE EXTRACTION ---
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


# --- 5. DATASET PREVIEW ---
st.header("2. Sample of Generated Data")
st.markdown(f"*(Based on {N_SAMPLES} shipments)*")
st.dataframe(df.head(5), use_container_width=True)
st.markdown("---")


# --- 6. FEATURE IMPORTANCE DASHBOARD ---
st.header("3. Key Levers Dashboard: Importance Ranking")

if not importance_df.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.plasma(np.linspace(0, 1, len(importance_df))) # Changed color scheme
    ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    ax.set_xlabel("Relative Influence on Delivery Outcome (%)")
    ax.set_title("Operational Levers: Importance Ranking (Decision Tree Output)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
    st.pyplot(fig)
else:
    st.warning("Feature importance could not be calculated. Try increasing the historical data volume.")


# --- 7. BUSINESS EXPLANATION: DYNAMIC LEVERS ---
st.header("4. The Reality: Why Levers Change Over Time")

st.markdown(
    """
    As a leader, the key insight here is that the levers **are not static**—they change as your business environment and internal operations change.
    
    ### Continuous Improvement through Retraining
    
    1.  **New Risk Emerges (e.g., Port Strike):** If a **Port Strike** occurs, a new feature like *Port Congestion Index* will suddenly shoot up in importance, displacing an old leader like *Traffic Congestion*.
    2.  **Mitigation Works (e.g., New Software):** If you invest heavily in a new routing system to fix **Route Risk**, that factor's influence will *decrease* in future model runs because the problem is being solved. The model will then naturally identify the *next* biggest problem (e.g., *Traffic Congestion*) as the new primary lever.
    3.  **The Feedback Loop:** By regularly retraining the Decision Tree on new data, we get a fresh picture of the current reality. This ensures our operational focus and resource allocation are always targeting the **highest-leverage, unmitigated risks** in the system, driving true continuous improvement.
    """
)
st.markdown("---")
st.success("Use the sliders at the top to simulate these changes and watch the Key Levers ranking adapt, just as it would in a real-world, dynamic logistics network.")
