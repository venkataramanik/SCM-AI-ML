import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Production MLOps Demo",
    layout="wide"
)
np.random.seed(42)

# --- A. Core Simulation Data ---
@st.cache_data
def load_and_simulate_data():
    N_SAMPLES = 1000
    df = pd.DataFrame({
        'Truck_ID': [f'TRK-{i+1:03d}' for i in range(N_SAMPLES)],
        'Mileage_Total': np.random.randint(50000, 500000, N_SAMPLES),
        'Last_Service_Days': np.random.randint(30, 700, N_SAMPLES),
        'Engine_Vibration_Avg': np.random.normal(0.5, 0.2, N_SAMPLES).clip(0.1, 1.0),
        'Oil_Pressure_Deviation': np.random.normal(0.1, 0.05, N_SAMPLES).clip(0.01, 0.3),
        'GPS_Lag_Hours': np.random.normal(0.5, 0.3, N_SAMPLES).clip(0, 1.5),
        'Target_Failure': np.random.choice([0, 1], N_SAMPLES, p=[0.9, 0.1])
    })
    
    # Introduce missing data for Imputation Demo
    df.loc[df.sample(frac=0.15).index, 'GPS_Lag_Hours'] = np.nan
    
    # Simple Model Training
    X = df[['Mileage_Total', 'Last_Service_Days', 'Engine_Vibration_Avg', 'Oil_Pressure_Deviation']].fillna(0)
    y = df['Target_Failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    
    # Generate Prediction Probabilities
    probabilities = model.predict_proba(X[['Mileage_Total', 'Last_Service_Days', 'Engine_Vibration_Avg', 'Oil_Pressure_Deviation']].fillna(0))[:, 1]
    df['Predicted_Probability'] = probabilities
    
    return df, X, y, model

df, X_all, y_all, model = load_and_simulate_data()

# --- Streamlit App Layout ---
st.title("The Production MLOps & Business Impact Dashboard")
st.markdown("This demo highlights the **business value (ROI)** and **operational resilience (MLOps)** of a Predictive Maintenance Model.")
st.markdown("---")

# ==============================================================================
# FEATURE 1: Cost-Sensitive Threshold Optimization
# ==============================================================================
st.header("1. Cost-Sensitive Threshold Optimization: Maximizing ROI")
st.markdown("We tune the model's decision threshold based on financial costs, ensuring maximum **net savings**, not just technical accuracy.")

col_cost_tp, col_cost_fp, col_cost_fn = st.columns(3)
with col_cost_tp:
    cost_tp = st.number_input("Cost Avoided (True Positive - Correct Service)", value=15000, min_value=100)
with col_cost_fp:
    cost_fp = st.number_input("Cost Incurred (False Positive - Unnecessary Service)", value=500, min_value=1)
with col_cost_fn:
    cost_fn = st.number_input("Catastrophic Cost (False Negative - Missed Breakdown)", value=20000, min_value=1000)

@st.cache_data
def calculate_net_savings(y_true, y_prob, cost_tp, cost_fp, cost_fn):
    thresholds = np.linspace(0.01, 0.99, 100)
    savings = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        except ValueError:
            # Fallback for extreme threshold cases
            tn, fp, fn, tp = (0, 0, 0, 0) 
        
        # Net Savings = (TP * Gain) - (FP * Cost) - (FN * Loss)
        net_savings = (tp * cost_tp) - (fp * cost_fp) - (fn * cost_fn)
        savings.append(net_savings)
        
    results = pd.DataFrame({'Threshold': thresholds, 'Net_Savings': savings})
    optimal_t = results.iloc[results['Net_Savings'].idxmax()]
    
    return results, optimal_t

results, optimal_t = calculate_net_savings(y_all, df['Predicted_Probability'], cost_tp, cost_fp, cost_fn)

fig_cost = px.line(results, x='Threshold', y='Net_Savings', title='Net Savings vs. Prediction Threshold')
fig_cost.add_vline(x=optimal_t['Threshold'], line_dash="dash", line_color="red", 
                   annotation_text=f"Optimal Threshold: {optimal_t['Threshold']:.2f}",
                   annotation_position="bottom right")
st.plotly_chart(fig_cost, use_container_width=True)

st.success(f"**Optimal Business Threshold:** **{optimal_t['Threshold']:.2f}** (Max Savings: **${optimal_t['Net_Savings']:,.0f}**)")
st.markdown("---")


# ==============================================================================
# FEATURE 2: Model Explainability
# ==============================================================================
st.header("2. Model Explainability: Building Trust and Action")
st.markdown("We use **Feature Importance (SHAP concept)** to show *why* a specific truck was flagged, providing **actionable insights** for technicians.")

truck_id = st.selectbox("Select Truck ID for Local Explanation:", df['Truck_ID'].sample(5, random_state=42).tolist())
truck_data = df[df['Truck_ID'] == truck_id].iloc[0]

# Generate pseudo-SHAP values based on model coefficients
coefficients = pd.Series(model.coef_[0], index=X_all.columns)
base_value = np.log(y_all.mean() / (1 - y_all.mean()))

contributions = (truck_data[X_all.columns] * coefficients).sort_values(ascending=False)

# Prepare data for Waterfall Chart concept
contributions_df = contributions.reset_index().rename(columns={'index': 'Feature', 0: 'Contribution'})
contributions_df = contributions_df[abs(contributions_df['Contribution']) > 0.01].head(5)

fig_exp = go.Figure(go.Waterfall(
    name="Local Explanation", orientation="v",
    base=base_value,
    y=contributions_df['Contribution'],
    x=contributions_df['Feature'],
    textposition="outside",
    connector={"line": {"color": "rgb(63, 63, 63)"}},
))
fig_exp.update_layout(
    title=f"Prediction Explanation for {truck_id} (Pred. Prob: {truck_data['Predicted_Probability']:.2f})",
    showlegend=False,
    yaxis_title="Feature Impact (Log-Odds)",
    height=400
)

st.plotly_chart(fig_exp, use_container_width=True)
st.info(f"The high failure prediction is mainly driven by: **{contributions_df.iloc[0]['Feature']}**.")
st.markdown("---")


# ==============================================================================
# FEATURE 3 & 4: Imputation Strategy & Skew Monitoring
# ==============================================================================
st.header("3. Data Integrity and MLOps Monitoring")
st.markdown("We address data gaps with smart imputation and mitigate **concept drift** with automated monitoring.")

col_impute_viz, col_skew_desc = st.columns(2)

with col_impute_viz:
    st.subheader("Data Imputation Strategy: GPS Lag")
    st.caption("Comparing simple Median fill vs. advanced KNN (which preserves distribution).")
    
    # Imputation: KNN (Smart fill)
    knn_imputer = KNNImputer(n_neighbors=5)
    X_impute_knn = knn_imputer.fit_transform(df[['GPS_Lag_Hours', 'Mileage_Total']])
    df['GPS_Lag_KNN'] = X_impute_knn[:, 0]
    
    # Imputation: Median (Simple fill)
    median_imputer = SimpleImputer(strategy='median')
    df['GPS_Lag_Median'] = median_imputer.fit_transform(df[['GPS_Lag_Hours']])
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=df['GPS_Lag_KNN'], name='KNN Imputation', opacity=0.7))
    fig_hist.add_trace(go.Histogram(x=df['GPS_Lag_Median'], name='Median Imputation', opacity=0.7))
    fig_hist.update_layout(barmode='overlay', title='Distribution of GPS Lag (Hours)', height=300)
    st.plotly_chart(fig_hist, use_container_width=True)

with col_skew_desc:
    st.subheader("Training-Serving Skew Simulation (Vertex AI Monitoring)")
    st.markdown("Checks if the live input data distribution has shifted from the training data (a key failure point).")
    
    train_mean = X_all['Engine_Vibration_Avg'].mean()
    live_mean = st.slider("Simulate Live Engine Vibration Mean:", min_value=0.4, max_value=0.8, value=0.55, step=0.01)
    
    # Simple Drift Check
    drift_score = abs(live_mean - train_mean) / train_mean
    
    st.metric(label="Feature Drift Score (PSI Concept)", value=f"{drift_score:.2f}", delta=f"Base Mean: {train_mean:.2f}", delta_color="off")
    
    if drift_score > 0.15:
        st.error("ALERT: Feature Skew Detected! Data distribution has shifted. Automated retraining is triggered.")
    else:
        st.info("Data distribution is stable and consistent.")

st.markdown("---")

# ==============================================================================
# FEATURE 5: MLOps Pipeline Status Visualization
# ==============================================================================
st.header("4. MLOps Pipeline Status: Safe Deployment")
st.markdown("We use a **Canary Deployment** strategy (Vertex AI Traffic Split) to roll out new models with minimal risk.")

st.subheader("Automated Model Registry & Canary Rollout")

traffic_split = pd.DataFrame({
    'Stage': ['Production (V2.0)', 'Canary (V2.1)', 'Training & Registry (V2.2)'],
    'Traffic Share': [95, 5, 0],
    'Color': ['#00CC66', '#FFCC00', '#3399FF']
})

# Simulate a check gate
prod_auc = 0.82
candidate_auc = st.slider("Candidate Model V2.2 AUC:", 0.70, 0.95, 0.85, 0.01)
auc_gate = 0.83

if candidate_auc < auc_gate:
    status_text = f"**REGISTRY GATE:** Candidate V2.2 **BLOCKED** (AUC {candidate_auc:.2f} < Threshold {auc_gate:.2f})."
    status_color = 'red'
    
else:
    status_text = f"**REGISTRY GATE:** Candidate V2.2 **APPROVED** (AUC {candidate_auc:.2f} > Threshold {auc_gate:.2f}). Ready for Canary."
    status_color = 'green'
    traffic_split.loc[traffic_split['Stage'] == 'Canary (V2.1)', 'Traffic Share'] = 10 # Increase traffic to canary if approved

# Removed custom HTML styling for maximum safety and used st.markdown with bold/italics
st.markdown(f"**Prod Model V2.0 AUC:** {prod_auc:.2f} | **Approval Threshold:** {auc_gate:.2f}")
st.markdown(f"*{status_text}*")


fig_pipeline = px.bar(traffic_split.iloc[0:2], x='Stage', y='Traffic Share', color='Stage',
                      color_discrete_map={'Canary (V2.1)': '#FFCC00', 'Production (V2.0)': '#00CC66'},
                      title="Live Endpoint Traffic Split")
fig_pipeline.update_layout(yaxis_title="Live Service Traffic (%)", height=300)
st.plotly_chart(fig_pipeline, use_container_width=True)

st.caption("The Canary split (Vertex AI Traffic Split) ensures safe, low-risk deployment.")
