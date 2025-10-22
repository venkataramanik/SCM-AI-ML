import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score

# Setting a fixed seed for reproducibility across all functions
np.random.seed(42)

# --- A. Core Simulation Data (Simulate a trained model and predictions) ---
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
        'Trailer_Type': np.random.choice(['Reefer', 'Flatbed', 'Dry Van', 'Dry Van', np.nan], N_SAMPLES, p=[0.2, 0.1, 0.6, 0.05, 0.05]),
        'Target_Failure': np.random.choice([0, 1], N_SAMPLES, p=[0.9, 0.1])
    })
    
    # Introduce missing data for Imputation Demo
    df.loc[df.sample(frac=0.15).index, 'GPS_Lag_Hours'] = np.nan
    
    # Simple Model Training (for realistic prediction probabilities)
    X = df[['Mileage_Total', 'Last_Service_Days', 'Engine_Vibration_Avg', 'Oil_Pressure_Deviation']].fillna(0)
    y = df['Target_Failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    
    # Generate Prediction Probabilities (the model output)
    probabilities = model.predict_proba(X[['Mileage_Total', 'Last_Service_Days', 'Engine_Vibration_Avg', 'Oil_Pressure_Deviation']].fillna(0))[:, 1]
    df['Predicted_Probability'] = probabilities
    
    return df, X, y, X_test, y_test, model

df, X_all, y_all, X_test, y_test, model = load_and_simulate_data()

# --- Streamlit App Layout ---
st.title("From Prototype to Production: The MLOps Dashboard")
st.markdown("This dashboard demonstrates the five key technical features required to move a machine learning model into a reliable, business-focused production system.")
st.markdown("---")

# ==============================================================================
# FEATURE 1: Cost-Sensitive Threshold Optimization
# ==============================================================================
st.header("1. 💰 Cost-Sensitive Threshold Optimization")
st.markdown("**Why it matters:** Accuracy doesn't equal profit. This feature finds the prediction threshold that delivers the highest financial savings, not just the best technical score.")

col_cost_fn, col_cost_fp, col_cost_tp = st.columns(3)
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
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Net Savings = (TP * Gain) - (FP * Cost) - (FN * Loss)
        net_savings = (tp * cost_tp) - (fp * cost_fp) - (fn * cost_fn)
        savings.append(net_savings)
        
    results = pd.DataFrame({'Threshold': thresholds, 'Net_Savings': savings})
    
    # Calculate always-monitor scenario (Baseline)
    baseline_savings = (y_true.sum() * cost_tp) - ((len(y_true) - y_true.sum()) * cost_fp)
    
    # Calculate optimal threshold
    optimal_t = results.iloc[results['Net_Savings'].idxmax()]
    
    return results, optimal_t, baseline_savings

results, optimal_t, baseline_savings = calculate_net_savings(y_all, df['Predicted_Probability'], cost_tp, cost_fp, cost_fn)

fig_cost = px.line(results, x='Threshold', y='Net_Savings', title='Net Savings vs. Prediction Threshold')
fig_cost.add_vline(x=optimal_t['Threshold'], line_dash="dash", line_color="red", 
                   annotation_text=f"Optimal Threshold: {optimal_t['Threshold']:.2f}")
st.plotly_chart(fig_cost, use_container_width=True)

st.success(f"**Optimal Business Threshold:** **{optimal_t['Threshold']:.2f}** (Max Savings: **${optimal_t['Net_Savings']:,.0f}**)")
st.caption(f"Note: This optimal threshold is rarely 0.5 and is determined by business costs.")
st.markdown("---")


# ==============================================================================
# FEATURE 2: Model Explainability (SHAP/LIME concept using simple coefficients)
# NOTE: Using coefficients as a stand-in for complex SHAP values for simplicity
# ==============================================================================
st.header("2. 💡 Model Explainability (Why a Specific Truck Fails)")
st.markdown("**Why it matters:** Technicians need to know *why* the model flagged a truck, not just the score. This provides the root cause for compliance and action.")

truck_id = st.selectbox("Select Truck ID for Local Explanation:", df['Truck_ID'].sample(5, random_state=42).tolist())
truck_data = df[df['Truck_ID'] == truck_id].iloc[0]

# Generate pseudo-SHAP values based on model coefficients
coefficients = pd.Series(model.coef_[0], index=X_all.columns)
# Base value is the average failure log-odds
base_value = np.log(y_all.mean() / (1 - y_all.mean()))

# Calculate contribution (SHAP concept)
contributions = (truck_data[X_all.columns] * coefficients).sort_values(ascending=False)

# Prepare data for Waterfall Chart concept
contributions_df = contributions.reset_index().rename(columns={'index': 'Feature', 0: 'Contribution'})
contributions_df = contributions_df[abs(contributions_df['Contribution']) > 0.01].head(5)

fig_exp = go.Figure(go.Waterfall(
    name="Local Explanation", orientation="v",
    base=base_value,
    y=contributions_df['Contribution'],
    textposition="outside",
    x=contributions_df['Feature'],
    connector={"line": {"color": "rgb(63, 63, 63)"}},
))
fig_exp.update_layout(
    title=f"Prediction Explanation for {truck_id} (Pred. Prob: {truck_data['Predicted_Probability']:.2f})",
    showlegend=False,
    yaxis_title="Feature Impact (Log-Odds)"
)

st.plotly_chart(fig_exp, use_container_width=True)
st.info(f"The model's prediction of **{truck_data['Predicted_Probability']:.2f}** failure probability for {truck_id} is mainly driven by: **{contributions_df.iloc[0]['Feature']}**.")
st.markdown("---")


# ==============================================================================
# FEATURE 3 & 4: Imputation Strategy Comparison & Skew Monitoring
# ==============================================================================
st.header("3. 🎛️ Data Imputation Strategy & Skew Monitoring")
st.markdown("**Why it matters:** We justify our data cleaning. We compare simple fills (Median) vs. smart fills (KNN) to show we preserve the true data pattern for stability.")

col_impute_viz, col_impute_desc = st.columns(2)

with col_impute_viz:
    st.subheader("Imputation Comparison: GPS Lag Hours")
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

with col_impute_desc:
    st.subheader("4. ⚠️ Training-Serving Skew Simulation")
    st.markdown("We simulate a drift in the **Engine Vibration** feature.")
    
    train_mean = X_all['Engine_Vibration_Avg'].mean()
    live_mean = st.slider("Simulate Live Engine Vibration Mean:", min_value=0.4, max_value=0.8, value=0.55, step=0.01)
    
    # Simple Drift Check (Difference metric - not true PSI, but conveys concept)
    drift_score = abs(live_mean - train_mean) / train_mean
    
    st.metric(label="Drift Score (vs. Training Data)", value=f"{drift_score:.2f}", delta=f"Base Mean: {train_mean:.2f}", delta_color="off")
    
    if drift_score > 0.15:
        st.error("🚨 ALERT: Feature Skew Detected! Data distribution has shifted by over 15%. Retraining may be required.")
    else:
        st.info("Data distribution is stable and consistent with training data.")

st.markdown("---")

# ==============================================================================
# FEATURE 5: MLOps Pipeline Status Visualization
# ==============================================================================
st.header("5. ⚙️ MLOps Pipeline (Industrialization & Risk Mitigation)")
st.markdown("**Why it matters:** This shows the system is automated, scalable, and safe, using **Google Vertex AI** concepts.")

st.subheader("Live Deployment Status (Vertex AI Endpoints)")

traffic_split = pd.DataFrame({
    'Stage': ['Training & Registry (V2.2)', 'Canary (V2.1)', 'Production (V2.0)'],
    'Traffic Share': [0, 5, 95],
    'Color': ['#3399FF', '#FFCC00', '#00CC66'] # Blue, Yellow, Green
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
    traffic_split.loc[traffic_split['Stage'] == 'Canary (V2.1)', 'Traffic Share'] = 10 # Increase traffic

st.markdown(f'<div style="background-color: lightgray; padding: 10px; border-radius: 5px;">**Prod Model V2.0 AUC:** {prod_auc:.2f} | **Approval Threshold:** {auc_gate:.2f}</div>', unsafe_allow_html=True)
st.markdown(f'<p style="color:{status_color}; font-weight:bold;">{status_text}</p>', unsafe_allow_html=True)


fig_pipeline = px.bar(traffic_split.iloc[1:], x='Stage', y='Traffic Share', color='Stage',
                      color_discrete_map={'Canary (V2.1)': '#FFCC00', 'Production (V2.0)': '#00CC66'},
                      title="Canary Deployment Traffic Split")
fig_pipeline.update_layout(yaxis_title="Live Service Traffic (%)", height=300)
st.plotly_chart(fig_pipeline, use_container_width=True)

st.caption("This visualization shows the safe, automated deployment process (Canary) and the automated gate checks (Model Registry AUC threshold) before a model impacts $100\%$ of the fleet.")
st.markdown("---")

### The cost-sensitive learning approach is highly relevant for predictive maintenance models, which must balance the high cost of a missed failure against the lower cost of unnecessary service.

[Cost-Optimised Machine Learning Model Comparison for Predictive Maintenance - MDPI](https://www.mdpi.com/2079-9292/14/12/2497)
