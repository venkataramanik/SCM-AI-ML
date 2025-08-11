import streamlit as st
import pandas as pd
from datetime import datetime
import random
import plotly.express as px

# --- Configuration ---
st.set_page_config(
    page_title="SCM C-Suite Simulation: AI/ML & Security",
    page_icon="👑",
    layout="wide",
)

# --- State Initialization ---
if 'simulation_log' not in st.session_state:
    st.session_state.simulation_log = []
if 'csuite_metrics' not in st.session_state:
    st.session_state.csuite_metrics = {
        'days': [0],
        'profitability': [1000],
        'operational_efficiency': [85],
        'customer_satisfaction': [90],
    }
if 'day' not in st.session_state:
    st.session_state.day = 0

# --- Constants ---
BASELINE_PROFIT = 100
BASELINE_OP_EFFICIENCY = 80
BASELINE_CUST_SAT = 85
AI_IMPACT_PROFIT_MIN = 20
AI_IMPACT_PROFIT_MAX = 50
AI_IMPACT_OP_EFFICIENCY_MIN = 5
AI_IMPACT_OP_EFFICIENCY_MAX = 10
AI_IMPACT_CUST_SAT_MIN = 3
AI_IMPACT_CUST_SAT_MAX = 8

# --- Helper Functions ---
def add_log(persona, message, level='info'):
    """Adds a timestamped message to the simulation log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.simulation_log.append({
        'timestamp': timestamp,
        'persona': persona,
        'message': message,
        'level': level,
    })

def advance_day(profit_change, efficiency_change, satisfaction_change, event_desc):
    """Advances the simulation by one day with new metrics."""
    st.session_state.day += 1
    current_profit = st.session_state.csuite_metrics['profitability'][-1] + profit_change
    current_efficiency = st.session_state.csuite_metrics['operational_efficiency'][-1] + efficiency_change
    current_satisfaction = st.session_state.csuite_metrics['customer_satisfaction'][-1] + satisfaction_change

    st.session_state.csuite_metrics['days'].append(st.session_state.day)
    st.session_state.csuite_metrics['profitability'].append(max(0, current_profit))
    st.session_state.csuite_metrics['operational_efficiency'].append(min(100, max(0, current_efficiency)))
    st.session_state.csuite_metrics['customer_satisfaction'].append(min(100, max(0, current_satisfaction)))
    
    add_log("System", f"Day {st.session_state.day}: {event_desc}", 'info')

def reset_simulation():
    """Resets all simulation state to its initial values."""
    st.session_state.clear()
    st.experimental_rerun()

# --- Persona-Specific Actions ---
def run_ai_forecasting():
    """Simulates implementing AI-driven forecasting."""
    profit_boost = random.randint(AI_IMPACT_PROFIT_MIN, AI_IMPACT_PROFIT_MAX)
    efficiency_boost = random.randint(AI_IMPACT_OP_EFFICIENCY_MIN - 2, AI_IMPACT_OP_EFFICIENCY_MAX - 2)
    satisfaction_boost = random.randint(AI_IMPACT_CUST_SAT_MIN - 2, AI_IMPACT_CUST_SAT_MAX - 2)
    advance_day(profit_boost, efficiency_boost, satisfaction_boost, "AI-Driven forecasting implemented, improving sales and reducing inventory costs.")
    add_log("C-Suite", "Decision: Implemented AI-Driven Forecasting. Saw positive impact on Profitability and Efficiency.", 'success')

def run_ai_logistics():
    """Simulates implementing AI-driven logistics."""
    profit_boost = random.randint(AI_IMPACT_PROFIT_MIN - 5, AI_IMPACT_PROFIT_MAX - 5)
    efficiency_boost = random.randint(AI_IMPACT_OP_EFFICIENCY_MIN, AI_IMPACT_OP_EFFICIENCY_MAX)
    satisfaction_boost = random.randint(AI_IMPACT_CUST_SAT_MIN, AI_IMPACT_CUST_SAT_MAX)
    advance_day(profit_boost, efficiency_boost, satisfaction_boost, "AI-Driven logistics optimization reduces fuel costs and improves delivery times.")
    add_log("C-Suite", "Decision: Implemented AI-Driven Logistics. Saw significant boost in Efficiency and Customer Satisfaction.", 'success')

def run_ai_supplier_risk():
    """Simulates implementing AI-driven supplier risk analysis."""
    profit_boost = random.randint(AI_IMPACT_PROFIT_MIN - 10, AI_IMPACT_PROFIT_MAX - 10)
    efficiency_boost = random.randint(AI_IMPACT_OP_EFFICIENCY_MIN - 5, AI_IMPACT_OP_EFFICIENCY_MAX - 5)
    satisfaction_boost = random.randint(AI_IMPACT_CUST_SAT_MIN - 5, AI_IMPACT_CUST_SAT_MAX - 5)
    advance_day(profit_boost, efficiency_boost, satisfaction_boost, "AI-driven supplier risk analysis helps avoid disruptions and maintain business continuity.")
    add_log("C-Suite", "Decision: Implemented AI-driven supplier risk analysis. Mitigated risk, ensuring stability.", 'success')

def run_integrated_data_platform():
    """Simulates implementing an integrated data platform."""
    profit_boost = random.randint(AI_IMPACT_PROFIT_MIN - 15, AI_IMPACT_PROFIT_MAX - 15)
    efficiency_boost = random.randint(AI_IMPACT_OP_EFFICIENCY_MIN + 2, AI_IMPACT_OP_EFFICIENCY_MAX + 2)
    satisfaction_boost = random.randint(AI_IMPACT_CUST_SAT_MIN - 5, AI_IMPACT_CUST_SAT_MAX - 5)
    advance_day(profit_boost, efficiency_boost, satisfaction_boost, "CIO decision: Implementing an integrated data platform, improving data flow and efficiency.")
    add_log("C-Suite", "Decision: Implemented Integrated Data Platform. Saw a strong boost in Operational Efficiency.", 'success')

def run_ai_security_platform():
    """Simulates implementing an AI-driven security platform."""
    profit_boost = random.randint(AI_IMPACT_PROFIT_MIN - 10, AI_IMPACT_PROFIT_MAX - 10)
    efficiency_boost = random.randint(AI_IMPACT_OP_EFFICIENCY_MIN - 2, AI_IMPACT_OP_EFFICIENCY_MAX - 2)
    satisfaction_boost = random.randint(AI_IMPACT_CUST_SAT_MIN + 2, AI_IMPACT_CUST_SAT_MAX + 2)
    advance_day(profit_boost, efficiency_boost, satisfaction_boost, "CISO decision: Implementing AI-driven security to protect supply chain data.")
    add_log("C-Suite", "Decision: Implemented AI-Driven Security Platform. Saw a boost in Customer Satisfaction and mitigated risk.", 'success')
    
def run_ai_supply_chain_optimization():
    """Simulates implementing end-to-end AI supply chain optimization."""
    profit_boost = random.randint(AI_IMPACT_PROFIT_MIN + 5, AI_IMPACT_PROFIT_MAX + 5)
    efficiency_boost = random.randint(AI_IMPACT_OP_EFFICIENCY_MIN + 5, AI_IMPACT_OP_EFFICIENCY_MAX + 5)
    satisfaction_boost = random.randint(AI_IMPACT_CUST_SAT_MIN + 5, AI_IMPACT_CUST_SAT_MAX + 5)
    advance_day(profit_boost, efficiency_boost, satisfaction_boost, "CSCO decision: Implementing end-to-end AI supply chain optimization for resilience and efficiency.")
    add_log("C-Suite", "Decision: Implemented End-to-End AI Supply Chain Optimization. Saw a significant boost across all metrics.", 'success')

def run_business_as_usual():
    """Simulates a day with no AI/ML implementation."""
    profit_change = random.randint(-15, 10)
    efficiency_change = random.randint(-5, 3)
    satisfaction_change = random.randint(-5, 2)
    advance_day(profit_change, efficiency_change, satisfaction_change, "Business as usual. Small fluctuations in metrics.")
    add_log("C-Suite", "Decision: No new AI/ML initiatives. Metrics fluctuate based on market conditions.", 'warning')

# --- UI Layout ---
st.title("👑 SCM C-Suite Simulation: AI/ML & Security Impact")
st.markdown("Simulate strategic decisions and observe their impact on key C-level metrics.")

st.sidebar.header("C-Suite Personas")
st.sidebar.markdown("""
- **CEO:** Driven by **Growth** & **Customer Satisfaction**.
- **CFO:** Driven by **Profitability** & **Working Capital**.
- **COO:** Driven by **Operational Efficiency** & **Resilience**.
- **CIO:** Driven by **Technology Adoption** & **Data Integration**.
- **CISO:** Driven by **Risk Mitigation** & **Data Security**.
- **CSCO:** Driven by **End-to-End Visibility** & **SCM Resilience**.
""")

st.sidebar.button("🔄 Reset Simulation", on_click=reset_simulation)
st.sidebar.markdown("---")

st.header("Strategic Decisions")
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    st.button("🔮 AI Forecasting", on_click=run_ai_forecasting)
with col2:
    st.button("🚚 AI Logistics", on_click=run_ai_logistics)
with col3:
    st.button("📝 AI Supplier Risk", on_click=run_ai_supplier_risk)
with col4:
    st.button("🌐 Data Platform", on_click=run_integrated_data_platform)
with col5:
    st.button("🛡️ AI Security", on_click=run_ai_security_platform)
with col6:
    st.button("🔗 AI SCM Optimization", on_click=run_ai_supply_chain_optimization)
with col7:
    st.button("📊 Business as Usual", on_click=run_business_as_usual)
st.markdown("---")

# --- Dashboard Metrics ---
st.header("C-Suite Dashboard")
df_metrics = pd.DataFrame(st.session_state.csuite_metrics)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📊 Profitability", value=f"${df_metrics['profitability'].iloc[-1]:.2f}")
with col2:
    st.metric(label="⚙️ Operational Efficiency", value=f"{df_metrics['operational_efficiency'].iloc[-1]:.2f}%")
with col3:
    st.metric(label="⭐ Customer Satisfaction", value=f"{df_metrics['customer_satisfaction'].iloc[-1]:.2f}%")

# --- Charts ---
st.subheader("Performance Trends Over Time")
fig_profit = px.line(df_metrics, x='days', y='profitability', title='Profitability Trend', markers=True)
st.plotly_chart(fig_profit, use_container_width=True)

fig_efficiency = px.line(df_metrics, x='days', y='operational_efficiency', title='Operational Efficiency Trend', markers=True)
st.plotly_chart(fig_efficiency, use_container_width=True)

fig_satisfaction = px.line(df_metrics, x='days', y='customer_satisfaction', title='Customer Satisfaction Trend', markers=True)
st.plotly_chart(fig_satisfaction, use_container_width=True)

st.markdown("---")
# --- Central Log ---
st.subheader("Decision Log")
if st.session_state.simulation_log:
    log_df = pd.DataFrame(st.session_state.simulation_log)
    st.dataframe(log_df.set_index('timestamp'), use_container_width=True, height=250)
else:
    st.info("Click a button above to start the simulation!")
