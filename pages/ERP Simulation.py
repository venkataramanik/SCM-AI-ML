import streamlit as st
import pandas as pd
from datetime import datetime
import random
import plotly.express as px
import time

# --- Configuration ---
st.set_page_config(
    page_title="Detailed C-Suite AI/ML Simulation",
    page_icon="🧠",
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
        'risk_score': [10],
        'ai_investments': set(),
    }
if 'day' not in st.session_state:
    st.session_state.day = 0
if 'ai_enabled' not in st.session_state:
    st.session_state.ai_enabled = {
        'forecasting': False,
        'logistics': False,
        'supplier_risk': False,
        'data_platform': False,
        'security': False,
        'scm_optimization': False,
    }

# --- Constants ---
# Metric Changes
PROFIT_PER_DAY = 5
EFFICIENCY_DECAY = -0.5
SATISFACTION_DECAY = -0.3
RISK_GROWTH = 0.5

# AI Impacts
AI_IMPACT = {
    'forecasting': {'profit': 15, 'efficiency': 5, 'satisfaction': 3},
    'logistics': {'profit': 10, 'efficiency': 8, 'satisfaction': 5},
    'supplier_risk': {'profit': 5, 'efficiency': 2, 'risk': -5},
    'data_platform': {'profit': 5, 'efficiency': 10, 'satisfaction': 2},
    'security': {'profit': 0, 'efficiency': -3, 'satisfaction': 8, 'risk': -15},
    'scm_optimization': {'profit': 25, 'efficiency': 15, 'satisfaction': 10, 'risk': -5},
}

# Market Events
MARKET_EVENTS = [
    {'name': 'Supply Chain Disruption', 'impact': {'profit': -150, 'efficiency': -20, 'satisfaction': -15, 'risk': 10}, 'chance': 0.1},
    {'name': 'Competitor Launch', 'impact': {'profit': -50, 'satisfaction': -10}, 'chance': 0.1},
    {'name': 'Security Breach', 'impact': {'profit': -200, 'satisfaction': -30, 'risk': 30}, 'chance': 0.05},
    {'name': 'Positive Market Trend', 'impact': {'profit': 100, 'satisfaction': 5}, 'chance': 0.1},
]

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

def advance_day(profit_change, efficiency_change, satisfaction_change, risk_change, event_desc):
    """Advances the simulation by one day with new metrics."""
    st.session_state.day += 1
    
    current_profit = st.session_state.csuite_metrics['profitability'][-1] + profit_change
    current_efficiency = st.session_state.csuite_metrics['operational_efficiency'][-1] + efficiency_change
    current_satisfaction = st.session_state.csuite_metrics['customer_satisfaction'][-1] + satisfaction_change
    current_risk = st.session_state.csuite_metrics['risk_score'][-1] + risk_change
    
    # Check for random market events
    event_message = ""
    for event in MARKET_EVENTS:
        if random.random() < event['chance']:
            event_name = event['name']
            event_impact = event['impact']
            
            # Reduce impact if AI solutions are in place
            impact_multiplier = 1.0
            if event_name == 'Supply Chain Disruption' and st.session_state.ai_enabled['supplier_risk']:
                impact_multiplier = 0.5 # AI mitigates half the impact
            if event_name == 'Security Breach' and st.session_state.ai_enabled['security']:
                impact_multiplier = 0.3 # AI mitigates most of the impact
            
            current_profit += event_impact.get('profit', 0) * impact_multiplier
            current_efficiency += event_impact.get('efficiency', 0) * impact_multiplier
            current_satisfaction += event_impact.get('satisfaction', 0) * impact_multiplier
            current_risk += event_impact.get('risk', 0) * impact_multiplier
            
            event_message = f"| **Market Event:** {event_name} occurred!"
            if impact_multiplier < 1.0:
                event_message += " AI solutions helped mitigate the impact."

    st.session_state.csuite_metrics['days'].append(st.session_state.day)
    st.session_state.csuite_metrics['profitability'].append(max(0, current_profit))
    st.session_state.csuite_metrics['operational_efficiency'].append(min(100, max(0, current_efficiency)))
    st.session_state.csuite_metrics['customer_satisfaction'].append(min(100, max(0, current_satisfaction)))
    st.session_state.csuite_metrics['risk_score'].append(min(100, max(0, current_risk)))

    add_log("System", f"Day {st.session_state.day}: {event_desc} {event_message}", 'info')

def reset_simulation():
    """Resets all simulation state to its initial values."""
    st.session_state.clear()
    st.experimental_rerun()

# --- Persona-Specific Actions (Dynamic Impacts based on AI investments) ---
def apply_investment(investment_key):
    """Applies the one-time impact of a new AI investment."""
    if investment_key not in st.session_state.csuite_metrics['ai_investments']:
        st.session_state.csuite_metrics['ai_investments'].add(investment_key)
        st.session_state.ai_enabled[investment_key] = True
        
        impact = AI_IMPACT[investment_key]
        profit_boost = impact.get('profit', 0)
        efficiency_boost = impact.get('efficiency', 0)
        satisfaction_boost = impact.get('satisfaction', 0)
        risk_reduction = -impact.get('risk', 0) # Risk is reduced

        advance_day(profit_boost, efficiency_boost, satisfaction_boost, risk_reduction,
                    f"Initiative: {investment_key.replace('_', ' ').title()} successfully implemented. Metrics boosted!")
        add_log("C-Suite", f"Strategic decision: Implemented {investment_key.replace('_', ' ').title()}.", 'success')
    else:
        advance_day(PROFIT_PER_DAY, EFFICIENCY_DECAY, SATISFACTION_DECAY, RISK_GROWTH, "Metrics continue to fluctuate.")
        add_log("C-Suite", f"The {investment_key.replace('_', ' ').title()} initiative is already in place. Continuing business as usual.", 'warning')

def run_business_as_usual():
    """Simulates a day with no new decisions."""
    advance_day(PROFIT_PER_DAY, EFFICIENCY_DECAY, SATISFACTION_DECAY, RISK_GROWTH, "Business as usual. Metrics are slowly degrading without new initiatives.")
    add_log("C-Suite", "Decision: No new AI/ML initiatives. Metrics slowly decline.", 'warning')

# --- UI Layout ---
st.title("🧠 Detailed C-Suite AI/ML Simulation")
st.markdown("A dynamic simulation to explore how strategic C-level decisions impact key business metrics over time, with random market events.")

# --- Sidebar ---
st.sidebar.header("C-Suite Priorities")
st.sidebar.markdown("""
- **CEO:** Market Share & Growth 📈
- **CFO:** Profitability & Cost Efficiency 💰
- **COO:** Operational Efficiency & Resilience ⚙️
- **CIO:** Technology & Data Integration 💻
- **CSCO:** Supply Chain Visibility & Risk 🔗
- **CISO:** Security & Risk Mitigation 🛡️
""")

st.sidebar.button("🔄 Reset Simulation", on_click=reset_simulation)
st.sidebar.markdown("---")
st.sidebar.header("AI Investment Status")
for key, value in st.session_state.ai_enabled.items():
    emoji = "✅" if value else "❌"
    st.sidebar.markdown(f"{emoji} {key.replace('_', ' ').title()}")

st.header("Strategic Decisions")
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    st.button("🔮 AI Forecasting", on_click=lambda: apply_investment('forecasting'))
with col2:
    st.button("🚚 AI Logistics", on_click=lambda: apply_investment('logistics'))
with col3:
    st.button("📝 AI Supplier Risk", on_click=lambda: apply_investment('supplier_risk'))
with col4:
    st.button("🌐 Data Platform", on_click=lambda: apply_investment('data_platform'))
with col5:
    st.button("🛡️ AI Security", on_click=lambda: apply_investment('security'))
with col6:
    st.button("🔗 AI SCM Optimization", on_click=lambda: apply_investment('scm_optimization'))
with col7:
    st.button("📊 Business as Usual", on_click=run_business_as_usual)

st.markdown("---")

# --- Dashboard Metrics ---
st.header("C-Suite Dashboard")
df_metrics = pd.DataFrame(st.session_state.csuite_metrics)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="📊 Profitability", value=f"${df_metrics['profitability'].iloc[-1]:.2f}")
with col2:
    st.metric(label="⚙️ Operational Efficiency", value=f"{df_metrics['operational_efficiency'].iloc[-1]:.2f}%")
with col3:
    st.metric(label="⭐ Customer Satisfaction", value=f"{df_metrics['customer_satisfaction'].iloc[-1]:.2f}%")
with col4:
    st.metric(label="⚠️ Risk Score", value=f"{df_metrics['risk_score'].iloc[-1]:.2f}")

# --- Charts ---
st.subheader("Performance Trends Over Time")
fig_metrics = px.line(df_metrics, x='days', y=['profitability', 'operational_efficiency', 'customer_satisfaction', 'risk_score'],
                      title='C-Suite Performance Over Time', markers=True)
fig_metrics.update_layout(yaxis_title="Metric Value", legend_title="Metrics")
st.plotly_chart(fig_metrics, use_container_width=True)

st.markdown("---")
# --- Central Log ---
st.subheader("Decision & Event Log")
if st.session_state.simulation_log:
    log_df = pd.DataFrame(st.session_state.simulation_log)
    st.dataframe(log_df.set_index('timestamp'), use_container_width=True, height=250)
else:
    st.info("Click a button above to start the simulation!")

