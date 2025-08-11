import streamlit as st
import pandas as pd
from datetime import datetime
import random
import plotly.express as px

# --- Configuration ---
st.set_page_config(
    page_title="Agentic SCM Simulation",
    page_icon="🤖",
    layout="wide",
)

# --- State Initialization ---
if 'simulation_log' not in st.session_state:
    st.session_state.simulation_log = []
if 'kpis' not in st.session_state:
    st.session_state.kpis = {
        'day': [0],
        'on_time_delivery_rate': [90.0],
        'supply_chain_cost': [1000.0],
        'inventory_days_of_supply': [30.0],
        'risk_exposure_score': [10.0],
    }
if 'day' not in st.session_state:
    st.session_state.day = 0

# --- Constants ---
# KPI decay/growth rates without agent intervention
ON_TIME_DECAY_RATE = -0.2
COST_GROWTH_RATE = 10
INVENTORY_DECAY_RATE = -0.5
RISK_GROWTH_RATE = 0.5

# Agentic framework's impact on KPIs
AGENT_IMPACTS = {
    'demand_forecast': {'on_time': 2, 'cost': -15, 'inventory': -2},
    'procurement': {'on_time': 5, 'cost': -20, 'risk': -5},
    'logistics': {'on_time': 5, 'cost': -10, 'inventory': -1},
    'customer_service': {'on_time': 1, 'satisfaction': 5},
}

# Market Events (simulating real-world disruptions)
MARKET_EVENTS = [
    {'name': 'Supplier Delay', 'impact': {'on_time': -15, 'cost': 100, 'risk': 5}, 'chance': 0.15},
    {'name': 'Unexpected Demand Spike', 'impact': {'on_time': -10, 'cost': 50, 'inventory': 5}, 'chance': 0.1},
    {'name': 'Logistics Network Congestion', 'impact': {'on_time': -12, 'cost': 30}, 'chance': 0.15},
    {'name': 'Geopolitical Event', 'impact': {'on_time': -20, 'cost': 150, 'risk': 20}, 'chance': 0.05},
]

# --- Helper Functions ---
def add_log(agent, message, level='info'):
    """Adds a timestamped message to the simulation log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.simulation_log.append({
        'timestamp': timestamp,
        'agent': agent,
        'message': message,
        'level': level,
    })

def advance_day(on_time_change, cost_change, inventory_change, risk_change, event_desc):
    """Advances the simulation by one day with new KPI values."""
    st.session_state.day += 1
    
    current_on_time = st.session_state.kpis['on_time_delivery_rate'][-1] + on_time_change
    current_cost = st.session_state.kpis['supply_chain_cost'][-1] + cost_change
    current_inventory = st.session_state.kpis['inventory_days_of_supply'][-1] + inventory_change
    current_risk = st.session_state.kpis['risk_exposure_score'][-1] + risk_change
    
    # Check for random market events
    event_message = ""
    for event in MARKET_EVENTS:
        if random.random() < event['chance']:
            event_name = event['name']
            event_impact = event['impact']
            
            # Agents react to events, mitigating impact
            agent_reaction = ""
            if event_name == 'Supplier Delay':
                if random.random() > st.session_state.kpis['risk_exposure_score'][-1] / 100.0:
                    current_on_time += AGENT_IMPACTS['procurement']['on_time'] * 0.7 # Procurement Agent finds a partial solution
                    current_cost += AGENT_IMPACTS['procurement']['cost'] * 0.5
                    agent_reaction = "Procurement Agent initiated alternative sourcing."
                else:
                    agent_reaction = "Procurement Agent failed to find a quick alternative."

            if event_name == 'Unexpected Demand Spike':
                if random.random() < 0.8: # High chance of success for forecasting agent
                    current_inventory += AGENT_IMPACTS['demand_forecast']['inventory'] * 0.8
                    current_cost += AGENT_IMPACTS['demand_forecast']['cost'] * 0.5
                    current_on_time += AGENT_IMPACTS['demand_forecast']['on_time'] * 0.5
                    agent_reaction = "Demand Forecasting Agent adjusted plans."
            
            if event_name == 'Logistics Network Congestion':
                 if random.random() < 0.7:
                     current_on_time += AGENT_IMPACTS['logistics']['on_time'] * 0.7
                     current_cost += AGENT_IMPACTS['logistics']['cost'] * 0.5
                     agent_reaction = "Logistics Agent rerouted shipments."
            
            current_on_time += event_impact.get('on_time', 0)
            current_cost += event_impact.get('cost', 0)
            current_inventory += event_impact.get('inventory', 0)
            current_risk += event_impact.get('risk', 0)
            
            event_message = f"| **Market Event:** {event_name} occurred! {agent_reaction}"

    st.session_state.kpis['day'].append(st.session_state.day)
    st.session_state.kpis['on_time_delivery_rate'].append(min(100, max(0, current_on_time)))
    st.session_state.kpis['supply_chain_cost'].append(max(0, current_cost))
    st.session_state.kpis['inventory_days_of_supply'].append(max(0, current_inventory)))
    st.session_state.kpis['risk_exposure_score'].append(min(100, max(0, current_risk)))

    add_log("System", f"Day {st.session_state.day}: {event_desc} {event_message}", 'info')

def run_simulation_day():
    """Advances the simulation by one day, allowing agents to act."""
    on_time_change = ON_TIME_DECAY_RATE
    cost_change = COST_GROWTH_RATE
    inventory_change = INVENTORY_DECAY_RATE
    risk_change = RISK_GROWTH_RATE
    
    # If the system is "Agentic", agents actively work to improve KPIs
    on_time_change += AGENT_IMPACTS['procurement']['on_time']
    cost_change += AGENT_IMPACTS['procurement']['cost']
    risk_change += AGENT_IMPACTS['procurement']['risk']

    on_time_change += AGENT_IMPACTS['logistics']['on_time']
    cost_change += AGENT_IMPACTS['logistics']['cost']
    inventory_change += AGENT_IMPACTS['logistics']['inventory']

    on_time_change += AGENT_IMPACTS['demand_forecast']['on_time']
    cost_change += AGENT_IMPACTS['demand_forecast']['cost']
    inventory_change += AGENT_IMPACTS['demand_forecast']['inventory']

    advance_day(on_time_change, cost_change, inventory_change, risk_change, "Agentic framework is active.")
    add_log("System", "Agents are working autonomously to optimize the supply chain.", 'success')

def reset_simulation():
    """Resets all simulation state to its initial values."""
    st.session_state.clear()
    st.experimental_rerun()

# --- UI Layout ---
st.title("🤖 Agentic Supply Chain Simulation")
st.markdown("Observe how an **Agentic Framework** with autonomous agents manages key supply chain KPIs and responds to market events.")

# --- Sidebar ---
st.sidebar.header("Agentic Framework")
st.sidebar.markdown("""
- **Demand Forecasting Agent:** Predicts demand to optimize inventory.
- **Procurement Agent:** Manages sourcing and supplier risk.
- **Logistics Agent:** Optimizes shipping and delivery routes.
- **Customer Service Agent:** Manages order updates and customer satisfaction.
""")
st.sidebar.button("🔄 Reset Simulation", on_click=reset_simulation)
st.sidebar.markdown("---")

st.header("Control Panel")
st.button("⏩ Run Agentic Day", on_click=run_simulation_day)
st.markdown("---")

# --- Dashboard Metrics ---
st.header("Supply Chain KPIs")
df_metrics = pd.DataFrame(st.session_state.kpis)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="✅ On-Time Delivery Rate", value=f"{df_metrics['on_time_delivery_rate'].iloc[-1]:.2f}%")
with col2:
    st.metric(label="💵 Total Supply Chain Cost", value=f"${df_metrics['supply_chain_cost'].iloc[-1]:.2f}")
with col3:
    st.metric(label="📦 Inventory Days of Supply", value=f"{df_metrics['inventory_days_of_supply'].iloc[-1]:.2f} days")
with col4:
    st.metric(label="🚨 Risk Exposure Score", value=f"{df_metrics['risk_exposure_score'].iloc[-1]:.2f}")

# --- Charts ---
st.subheader("Performance Trends Over Time")
fig_metrics = px.line(df_metrics, x='day', y=df_metrics.columns[1:],
                      title='Supply Chain KPIs Over Time', markers=True)
fig_metrics.update_layout(yaxis_title="Value", legend_title="KPIs")
st.plotly_chart(fig_metrics, use_container_width=True)

st.markdown("---")
# --- Central Log ---
st.subheader("Agent & Event Log")
if st.session_state.simulation_log:
    log_df = pd.DataFrame(st.session_state.simulation_log)
    st.dataframe(log_df.set_index('timestamp'), use_container_width=True, height=250)
else:
    st.info("Click 'Run Agentic Day' to start the simulation!")
