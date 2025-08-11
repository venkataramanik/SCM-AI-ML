import streamlit as st
import pandas as pd
from datetime import datetime
import random
import plotly.express as px
import time

# --- Configuration ---
st.set_page_config(
    page_title="Advanced Agentic SCM Simulation",
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
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'message_bus' not in st.session_state:
    st.session_state.message_bus = {}
if 'agent_states' not in st.session_state:
    st.session_state.agent_states = {
        'demand_forecast': {'current_forecast': 100},
        'procurement': {'pending_orders': []},
        'logistics': {'shipments': []},
        'learning': { # New state to track agent 'learning'
            'Supplier Delay': 0,
            'Labor Strike': 0,
            'Logistics Network Congestion': 0,
            'Port Closure': 0,
        }
    }
if 'human_intervention_needed' not in st.session_state:
    st.session_state.human_intervention_needed = False


# --- Constants ---
# KPI decay/growth rates without agent intervention
ON_TIME_DECAY_RATE = -0.2
COST_GROWTH_RATE = 15
INVENTORY_GROWTH_RATE = 0.5
RISK_GROWTH_RATE = 0.5

# Base agentic framework's impact on KPIs
BASE_AGENT_IMPACTS = {
    'demand_forecast': {'on_time': (2.0, 5.0), 'cost': (-30.0, -20.0), 'inventory': (-5.0, -3.0)},
    'procurement': {'on_time': (6.0, 10.0), 'cost': (-40.0, -25.0), 'risk': (-10.0, -7.0)},
    'logistics': {'on_time': (6.0, 10.0), 'cost': (-25.0, -15.0), 'inventory': (-5.0, -2.0)},
}

# Market Events (simulating real-world disruptions)
MARKET_EVENTS = [
    {'name': 'Supplier Delay', 'impact': {'on_time': -10, 'cost': 80, 'risk': 4}, 'chance': 0.1, 'trigger': 'procurement'},
    {'name': 'Unexpected Demand Spike', 'impact': {'on_time': -8, 'cost': 40, 'inventory': 5}, 'chance': 0.1, 'trigger': 'demand_forecast'},
    {'name': 'Logistics Network Congestion', 'impact': {'on_time': -10, 'cost': 25}, 'chance': 0.15, 'trigger': 'logistics'},
    {'name': 'Labor Strike', 'impact': {'on_time': -20, 'cost': 150, 'risk': 10}, 'chance': 0.05, 'trigger': 'procurement'},
    {'name': 'Port Closure', 'impact': {'on_time': -25, 'cost': 200, 'risk': 12}, 'chance': 0.03, 'trigger': 'logistics'},
    # The 'black swan' event that requires human intervention
    {'name': 'Global Supply Chain Freeze', 'impact': {'on_time': -50, 'cost': 500, 'risk': 30}, 'chance': 0.01, 'trigger': 'human_intervention'},
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

def advance_day():
    """Advances the simulation by one day with new KPI values."""
    if st.session_state.human_intervention_needed:
        return # Halt simulation if human intervention is needed
        
    st.session_state.day += 1

    # Apply natural decay/growth
    on_time_change = ON_TIME_DECAY_RATE
    cost_change = COST_GROWTH_RATE
    inventory_change = INVENTORY_GROWTH_RATE
    risk_change = RISK_GROWTH_RATE
    
    # Process market events and their impacts
    for event in MARKET_EVENTS:
        if random.random() < event['chance']:
            event_name = event['name']
            event_impact = event['impact']
            
            event_message = f"| **Market Event:** {event_name} occurred! "
            add_log("System", event_message, 'warning')
            
            # Send a message to the relevant agent or human intervention channel
            st.session_state.message_bus[event['trigger']] = {'type': 'event', 'data': event_name, 'impact': event_impact}

            on_time_change += event_impact.get('on_time', 0)
            cost_change += event_impact.get('cost', 0)
            inventory_change += event_impact.get('inventory', 0)
            risk_change += event_impact.get('risk', 0)
    
    # Let agents run and react
    agent_outputs = run_agents()

    # Apply agent impacts
    for agent, impacts in agent_outputs.items():
        if impacts:
            on_time_change += impacts.get('on_time', 0)
            cost_change += impacts.get('cost', 0)
            inventory_change += impacts.get('inventory', 0)
            risk_change += impacts.get('risk', 0)

    current_on_time = st.session_state.kpis['on_time_delivery_rate'][-1] + on_time_change
    current_cost = st.session_state.kpis['supply_chain_cost'][-1] + cost_change
    current_inventory = st.session_state.kpis['inventory_days_of_supply'][-1] + inventory_change
    current_risk = st.session_state.kpis['risk_exposure_score'][-1] + risk_change
    
    st.session_state.kpis['day'].append(st.session_state.day)
    st.session_state.kpis['on_time_delivery_rate'].append(min(100, max(0, current_on_time)))
    st.session_state.kpis['supply_chain_cost'].append(max(0, current_cost))
    st.session_state.kpis['inventory_days_of_supply'].append(max(0, current_inventory))
    st.session_state.kpis['risk_exposure_score'].append(min(100, max(0, current_risk)))

    add_log("System", f"Day {st.session_state.day}: KPI changes calculated.", 'info')

def run_agents():
    """Orchestrates agent interactions for the current day."""
    impacts = {}
    
    # Check for human intervention message first
    if 'human_intervention' in st.session_state.message_bus:
        message = st.session_state.message_bus.pop('human_intervention')
        add_log("System", f"Agents unable to handle event: '{message['data']}'. Requiring human takeover.", 'critical')
        st.session_state.human_intervention_needed = True
        st.session_state.simulation_running = False
        return {}
    
    # Agent 1: Demand Forecasting Agent
    forecast_impacts = run_demand_forecast_agent()
    if forecast_impacts:
        impacts['demand_forecast'] = forecast_impacts

    # Agent 2: Procurement Agent
    procurement_impacts = run_procurement_agent()
    if procurement_impacts:
        impacts['procurement'] = procurement_impacts

    # Agent 3: Logistics Agent
    logistics_impacts = run_logistics_agent()
    if logistics_impacts:
        impacts['logistics'] = logistics_impacts

    return impacts


def run_demand_forecast_agent():
    """Simulates the Demand Forecasting Agent's actions."""
    impacts = {}
    
    # Proactive "business-as-usual" optimization
    impacts['on_time'] = random.uniform(0.1, 0.5)
    impacts['cost'] = random.uniform(-5.0, -1.0)
    
    if 'demand_forecast' in st.session_state.message_bus:
        message = st.session_state.message_bus.pop('demand_forecast')
        if message['type'] == 'event' and message['data'] == 'Unexpected Demand Spike':
            add_log("DemandForecastAgent", "Detected unexpected demand spike. Adjusting forecast and communicating to Procurement.", 'warning')
            st.session_state.agent_states['demand_forecast']['current_forecast'] = 150
            st.session_state.message_bus['procurement'] = {'type': 'forecast_update', 'data': 150}
            
            # Agents help mitigate the problem
            impacts['on_time'] += random.uniform(*BASE_AGENT_IMPACTS['demand_forecast']['on_time'])
            impacts['cost'] += random.uniform(*BASE_AGENT_IMPACTS['demand_forecast']['cost'])
            impacts['inventory'] = random.uniform(*BASE_AGENT_IMPACTS['demand_forecast']['inventory'])
    
    add_log("DemandForecastAgent", "Proactively forecasting and optimizing inventory levels.")
    
    # Every few days, provide a new forecast
    if st.session_state.day % 5 == 0:
        new_forecast = random.randint(90, 110)
        st.session_state.agent_states['demand_forecast']['current_forecast'] = new_forecast
        st.session_state.message_bus['procurement'] = {'type': 'forecast_update', 'data': new_forecast}
        add_log("DemandForecastAgent", f"Published new forecast: {new_forecast} units.")
    
    return impacts

def run_procurement_agent():
    """Simulates the Procurement Agent's actions."""
    impacts = {}
    
    # Proactive "business-as-usual" optimization
    impacts['on_time'] = random.uniform(0.1, 0.5)
    impacts['cost'] = random.uniform(-5.0, -1.0)
    
    # Check for incoming messages
    if 'procurement' in st.session_state.message_bus:
        message = st.session_state.message_bus.pop('procurement')
        event_name = message['data']

        if event_name in ['Supplier Delay', 'Labor Strike']:
            add_log("ProcurementAgent", f"Received alert: {event_name}. Searching for alternative suppliers and pre-ordering materials.", 'warning')
            
            # Higher risk score means lower chance of success
            if random.random() > st.session_state.kpis['risk_exposure_score'][-1] / 100.0:
                # Use .get() with a default value to prevent KeyError
                learning_bonus = st.session_state.agent_states['learning'].get(event_name, 0) * 0.5
                add_log("ProcurementAgent", f"Successfully found an alternative supplier. Learning improved for '{event_name}'.", 'success')
                # Only increment if the key exists
                if event_name in st.session_state.agent_states['learning']:
                    st.session_state.agent_states['learning'][event_name] += 1
                
                impacts['on_time'] += random.uniform(BASE_AGENT_IMPACTS['procurement']['on_time'][0] + learning_bonus, BASE_AGENT_IMPACTS['procurement']['on_time'][1] + learning_bonus)
                impacts['cost'] += random.uniform(BASE_AGENT_IMPACTS['procurement']['cost'][0], BASE_AGENT_IMPACTS['procurement']['cost'][1])
                impacts['risk'] = random.uniform(BASE_AGENT_IMPACTS['procurement']['risk'][0], BASE_AGENT_IMPACTS['procurement']['risk'][1])
            else:
                add_log("ProcurementAgent", f"Failed to find a quick alternative. Shipment will be delayed.", 'error')

        elif message['type'] == 'forecast_update':
            new_forecast = message['data']
            add_log("ProcurementAgent", f"Received new forecast of {new_forecast}. Adjusting order quantities.")
            # Adjust inventory and cost based on new orders
            impacts['cost'] += random.uniform(-5.0, -2.0)
            impacts['inventory'] = random.uniform(-1.0, 1.0)
    
    add_log("ProcurementAgent", "Executing daily purchasing and proactively monitoring suppliers.")
    
    return impacts

def run_logistics_agent():
    """Simulates the Logistics Agent's actions."""
    impacts = {}
    
    # Proactive "business-as-usual" optimization
    impacts['on_time'] = random.uniform(0.1, 0.5)
    impacts['cost'] = random.uniform(-5.0, -1.0)
    
    # Check for incoming messages
    if 'logistics' in st.session_state.message_bus:
        message = st.session_state.message_bus.pop('logistics')
        event_name = message['data']
        
        if event_name in ['Logistics Network Congestion', 'Port Closure']:
            add_log("LogisticsAgent", f"Detected {event_name}. Rerouting shipments and seeking alternative transport.", 'warning')
            
            # Chance of successful reroute
            if random.random() < 0.7:
                # Use .get() with a default value to prevent KeyError
                learning_bonus = st.session_state.agent_states['learning'].get(event_name, 0) * 0.5
                add_log("LogisticsAgent", f"Rerouting successful. Learning improved for '{event_name}'.", 'success')
                # Only increment if the key exists
                if event_name in st.session_state.agent_states['learning']:
                    st.session_state.agent_states['learning'][event_name] += 1

                impacts['on_time'] += random.uniform(BASE_AGENT_IMPACTS['logistics']['on_time'][0] + learning_bonus, BASE_AGENT_IMPACTS['logistics']['on_time'][1] + learning_bonus)
                impacts['cost'] += random.uniform(BASE_AGENT_IMPACTS['logistics']['cost'][0], BASE_AGENT_IMPACTS['logistics']['cost'][1])
            else:
                add_log("LogisticsAgent", f"Failed to find an efficient alternative route. Delays expected.", 'error')
    
    add_log("LogisticsAgent", "Optimizing daily delivery routes to improve efficiency.")
    
    return impacts

def start_simulation():
    if not st.session_state.human_intervention_needed:
        st.session_state.simulation_running = True

def stop_simulation():
    st.session_state.simulation_running = False

def human_intervene():
    """Action for human intervention button. Applies a large fix and resets the state."""
    st.session_state.kpis['on_time_delivery_rate'][-1] += 30 # Big manual fix
    st.session_state.kpis['supply_chain_cost'][-1] -= 200 # Big manual fix
    st.session_state.human_intervention_needed = False
    st.session_state.simulation_running = False
    add_log("Human", "Intervened to solve the black swan event. The supply chain is recovering.", 'success')

def reset_simulation():
    """Resets all simulation state to its initial values."""
    st.session_state.clear()
    # The app will automatically re-run after the callback finishes.

# --- UI Layout ---
st.title("🤖 Advanced Agentic Supply Chain Simulation")
st.markdown("This simulation models an **Agentic Framework** with autonomous, interacting agents. They work proactively to manage key KPIs and react to dynamic market events.")

# --- Sidebar ---
st.sidebar.header("Control Panel")
if st.session_state.simulation_running:
    st.sidebar.button("⏹️ Stop Simulation", on_click=stop_simulation)
elif st.session_state.human_intervention_needed:
    st.sidebar.warning("🚨 A major crisis requires human intervention!")
    st.sidebar.button("👨‍💼 Human Intervention: Resolve Crisis", on_click=human_intervene)
else:
    st.sidebar.button("▶️ Start Simulation", on_click=start_simulation)
st.sidebar.button("🔄 Reset Simulation", on_click=reset_simulation)
st.sidebar.markdown("---")
st.sidebar.header("Agentic Framework")
st.sidebar.markdown("""
- **Demand Forecast Agent:** Predicts demand to optimize inventory.
- **Procurement Agent:** Manages sourcing, orders, and supplier risk.
- **Logistics Agent:** Optimizes shipping and delivery routes.
- **Human:** Steps in for unprecedented, "black swan" events.
""")

st.header("Simulation Dashboard")

st.markdown("---")
# --- Central Log ---
st.subheader("Agent & Event Log")
if st.session_state.simulation_log:
    log_df = pd.DataFrame(st.session_state.simulation_log)
    st.dataframe(log_df.set_index('timestamp'), use_container_width=True, height=250)
else:
    st.info("Click 'Start Simulation' to begin.")

st.markdown("---")
st.subheader("Performance Trends Over Time")
df_metrics = pd.DataFrame(st.session_state.kpis)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="✅ On-Time Delivery Rate", value=f"{df_metrics['on_time_delivery_rate'].iloc[-1]:.2f}%")
with col2:
    st.metric(label="💵 Supply Chain Cost", value=f"${df_metrics['supply_chain_cost'].iloc[-1]:.2f}")
with col3:
    st.metric(label="📦 Inventory Days", value=f"{df_metrics['inventory_days_of_supply'].iloc[-1]:.2f} days")
with col4:
    st.metric(label="🚨 Risk Exposure Score", value=f"{df_metrics['risk_exposure_score'].iloc[-1]:.2f}")

# --- Charts ---
fig_metrics = px.line(df_metrics, x='day', y=df_metrics.columns[1:],
                      title='Supply Chain KPIs Over Time', markers=True)
fig_metrics.update_layout(yaxis_title="Value", legend_title="KPIs")
st.plotly_chart(fig_metrics, use_container_width=True)

# --- The Simulation Loop ---
if st.session_state.simulation_running:
    time.sleep(1) # Simulates a day passing every second
    advance_day()
    st.rerun()
