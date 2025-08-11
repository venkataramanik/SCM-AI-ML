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
if 'human_intervention_needed' in st.session_state and st.session_state.human_intervention_needed:
    st.session_state.simulation_running = False
else:
    st.session_state.human_intervention_needed = False
if 'agent_boost' not in st.session_state:
    st.session_state.agent_boost = None
if 'boost_counter' not in st.session_state:
    st.session_state.boost_counter = 0

# --- Constants ---
# Corrected KPI decay/growth rates without agent intervention to be less punishing
ON_TIME_DECAY_RATE = -0.1
COST_GROWTH_RATE = 10
INVENTORY_GROWTH_RATE = 0.2
RISK_GROWTH_RATE = 0.3

# Corrected base agentic framework's impact on KPIs to be more effective
BASE_AGENT_IMPACTS = {
    'demand_forecast': {'on_time': (4.0, 7.0), 'cost': (-40.0, -30.0), 'inventory': (-8.0, -5.0)},
    'procurement': {'on_time': (8.0, 12.0), 'cost': (-50.0, -35.0), 'risk': (-15.0, -10.0)},
    'logistics': {'on_time': (8.0, 12.0), 'cost': (-35.0, -20.0), 'inventory': (-8.0, -5.0)},
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

BOOST_CHANCE = 0.05
BOOST_DURATION = 3
BOOST_MULTIPLIER = 1.5

# --- Helper Functions ---
def add_log(agent, message, level='info'):
    """Adds a timestamped message to the simulation log."""
    st.session_state.simulation_log.append({
        'agent': agent,
        'message': message,
        'level': level,
    })

def trigger_agent_boost():
    """Triggers an agent boost event."""
    if random.random() < BOOST_CHANCE and st.session_state.agent_boost is None:
        boost_type = random.choice(["Procurement Boost", "Logistics Boost"])
        st.session_state.agent_boost = boost_type
        st.session_state.boost_counter = BOOST_DURATION
        add_log("System", f"A random event has occurred! An '{boost_type}' has been activated for {BOOST_DURATION} days.", 'success')

def advance_day():
    """Advances the simulation by one day with new KPI values."""
    if st.session_state.human_intervention_needed:
        return # Halt simulation if human intervention is needed
        
    st.session_state.day += 1

    # Check and trigger a random boost
    trigger_agent_boost()

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
            
            event_message = f"**Market Event:** {event_name} occurred!"
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
    
    # Decrement boost counter and remove boost if expired
    if st.session_state.agent_boost:
        st.session_state.boost_counter -= 1
        if st.session_state.boost_counter <= 0:
            add_log("System", f"{st.session_state.agent_boost} has expired.", 'info')
            st.session_state.agent_boost = None

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
    impacts['on_time'] = random.uniform(0.5, 1.0)
    impacts['cost'] = random.uniform(-10.0, -5.0)
    
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
    impacts['on_time'] = random.uniform(0.5, 1.0)
    impacts['cost'] = random.uniform(-10.0, -5.0)

    # Check for incoming messages
    if 'procurement' in st.session_state.message_bus:
        message = st.session_state.message_bus.pop('procurement')
        event_name = message['data']

        if event_name in ['Supplier Delay', 'Labor Strike']:
            add_log("ProcurementAgent", f"Received alert: {event_name}. Searching for alternative suppliers and pre-ordering materials.", 'warning')
            
            # The chance of success is now more balanced
            if random.random() < (100 - st.session_state.kpis['risk_exposure_score'][-1]) / 100.0:
                # Apply boost if active
                learning_bonus = st.session_state.agent_states['learning'].get(event_name, 0) * 0.5
                
                boost_multiplier = 1
                if st.session_state.agent_boost == "Procurement Boost":
                    boost_multiplier = BOOST_MULTIPLIER
                    add_log("ProcurementAgent", "Applying Procurement Boost to mitigate event.", 'info')

                add_log("ProcurementAgent", f"Successfully found an alternative supplier. Learning improved for '{event_name}'.", 'success')
                # Only increment if the key exists
                if event_name in st.session_state.agent_states['learning']:
                    st.session_state.agent_states['learning'][event_name] += 1
                
                impacts['on_time'] += random.uniform((BASE_AGENT_IMPACTS['procurement']['on_time'][0] + learning_bonus) * boost_multiplier, (BASE_AGENT_IMPACTS['procurement']['on_time'][1] + learning_bonus) * boost_multiplier)
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
    impacts['on_time'] = random.uniform(0.5, 1.0)
    impacts['cost'] = random.uniform(-10.0, -5.0)
    
    # Check for incoming messages
    if 'logistics' in st.session_state.message_bus:
        message = st.session_state.message_bus.pop('logistics')
        event_name = message['data']
        
        if event_name in ['Logistics Network Congestion', 'Port Closure']:
            add_log("LogisticsAgent", f"Detected {event_name}. Rerouting shipments and seeking alternative transport.", 'warning')
            
            # Chance of successful reroute
            if random.random() < 0.7:
                # Apply boost if active
                learning_bonus = st.session_state.agent_states['learning'].get(event_name, 0) * 0.5
                
                boost_multiplier = 1
                if st.session_state.agent_boost == "Logistics Boost":
                    boost_multiplier = BOOST_MULTIPLIER
                    add_log("LogisticsAgent", "Applying Logistics Boost to mitigate event.", 'info')

                add_log("LogisticsAgent", f"Rerouting successful. Learning improved for '{event_name}'.", 'success')
                # Only increment if the key exists
                if event_name in st.session_state.agent_states['learning']:
                    st.session_state.agent_states['learning'][event_name] += 1

                impacts['on_time'] += random.uniform((BASE_AGENT_IMPACTS['logistics']['on_time'][0] + learning_bonus) * boost_multiplier, (BASE_AGENT_IMPACTS['logistics']['on_time'][1] + learning_bonus) * boost_multiplier)
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
    st.session_state.kpis['on_time_delivery_rate'][-1] += 50 # Increased manual fix
    st.session_state.kpis['supply_chain_cost'][-1] -= 300 # Increased manual fix
    st.session_state.human_intervention_needed = False
    st.session_state.simulation_running = False
    add_log("Human", "Intervened to solve the black swan event. The supply chain is recovering.", 'success')

def reset_simulation():
    """Resets all simulation state to its initial values."""
    st.session_state.clear()
    st.experimental_rerun()

# --- UI Layout ---
st.title("🤖 Advanced Agentic Supply Chain Simulation")
st.markdown("This simulation models an **Agentic Framework** with autonomous, interacting agents. They work proactively to manage key KPIs and react to dynamic market events.")
st.markdown("---")
st.markdown("""
### 🚀 The Agent Boost Mechanism

The **Agent Boost** is a special, temporary state that enhances an agent's ability to respond to a supply chain disruption. In this simulation, a boost event (either a **Procurement Boost** or a **Logistics Boost**) has a small chance of being randomly triggered and will last for a few days. When a boosted agent successfully mitigates a market event during this time, its positive impact on the key performance indicators (KPIs) is significantly multiplied. This simulates a temporary improvement in an agent's efficiency or access to better resources.

### 🏃 How to Run the Simulation

To start, simply click the **▶️ Start Simulation** button in the control panel on the left. The simulation will advance one "day" at a time, with agents taking action and events potentially occurring. You can observe the performance trends over time and the detailed agent logs below. If a catastrophic event occurs, the simulation will pause and require a human intervention.
""")
st.markdown("---")


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
- **Procurement Agent:** Manages sourcing, orders, and supplier risk.<br>
- **Logistics Agent:** Optimizes shipping and delivery routes.
- **Human:** Steps in for unprecedented, "black swan" events.
""")

st.header("Simulation Dashboard")

# --- Performance Metrics ---
df_metrics = pd.DataFrame(st.session_state.kpis)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(label="On-Time Delivery Rate", value=f"{df_metrics['on_time_delivery_rate'].iloc[-1]:.2f}%")
with col2:
    st.metric(label="Supply Chain Cost", value=f"${df_metrics['supply_chain_cost'].iloc[-1]:.2f}")
with col3:
    st.metric(label="Inventory Days", value=f"{df_metrics['inventory_days_of_supply'].iloc[-1]:.2f} days")
with col4:
    st.metric(label="Risk Exposure Score", value=f"{df_metrics['risk_exposure_score'].iloc[-1]:.2f}")
with col5:
    st.metric(label="Active Agent Boost", value=st.session_state.agent_boost if st.session_state.agent_boost else "None")

st.markdown("---")

st.subheader("Agent & Event Log")

# --- Central Log (Scrollable Container) ---
log_container = st.container(height=300)
with log_container:
    # Display the log in reverse chronological order
    for log_entry in reversed(st.session_state.simulation_log):
        agent = log_entry['agent']
        message = log_entry['message']
        level = log_entry['level']
        
        # Determine a simple text avatar based on the agent name
        avatar_text = ""
        if agent == "System":
            avatar_text = "System"
        elif agent == "Human":
            avatar_text = "Human"
        elif agent == "DemandForecastAgent":
            avatar_text = "Demand"
        elif agent == "ProcurementAgent":
            avatar_text = "Procurement"
        elif agent == "LogisticsAgent":
            avatar_text = "Logistics"
        else:
            avatar_text = "Unknown"

        with st.chat_message(name=agent, avatar=avatar_text):
            if level == "warning":
                st.warning(message)
            elif level == "error":
                st.error(message)
            elif level == "success":
                st.success(message)
            elif level == "critical":
                st.exception(Exception(message))
            else:
                st.info(message)
st.markdown("---")

st.subheader("Performance Trends Over Time")

# --- Charts ---
df_metrics = pd.DataFrame(st.session_state.kpis)
fig_metrics = px.line(df_metrics, x='day', y=df_metrics.columns[1:],
                      title='Supply Chain KPIs Over Time', markers=True)
fig_metrics.update_layout(yaxis_title="Value", legend_title="KPIs")
st.plotly_chart(fig_metrics, use_container_width=True)

# --- The Simulation Loop ---
if st.session_state.simulation_running:
    time.sleep(1) # Simulates a day passing every second
    advance_day()
    st.experimental_rerun()
