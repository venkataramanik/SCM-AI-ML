import streamlit as st
import pandas as pd
import random
import plotly.express as px
import time

# --- Configuration ---
st.set_page_config(
    page_title="Advanced Agentic SCM Simulation",
    layout="wide",
)

# --- State Initialization ---
def initialize_state():
    """Initializes or resets all session state variables."""
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
            'learning': {
                'Supplier Delay': 0,
                'Labor Strike': 0,
                'Logistics Network Congestion': 0,
                'Port Closure': 0,
            }
        }
    if 'human_intervention_needed' not in st.session_state:
        st.session_state.human_intervention_needed = False
    if 'agent_boost' not in st.session_state:
        st.session_state.agent_boost = None
    if 'boost_counter' not in st.session_state:
        st.session_state.boost_counter = 0

initialize_state()

# --- Constants ---
ON_TIME_DECAY_RATE = -0.1
COST_GROWTH_RATE = 10
INVENTORY_GROWTH_RATE = 0.2
RISK_GROWTH_RATE = 0.3

BASE_AGENT_IMPACTS = {
    'demand_forecast': {'on_time': (4.0, 7.0), 'cost': (-40.0, -30.0), 'inventory': (-8.0, -5.0)},
    'procurement': {'on_time': (8.0, 12.0), 'cost': (-50.0, -35.0), 'risk': (-15.0, -10.0)},
    'logistics': {'on_time': (8.0, 12.0), 'cost': (-35.0, -20.0), 'inventory': (-8.0, -5.0)},
}

MARKET_EVENTS = [
    {'name': 'Supplier Delay', 'impact': {'on_time': -10, 'cost': 80, 'risk': 4}, 'chance': 0.1, 'trigger': 'procurement'},
    {'name': 'Unexpected Demand Spike', 'impact': {'on_time': -8, 'cost': 40, 'inventory': 5}, 'chance': 0.1, 'trigger': 'demand_forecast'},
    {'name': 'Logistics Network Congestion', 'impact': {'on_time': -10, 'cost': 25}, 'chance': 0.15, 'trigger': 'logistics'},
    {'name': 'Labor Strike', 'impact': {'on_time': -20, 'cost': 150, 'risk': 10}, 'chance': 0.05, 'trigger': 'procurement'},
    {'name': 'Port Closure', 'impact': {'on_time': -25, 'cost': 200, 'risk': 12}, 'chance': 0.03, 'trigger': 'logistics'},
    {'name': 'Global Supply Chain Freeze', 'impact': {'on_time': -50, 'cost': 500, 'risk': 30}, 'chance': 0.01, 'trigger': 'human_intervention'},
]

BOOST_CHANCE = 0.05
BOOST_DURATION = 3
BOOST_MULTIPLIER = 1.5

# --- Helper Functions ---
def add_log(agent, message, level='info'):
    """Adds a timestamped message to the simulation log."""
    st.session_state.simulation_log.append({
        'day': st.session_state.day,
        'agent': agent,
        'message': message,
        'level': level,
        'active_boost': st.session_state.agent_boost if st.session_state.agent_boost else 'None'
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

    trigger_agent_boost()

    on_time_change = ON_TIME_DECAY_RATE
    cost_change = COST_GROWTH_RATE
    inventory_change = INVENTORY_GROWTH_RATE
    risk_change = RISK_GROWTH_RATE
    
    for event in MARKET_EVENTS:
        if random.random() < event['chance']:
            event_name = event['name']
            event_impact = event['impact']
            
            event_message = f"**Market Event:** {event_name} occurred!"
            add_log("System", event_message, 'warning')
            
            st.session_state.message_bus[event['trigger']] = {'type': 'event', 'data': event_name, 'impact': event_impact}

            on_time_change += event_impact.get('on_time', 0)
            cost_change += event_impact.get('cost', 0)
            inventory_change += event_impact.get('inventory', 0)
            risk_change += event_impact.get('risk', 0)
    
    agent_outputs = run_agents()

    for agent, impacts in agent_outputs.items():
        if impacts:
            on_time_change += impacts.get('on_time', 0)
            cost_change += impacts.get('cost', 0)
            inventory_change += impacts.get('inventory', 0)
            risk_change += impacts.get('risk', 0)
    
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
    st.session_state.kpis['inventory_days_of_supply'].append(max(0, current_inventory)))
    st.session_state.kpis['risk_exposure_score'].append(min(100, max(0, current_risk)))

    add_log("System", f"Day {st.session_state.day}: KPI changes calculated.", 'info')

def run_agents():
    """Orchestrates agent interactions for the current day."""
    impacts = {}
    
    if 'human_intervention' in st.session_state.message_bus:
        message = st.session_state.message_bus.pop('human_intervention')
        add_log("System", f"Agents unable to handle event: '{message['data']}'. Requiring human takeover.", 'critical')
        st.session_state.human_intervention_needed = True
        st.session_state.simulation_running = False
        return {}
    
    forecast_impacts = run_demand_forecast_agent()
    if forecast_impacts:
        impacts['demand_forecast'] = forecast_impacts

    procurement_impacts = run_procurement_agent()
    if procurement_impacts:
        impacts['procurement'] = procurement_impacts

    logistics_impacts = run_logistics_agent()
    if logistics_impacts:
        impacts['logistics'] = logistics_impacts

    return impacts

def run_demand_forecast_agent():
    """Simulates the Demand Forecasting Agent's actions."""
    impacts = {}
    impacts['on_time'] = random.uniform(0.5, 1.0)
    impacts['cost'] = random.uniform(-10.0, -5.0)
    
    if 'demand_forecast' in st.session_state.message_bus:
        message = st.session_state.message_bus.pop('demand_forecast')
        if message['type'] == 'event' and message['data'] == 'Unexpected Demand Spike':
            add_log("DemandForecastAgent", "Detected unexpected demand spike. Adjusting forecast and communicating to Procurement.", 'warning')
            st.session_state.agent_states['demand_forecast']['current_forecast'] = 150
            st.session_state.message_bus['procurement'] = {'type': 'forecast_update', 'data': 150}
            
            impacts['on_time'] += random.uniform(*BASE_AGENT_IMPACTS['demand_forecast']['on_time'])
            impacts['cost'] += random.uniform(*BASE_AGENT_IMPACTS['demand_forecast']['cost'])
            impacts['inventory'] = random.uniform(*BASE_AGENT_IMPACTS['demand_forecast']['inventory'])
    
    if st.session_state.day % 5 == 0:
        new_forecast = random.randint(90, 110)
        st.session_state.agent_states['demand_forecast']['current_forecast'] = new_forecast
        st.session_state.message_bus['procurement'] = {'type': 'forecast_update', 'data': new_forecast}
        add_log("DemandForecastAgent", f"Published new forecast: {new_forecast} units.")
    
    return impacts

def run_procurement_agent():
    """Simulates the Procurement Agent's actions."""
    impacts = {}
    impacts['on_time'] = random.uniform(0.5, 1.0)
    impacts['cost'] = random.uniform(-10.0, -5.0)

    if 'procurement' in st.session_state.message_bus:
        message = st.session_state.message_bus.pop('procurement')
        event_name = message['data']

        if event_name in ['Supplier Delay', 'Labor Strike']:
            add_log("ProcurementAgent", f"Received alert: {event_name}. Searching for alternative suppliers and pre-ordering materials.", 'warning')
            
            if random.random() < (100 - st.session_state.kpis['risk_exposure_score'][-1]) / 100.0:
                learning_bonus = st.session_state.agent_states['learning'].get(event_name, 0) * 0.5
                boost_multiplier = 1
                if st.session_state.agent_boost == "Procurement Boost":
                    boost_multiplier = BOOST_MULTIPLIER
                    add_log("ProcurementAgent", "Applying Procurement Boost to mitigate event.", 'info')

                add_log("ProcurementAgent", f"Successfully found an alternative supplier. Learning improved for '{event_name}'.", 'success')
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
            impacts['cost'] += random.uniform(-5.0, -2.0)
            impacts['inventory'] = random.uniform(-1.0, 1.0)
    
    return impacts

def run_logistics_agent():
    """Simulates the Logistics Agent's actions."""
    impacts = {}
    impacts['on_time'] = random.uniform(0.5, 1.0)
    impacts['cost'] = random.uniform(-10.0, -5.0)
    
    if 'logistics' in st.session_state.message_bus:
        message = st.session_state.message_bus.pop('logistics')
        event_name = message['data']
        
        if event_name in ['Logistics Network Congestion', 'Port Closure']:
            add_log("LogisticsAgent", f"Detected {event_name}. Rerouting shipments and seeking alternative transport.", 'warning')
            
            if random.random() < 0.7:
                learning_bonus = st.session_state.agent_states['learning'].get(event_name, 0) * 0.5
                boost_multiplier = 1
                if st.session_state.agent_boost == "Logistics Boost":
                    boost_multiplier = BOOST_MULTIPLIER
                    add_log("LogisticsAgent", "Applying Logistics Boost to mitigate event.", 'info')

                add_log("LogisticsAgent", f"Rerouting successful. Learning improved for '{event_name}'.", 'success')
                if event_name in st.session_state.agent_states['learning']:
                    st.session_state.agent_states['learning'][event_name] += 1

                impacts['on_time'] += random.uniform((BASE_AGENT_IMPACTS['logistics']['on_time'][0] + learning_bonus) * boost_multiplier, (BASE_AGENT_IMPACTS['logistics']['on_time'][1] + learning_bonus) * boost_multiplier)
                impacts['cost'] += random.uniform(BASE_AGENT_IMPACTS['logistics']['cost'][0], BASE_AGENT_IMPACTS['logistics']['cost'][1])
            else:
                add_log("LogisticsAgent", f"Failed to find an efficient alternative route. Delays expected.", 'error')
    
    return impacts

def start_simulation():
    if not st.session_state.human_intervention_needed:
        st.session_state.simulation_running = True

def stop_simulation():
    st.session_state.simulation_running = False

def human_intervene():
    """Action for human intervention button. Applies a large fix and resets the state."""
    st.session_state.kpis['on_time_delivery_rate'][-1] += 50
    st.session_state.kpis['supply_chain_cost'][-1] -= 300
    st.session_state.human_intervention_needed = False
    st.session_state.simulation_running = False
    add_log("Human", "Intervened to solve the black swan event. The supply chain is recovering.", 'success')

def reset_simulation():
    """Resets all simulation state to its initial values."""
    st.session_state.clear()
    initialize_state()
    st.rerun()

# --- UI Layout ---
st.title("Advanced Agentic Supply Chain Simulation")
st.markdown("This simulation models an **Agentic Framework** with autonomous, interacting agents. They work proactively to manage key KPIs and react to dynamic market events.")
st.markdown("---")
st.markdown("""
### The Agent Boost Mechanism

The **Agent Boost** is a special, temporary state that enhances an agent's ability to respond to a supply chain disruption. In this simulation, a boost event (either a **Procurement Boost** or a **Logistics Boost**) has a small chance of being randomly triggered and will last for a few days. When a boosted agent successfully mitigates a market event during this time, its positive impact on the key performance indicators (KPIs) is significantly multiplied. This simulates a temporary improvement in an agent's efficiency or access to better resources.

### How to Run the Simulation

To start, simply click the **Start Simulation** button in the control panel on the left. The simulation will advance one "day" at a time, with agents taking action and events potentially occurring. You can observe the performance trends over time and the detailed agent logs below. If a catastrophic event occurs, the simulation will pause and require a human intervention.
""")
st.markdown("---")


# --- Sidebar ---
st.sidebar.header("Control Panel")
if st.session_state.simulation_running:
    st.sidebar.button("Stop Simulation", on_click=stop_simulation)
elif st.session_state.human_intervention_needed:
    st.sidebar.warning("A major crisis requires human intervention!")
    st.sidebar.button("Human Intervention: Resolve Crisis", on_click=human_intervene)
else:
    st.sidebar.button("Start Simulation", on_click=start_simulation)
st.sidebar.button("Reset Simulation", on_click=reset_simulation)
st.sidebar.markdown("---")
st.sidebar.header("Agentic Framework")
st.sidebar.markdown("""
- **Demand Forecast Agent:** Predicts demand to optimize inventory.
- **Procurement Agent:** Manages sourcing, orders, and supplier risk.
- **Logistics Agent:** Optimizes shipping and delivery routes.
- **Human:** Steps in for unprecedented, "black swan" events.
""")

st.header("Simulation Dashboard")

# --- Performance Metrics and Graph Display ---
df_metrics = pd.DataFrame(st.session_state.kpis)
latest_kpis = df_metrics.iloc[-1]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="On-Time Delivery Rate", value=f"{latest_kpis['on_time_delivery_rate']:.2f}%")
with col2:
    st.metric(label="Supply Chain Cost", value=f"${latest_kpis['supply_chain_cost']:.2f}")
with col3:
    st.metric(label="Inventory Days", value=f"{latest_kpis['inventory_days_of_supply']:.2f} days")
with col4:
    st.metric(label="Risk Exposure Score", value=f"{latest_kpis['risk_exposure_score']:.2f}")

st.markdown(f"**Active Agent Boost:** {st.session_state.agent_boost if st.session_state.agent_boost else 'None'}")
    
st.markdown("---")

st.subheader("Agent & Event Log")

# --- Central Log (Scrollable Container with DataFrame) ---
st.markdown("""
<style>
.stDataFrame table {
    table-layout: fixed;
    width: 100%;
}
.stDataFrame th:nth-child(4), .stDataFrame td:nth-child(4) {
    word-wrap: break-word;
    white-space: normal;
    width: 40%;
}
.stDataFrame th:nth-child(5), .stDataFrame td:nth-child(5) {
    word-wrap: break-word;
    white-space: normal;
    width: 20%;
}
.stDataFrame th:nth-child(1), .stDataFrame td:nth-child(1) {
    width: 5%;
}
.stDataFrame th:nth-child(2), .stDataFrame td:nth-child(2) {
    width: 15%;
}
.stDataFrame th:nth-child(3), .stDataFrame td:nth-child(3) {
    width: 15%;
}
</style>
""", unsafe_allow_html=True)

log_container = st.container(height=300)
with log_container:
    if len(st.session_state.simulation_log) > 0:
        df_log = pd.DataFrame(st.session_state.simulation_log)
        df_log = df_log.rename(columns={'active_boost': 'Active Boost'})
        st.dataframe(df_log.sort_values(by='day', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Simulation log is currently empty.")
st.markdown("---")

st.subheader("Performance Trends Over Time")

# --- The Chart Display Logic (Always renders from the latest data) ---
# FIX: Explicitly cast the DataFrame columns to a list for Plotly.
fig_metrics = px.line(df_metrics, x='day', y=list(df_metrics.columns[1:]),
                      title='Supply Chain KPIs Over Time', markers=True)
fig_metrics.update_layout(yaxis_title="Value", legend_title="KPIs")
st.plotly_chart(fig_metrics, use_container_width=True)


# --- The Simulation Loop ---
if st.session_state.simulation_running:
    time.sleep(1)
    advance_day()
    st.rerun()
