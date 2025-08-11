import streamlit as st
import pandas as pd
from datetime import datetime
import random
import time

# --- Configuration ---
st.set_page_config(
    page_title="SCM Persona Simulation",
    page_icon="👥",
    layout="wide",
)

# --- State Initialization ---
if 'simulation_log' not in st.session_state:
    st.session_state.simulation_log = []
if 'inventory' not in st.session_state:
    st.session_state.inventory = {'raw_materials': 100, 'finished_goods': 50}
if 'orders' not in st.session_state:
    st.session_state.orders = []
if 'transportation' not in st.session_state:
    st.session_state.transportation = {'deliveries_sent': 0, 'on_time_deliveries': 0, 'total_cost': 0}
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = {
        'last_forecast': 60,
        'actual_demand': 55,
        'forecast_accuracy': 91.67
    }
if 'warehouse_metrics' not in st.session_state:
    st.session_state.warehouse_metrics = {'orders_picked': 0, 'picking_rate': 0}

# --- Constants ---
AVG_ORDER_SIZE = 5
AVG_PICKING_TIME_PER_UNIT = 0.5 # minutes
TRANSPORT_COST_PER_UNIT = 2
ON_TIME_PROBABILITY = 0.9

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

def reset_simulation():
    """Resets all simulation state to its initial values and re-runs the app."""
    st.session_state.clear()
    st.experimental_rerun()

# --- Persona-Specific Logic ---
def run_demand_planner_task():
    """Simulates the Demand Planner's daily task of generating a new forecast."""
    add_log("Demand Planner", "Generating new forecast...", 'info')
    
    # Generate new random actual demand
    new_actual_demand = random.randint(40, 70)
    
    # Generate a new forecast with some random error
    new_forecast = new_actual_demand + random.randint(-5, 5)
    
    # Calculate forecast accuracy
    forecast_accuracy = (1 - abs(new_actual_demand - new_forecast) / new_actual_demand) * 100
    
    st.session_state.forecast_data = {
        'last_forecast': new_forecast,
        'actual_demand': new_actual_demand,
        'forecast_accuracy': round(forecast_accuracy, 2)
    }
    
    add_log("Demand Planner", f"New forecast generated: {new_forecast} units. Actual demand: {new_actual_demand} units. Accuracy: {st.session_state.forecast_data['forecast_accuracy']}%", 'success')

def run_warehouse_manager_task():
    """Simulates the Warehouse Manager fulfilling orders and receiving goods."""
    add_log("Warehouse Manager", "Receiving new raw materials...", 'info')
    raw_material_received = random.randint(50, 100)
    st.session_state.inventory['raw_materials'] += raw_material_received
    add_log("Warehouse Manager", f"{raw_material_received} new raw materials received. Inventory updated.", 'success')

    # Fulfill a random number of orders
    orders_to_fulfill = random.randint(1, 3)
    orders_fulfilled = 0
    for _ in range(orders_to_fulfill):
        if st.session_state.inventory['finished_goods'] >= AVG_ORDER_SIZE:
            st.session_state.inventory['finished_goods'] -= AVG_ORDER_SIZE
            st.session_state.orders.append({'id': len(st.session_state.orders) + 1, 'quantity': AVG_ORDER_SIZE, 'status': 'Awaiting Delivery'})
            orders_fulfilled += 1
            add_log("Warehouse Manager", f"Order #{len(st.session_state.orders)} fulfilled ({AVG_ORDER_SIZE} units). Ready for transport.", 'success')
        else:
            add_log("Warehouse Manager", "Not enough finished goods to fulfill an order.", 'warning')

    if orders_fulfilled > 0:
        picking_rate = (orders_fulfilled * AVG_ORDER_SIZE) / (orders_fulfilled * AVG_PICKING_TIME_PER_UNIT) # units per minute
        st.session_state.warehouse_metrics['orders_picked'] += orders_fulfilled
        st.session_state.warehouse_metrics['picking_rate'] = round(picking_rate * 60, 2) # units per hour
        add_log("Warehouse Manager", f"{orders_fulfilled} orders picked. Picking rate: {st.session_state.warehouse_metrics['picking_rate']} units/hour.", 'info')

def run_transport_manager_task():
    """Simulates the Transport Manager optimizing routes and delivering goods."""
    add_log("Transport Manager", "Checking for orders to deliver...", 'info')
    
    orders_to_deliver = [o for o in st.session_state.orders if o['status'] == 'Awaiting Delivery']
    if not orders_to_deliver:
        add_log("Transport Manager", "No orders waiting for delivery.", 'warning')
        return

    add_log("Transport Manager", f"Optimizing routes for {len(orders_to_deliver)} orders...", 'info')
    
    for order in orders_to_deliver:
        # Simulate on-time delivery with a random probability
        if random.random() < ON_TIME_PROBABILITY:
            st.session_state.transportation['on_time_deliveries'] += 1
            add_log("Transport Manager", f"Order #{order['id']} delivered successfully and on-time!", 'success')
        else:
            add_log("Transport Manager", f"Order #{order['id']} delivered late.", 'error')
        
        st.session_state.transportation['deliveries_sent'] += 1
        st.session_state.transportation['total_cost'] += order['quantity'] * TRANSPORT_COST_PER_UNIT
        order['status'] = 'Delivered'
    
    add_log("Transport Manager", f"{len(orders_to_deliver)} deliveries dispatched. Total cost updated.", 'success')

# --- UI Layout ---
st.title("👥 SCM Persona Simulation Dashboard")
st.markdown("A demonstration of key supply chain personas, their priorities, and their daily operational tasks.")

# --- Persona Tabs ---
tab1, tab2, tab3 = st.tabs(["🔮 Demand Planner", "📦 Warehouse Manager", "🚚 Transport Manager"])

with tab1:
    st.header("🔮 Demand Planner Dashboard")
    st.subheader("Key Performance Indicators")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Forecast Accuracy", value=f"{st.session_state.forecast_data['forecast_accuracy']}%")
    with col2:
        st.metric(label="Inventory Turns", value=round(st.session_state.forecast_data['actual_demand'] / (st.session_state.inventory['finished_goods'] + st.session_state.inventory['raw_materials']), 2))
    st.markdown("---")
    st.subheader("Daily Task")
    st.button("Generate New Forecast", on_click=run_demand_planner_task)
    st.info("The Demand Planner's main task is to predict future demand. This impacts inventory levels and production plans.")

with tab2:
    st.header("📦 Warehouse Manager Dashboard")
    st.subheader("Key Performance Indicators")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Order Picking Rate", value=f"{st.session_state.warehouse_metrics['picking_rate']} units/hour")
    with col2:
        st.metric(label="Finished Goods Inventory", value=st.session_state.inventory['finished_goods'])
    st.markdown("---")
    st.subheader("Daily Task")
    st.button("Fulfill Orders & Receive Goods", on_click=run_warehouse_manager_task)
    st.info("The Warehouse Manager focuses on the physical movement of goods, ensuring orders are filled and inventory is accurate.")

with tab3:
    st.header("🚚 Transport Manager Dashboard")
    st.subheader("Key Performance Indicators")
    col1, col2 = st.columns(2)
    with col1:
        on_time_rate = (st.session_state.transportation['on_time_deliveries'] / st.session_state.transportation['deliveries_sent']) * 100 if st.session_state.transportation['deliveries_sent'] > 0 else 0
        st.metric(label="On-Time Delivery Rate", value=f"{round(on_time_rate, 2)}%")
    with col2:
        cost_per_unit = st.session_state.transportation['total_cost'] / (st.session_state.transportation['deliveries_sent'] * AVG_ORDER_SIZE) if st.session_state.transportation['deliveries_sent'] > 0 else 0
        st.metric(label="Transport Cost per Unit", value=f"${round(cost_per_unit, 2)}")
    st.markdown("---")
    st.subheader("Daily Task")
    st.button("Optimize Routes & Deliver Shipments", on_click=run_transport_manager_task)
    st.info("The Transport Manager is responsible for moving goods from the warehouse to the customer, balancing speed and cost.")

st.markdown("---")

# --- Central Log ---
st.subheader("System Log")
col_log, col_reset = st.columns([4, 1])
with col_reset:
    st.button("🔄 Reset Simulation", on_click=reset_simulation)
with col_log:
    if st.session_state.simulation_log:
        log_df = pd.DataFrame(st.session_state.simulation_log)
        st.dataframe(log_df.set_index('timestamp'), use_container_width=True, height=250)
    else:
        st.info("Click on a Persona's 'Daily Task' button to start the simulation!")
