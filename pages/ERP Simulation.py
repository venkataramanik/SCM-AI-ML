import streamlit as st
import pandas as pd
from datetime import datetime
import random
import time

# --- Configuration ---
st.set_page_config(
    page_title="Integrated SCM Simulation",
    page_icon="🔗",
    layout="wide",
)

# --- State Initialization ---
if 'simulation_log' not in st.session_state:
    st.session_state.simulation_log = []
if 'crm_orders' not in st.session_state:
    st.session_state.crm_orders = []
if 'erp_production_queue' not in st.session_state:
    st.session_state.erp_production_queue = []
if 'wms_inventory' not in st.session_state:
    st.session_state.wms_inventory = {'raw_materials': 100, 'finished_goods': 20}
if 'demand_forecast' not in st.session_state:
    st.session_state.demand_forecast = None
if 'tms_deliveries' not in st.session_state:
    st.session_state.tms_deliveries = []
if 'finance' not in st.session_state:
    st.session_state.finance = {'cash': 10000, 'revenue': 0, 'cogs': 0}

# --- Constants ---
ORDER_SIZE_RANGE = (5, 15)
RAW_MATERIAL_COST = 5
FINISHED_GOOD_PRICE = 25

# --- Helper Functions ---
def add_log(system, message, level='info'):
    """Adds a timestamped message to the simulation log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.simulation_log.append({
        'timestamp': timestamp,
        'system': system,
        'message': message,
        'level': level,
    })

def reset_simulation():
    """Resets all simulation state to its initial values and re-runs the app."""
    st.session_state.clear()
    st.experimental_rerun()

# --- Simulation Logic Functions ---
def run_demand_forecasting():
    """Simulates generating a demand forecast for the next period."""
    add_log("Demand Forecasting", "Generating demand forecast...", 'info')
    forecast_value = random.randint(30, 80)
    st.session_state.demand_forecast = forecast_value
    add_log("Demand Forecasting", f"Forecast for next period: {forecast_value} units. Data sent to ERP.", 'success')

def run_crm():
    """Simulates a customer placing a new order via the CRM."""
    order_id = len(st.session_state.crm_orders) + 1
    order_quantity = random.randint(*ORDER_SIZE_RANGE)
    new_order = {
        'order_id': order_id,
        'quantity': order_quantity,
        'status': 'New',
        'revenue': order_quantity * FINISHED_GOOD_PRICE,
    }
    st.session_state.crm_orders.append(new_order)
    add_log("CRM", f"New order #{order_id} for {order_quantity} units received. Order sent to ERP.", 'success')

def run_erp():
    """ERP acts as the central brain, processing orders and managing flow."""
    add_log("ERP", "Checking for new orders from CRM...", 'info')
    
    # Check for new CRM orders
    new_orders_to_process = [o for o in st.session_state.crm_orders if o['status'] == 'New']
    for order in new_orders_to_process:
        add_log("ERP", f"Processing order #{order['order_id']} from CRM. Checking WMS for finished goods.", 'info')
        
        if st.session_state.wms_inventory['finished_goods'] >= order['quantity']:
            order['status'] = 'Ready for Delivery'
            add_log("ERP", f"Order #{order['order_id']} has finished goods in stock. Sending shipping order to TMS.", 'success')
        else:
            add_log("ERP", f"Not enough finished goods for order #{order['order_id']}. Checking WMS for raw materials for production.", 'warning')
            required_raw_materials = order['quantity']
            
            if st.session_state.wms_inventory['raw_materials'] >= required_raw_materials:
                order['status'] = 'In Production Queue'
                st.session_state.erp_production_queue.append(order)
                add_log("ERP", f"Raw materials available. Sending production order for #{order['order_id']} to WMS.", 'success')
            else:
                add_log("ERP", f"Insufficient raw materials for order #{order['order_id']}. Waiting for raw materials.", 'error')
                
    # Check for completed production
    completed_production_orders = [o for o in st.session_state.erp_production_queue if o['status'] == 'Production Complete']
    for order in completed_production_orders:
        order['status'] = 'Ready for Delivery'
        add_log("ERP", f"Production for order #{order['order_id']} complete. Sending shipping order to TMS.", 'success')

def run_wms():
    """WMS manages inventory and executes production orders from the ERP."""
    add_log("WMS", "Checking for new production orders from ERP...", 'info')

    # Execute production orders
    production_orders = [o for o in st.session_state.erp_production_queue if o['status'] == 'In Production Queue']
    if not production_orders:
        add_log("WMS", "No new production orders to process.", 'info')
        return

    for order in production_orders:
        required_materials = order['quantity']
        if st.session_state.wms_inventory['raw_materials'] >= required_materials:
            add_log("WMS", f"Processing production order for #{order['order_id']}. Pulling {required_materials} raw materials.", 'success')
            st.session_state.wms_inventory['raw_materials'] -= required_materials
            
            # Simulate production time
            time.sleep(0.5) 
            st.session_state.wms_inventory['finished_goods'] += required_materials
            order['status'] = 'Production Complete'
            add_log("WMS", f"Production for order #{order['order_id']} complete. {required_materials} finished goods added to inventory.", 'success')
        else:
            add_log("WMS", f"WMS Error: Not enough raw materials to fulfill production order #{order['order_id']}.", 'error')
            
def run_tms():
    """TMS handles the logistics and delivery of finished goods."""
    add_log("TMS", "Checking for new shipping orders from ERP...", 'info')

    shipping_orders = [o for o in st.session_state.crm_orders if o['status'] == 'Ready for Delivery']
    if not shipping_orders:
        add_log("TMS", "No new shipping orders to process.", 'info')
        return

    for order in shipping_orders:
        if st.session_state.wms_inventory['finished_goods'] >= order['quantity']:
            add_log("TMS", f"Processing shipping order for #{order['order_id']}. Dispatching truck to customer.", 'success')
            st.session_state.wms_inventory['finished_goods'] -= order['quantity']
            
            # Update finance
            st.session_state.finance['cash'] += order['revenue']
            st.session_state.finance['revenue'] += order['revenue']

            order['status'] = 'Delivered'
            st.session_state.tms_deliveries.append(order)
            add_log("TMS", f"Order #{order['order_id']} successfully delivered. Invoice closed.", 'success')
        else:
            add_log("TMS", f"TMS Error: Not enough finished goods in WMS for shipping order #{order['order_id']}.", 'error')
            order['status'] = 'ERP-Pending-Inventory'


# --- UI Layout ---
st.title("🔗 Integrated SCM Simulation Dashboard")
st.markdown("A demonstration of how CRM, ERP, Demand Forecasting, WMS, and TMS systems interact.")

# --- Key Metrics ---
st.subheader("Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(label="Cash", value=f"${st.session_state.finance['cash']:.2f}")
with col2:
    st.metric(label="Total Revenue", value=f"${st.session_state.finance['revenue']:.2f}")
with col3:
    st.metric(label="Open Orders", value=len([o for o in st.session_state.crm_orders if o['status'] != 'Delivered']))
with col4:
    st.metric(label="Raw Materials", value=st.session_state.wms_inventory['raw_materials'])
with col5:
    st.metric(label="Finished Goods", value=st.session_state.wms_inventory['finished_goods'])

st.markdown("---")

# --- Action Buttons ---
st.subheader("System Actions")
col_actions = st.columns(6)
with col_actions[0]:
    st.button("🔮 Forecast Demand", on_click=run_demand_forecasting)
with col_actions[1]:
    st.button("🙋 New CRM Order", on_click=run_crm)
with col_actions[2]:
    st.button("🧠 Run ERP", on_click=run_erp)
with col_actions[3]:
    st.button("📦 Run WMS", on_click=run_wms)
with col_actions[4]:
    st.button("🚚 Run TMS", on_click=run_tms)
with col_actions[5]:
    st.button("🔄 Reset", on_click=reset_simulation)

st.markdown("---")

# --- Log and Dataframes ---
st.subheader("Simulation Log")
log_df = pd.DataFrame(st.session_state.simulation_log)
if not log_df.empty:
    st.dataframe(log_df.set_index('timestamp'), use_container_width=True)

st.subheader("Current Order Status")
if st.session_state.crm_orders:
    orders_df = pd.DataFrame(st.session_state.crm_orders)
    st.dataframe(orders_df, use_container_width=True)
else:
    st.info("No orders have been placed yet.")

