import streamlit as st
import time
from datetime import datetime

# --- Configuration ---
st.set_page_config(
    page_title="SCOR Flow Simulation",
    page_icon="📦",
    layout="wide",
)

# --- State Management ---
if 'inventory' not in st.session_state:
    st.session_state.inventory = {
        'raw_materials': 50,
        'finished_goods': 0,
    }
if 'orders' not in st.session_state:
    st.session_state.orders = []
if 'finance' not in st.session_state:
    st.session_state.finance = {
        'cash': 10000,
        'revenue': 0,
        'cost_of_goods': 0,
    }
if 'simulation_log' not in st.session_state:
    st.session_state.simulation_log = []

# --- Constants ---
PRODUCT_COST = 20
PRODUCT_PRICE = 50
RAW_MATERIAL_COST = 10
RAW_MATERIAL_QUANTITY = 100
ORDER_QUANTITY = 5

# --- Helper Functions ---
def add_log(message, level='info'):
    """Adds a timestamped message to the simulation log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.simulation_log.append({
        'timestamp': timestamp,
        'message': message,
        'level': level,
    })
    st.rerun()

def place_order():
    """Simulates a customer placing a new order."""
    new_order = {
        'id': len(st.session_state.orders) + 1,
        'quantity': ORDER_QUANTITY,
        'status': 'New',
        'revenue': ORDER_QUANTITY * PRODUCT_PRICE,
    }
    st.session_state.orders.append(new_order)
    add_log(f"New customer order #{new_order['id']} placed for {new_order['quantity']} units.", 'success')

def source_materials():
    """Simulates sourcing new raw materials."""
    add_log("Sourcing new raw materials...", 'info')
    
    st.session_state.inventory['raw_materials'] += RAW_MATERIAL_QUANTITY
    st.session_state.finance['cash'] -= (RAW_MATERIAL_QUANTITY * RAW_MATERIAL_COST)
    st.session_state.finance['cost_of_goods'] += (RAW_MATERIAL_QUANTITY * RAW_MATERIAL_COST)
    
    add_log(f"Received {RAW_MATERIAL_QUANTITY} raw materials.", 'success')

def make_goods():
    """Simulates producing goods for a pending order."""
    pending_order_idx = next((i for i, o in enumerate(st.session_state.orders) if o['status'] == 'New'), None)
    
    if pending_order_idx is None:
        add_log("No new orders to produce.", 'warning')
        return

    order = st.session_state.orders[pending_order_idx]
    required_materials = order['quantity']

    if st.session_state.inventory['raw_materials'] < required_materials:
        add_log(f"Not enough raw materials to produce order #{order['id']}. Sourcing materials first.", 'error')
        return

    add_log(f"Starting production for order #{order['id']}...", 'info')
    
    st.session_state.inventory['raw_materials'] -= required_materials
    st.session_state.inventory['finished_goods'] += required_materials
    st.session_state.orders[pending_order_idx]['status'] = 'Production Complete'

    add_log(f"Production complete for order #{order['id']}. {required_materials} finished goods added to inventory.", 'success')

def deliver_order():
    """Simulates shipping a completed order to the customer."""
    completed_order_idx = next((i for i, o in enumerate(st.session_state.orders) if o['status'] == 'Production Complete'), None)
    
    if completed_order_idx is None:
        add_log("No completed orders to deliver.", 'warning')
        return

    order = st.session_state.orders[completed_order_idx]
    required_goods = order['quantity']

    if st.session_state.inventory['finished_goods'] < required_goods:
        add_log(f"Not enough finished goods in stock to ship order #{order['id']}.", 'error')
        return

    add_log(f"Shipping order #{order['id']}...", 'info')

    st.session_state.inventory['finished_goods'] -= required_goods
    st.session_state.finance['cash'] += order['revenue']
    st.session_state.finance['revenue'] += order['revenue']
    st.session_state.orders[completed_order_idx]['status'] = 'Delivered'
    
    add_log(f"Order #{order['id']} has been successfully delivered and paid.", 'success')
    
def reset_simulation():
    """Resets the entire simulation to its initial state."""
    st.session_state.inventory = {
        'raw_materials': 50,
        'finished_goods': 0,
    }
    st.session_state.orders = []
    st.session_state.finance = {
        'cash': 10000,
        'revenue': 0,
        'cost_of_goods': 0,
    }
    st.session_state.simulation_log = []
    add_log("Simulation reset.", 'info')

# --- UI Layout ---
st.title("📦 SCOR Flow Simulation")

st.markdown("""
A simple simulation of the core SCOR model (Plan, Source, Make, Deliver) as an ERP system.
Use the buttons below to trigger each step of the process.
""")

# --- KPI Dashboard ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Open Orders", value=len([o for o in st.session_state.orders if o['status'] != 'Delivered']))
with col2:
    st.metric(label="Cash", value=f"${st.session_state.finance['cash']:.2f}")
with col3:
    st.metric(label="Raw Materials", value=st.session_state.inventory['raw_materials'])
with col4:
    st.metric(label="Finished Goods", value=st.session_state.inventory['finished_goods'])

st.markdown("---")

# --- Action Buttons ---
st.header("SCOR Actions")
col_b1, col_b2, col_b3, col_b4 = st.columns(4)

with col_b1:
    st.button("📋 Plan: Place Order", on_click=place_order)
with col_b2:
    st.button("📦 Source: Get Materials", on_click=source_materials)
with col_b3:
    st.button("🏭 Make: Produce Goods", on_click=make_goods)
with col_b4:
    st.button("🚚 Deliver: Ship Orders", on_click=deliver_order)

st.markdown("---")

# --- Main Content Area (Orders and Log) ---
col_orders, col_log = st.columns(2)

with col_orders:
    st.subheader("Order Status")
    if st.session_state.orders:
        orders_df = st.dataframe(st.session_state.orders, use_container_width=True)
    else:
        st.write("No orders placed yet.")
    
    if st.button("🔄 Reset Simulation"):
        reset_simulation()

with col_log:
    st.subheader("Simulation Log")
    for entry in reversed(st.session_state.simulation_log):
        if entry['level'] == 'info':
            st.info(f"**{entry['timestamp']}** - {entry['message']}", icon="ℹ️")
        elif entry['level'] == 'success':
            st.success(f"**{entry['timestamp']}** - {entry['message']}", icon="✅")
        elif entry['level'] == 'warning':
            st.warning(f"**{entry['timestamp']}** - {entry['message']}", icon="⚠️")
        elif entry['level'] == 'error':
            st.error(f"**{entry['timestamp']}** - {entry['message']}", icon="❌")

