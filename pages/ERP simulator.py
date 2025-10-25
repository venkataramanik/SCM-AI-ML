import streamlit as st
import pandas as pd
import uuid
import numpy as np

# --- 1. CORE ERP DATA AND CONSTANTS ---

# Constants
LABOR_RATE_PER_CHAIR = 50.00
SALES_PRICE_PER_CHAIR = 250.00
# 1 FG-CHAIR requires: 5 Wood Planks and 20 Screws
BOM = {
    'FG-CHAIR': [
        {'material_id': 'RM-WOOD', 'quantity': 5},
        {'material_id': 'RM-SCREW', 'quantity': 20}
    ]
}

# Define the sequential steps
STEPS = [
    "Initialize System",
    "Sales Order Entry (Customer Demand)", # Updated step label
    "Run Production Planning (MRP)",
    "Execute Procurement (PO & Receipt)",
    "Manufacturing Execution (Consume RM & Produce FG)",
    "Shipping & Customer Invoice (Recognize Revenue & COGS)",
    "Generate Financial Report (P&L)"
]

# --- 2. STATE INITIALIZATION ---

def initialize_state():
    """Sets up or resets the initial state in Streamlit's session_state."""
    # This function now unconditionally sets all state variables, acting as a reliable reset.
    st.session_state.step = 0
    st.session_state.inventory = {
        'FG-CHAIR': {'type': 'Finished Goods', 'stock': 10, 'uom': 'EA', 'cost': 120.00},
        'RM-WOOD': {'type': 'Raw Material', 'stock': 100, 'uom': 'PLANK', 'cost': 10.00},
        'RM-SCREW': {'type': 'Raw Material', 'stock': 500, 'uom': 'UNIT', 'cost': 0.10}
    }
    st.session_state.ledger = pd.DataFrame(columns=['timestamp', 'type', 'amount', 'related_id', 'details'])
    st.session_state.sales_orders = []
    st.session_state.purchase_orders = []
    st.session_state.production_orders = []
    st.session_state.log = ["System Initialized. Click 'Next Step' to begin the ERP cycle."]
    st.session_state.main_so_id = None
    st.session_state.main_prod_order = None
    st.session_state.metrics = {'demand': 0, 'revenue': 0, 'profit': 0}
    
    # Check if initial_demand exists, otherwise set default (used for configuration)
    if 'initial_demand' not in st.session_state:
        st.session_state.initial_demand = 15

def log_transaction(entry_type, amount, related_id, details=""):
    """Logs financial activity to the ledger."""
    new_entry = pd.DataFrame([{
        'timestamp': pd.Timestamp.now().strftime("%H:%M:%S"),
        'type': entry_type,
        'amount': amount,
        'related_id': related_id,
        'details': details
    }])
    st.session_state.ledger = pd.concat([st.session_state.ledger, new_entry], ignore_index=True)
    st.session_state.log.append(f"[FINANCE] Entry: {entry_type} | ${amount:,.2f} | {details}")

def log_message(message, module="INFO"):
    """Adds a status message to the audit log."""
    st.session_state.log.append(f"[{module}] {message}")

# --- 3. ERP LOGIC FUNCTIONS (Adapted for Session State) ---

def create_sales_order(quantity):
    """Creates a new customer order."""
    order_id = f"SO-{uuid.uuid4().hex[:6]}"
    order = {
        'id': order_id, 
        'product': 'FG-CHAIR', 
        'quantity': quantity,
        'status': 'DEMAND_CREATED',
    }
    st.session_state.sales_orders.append(order)
    st.session_state.main_so_id = order_id
    st.session_state.metrics['demand'] = quantity
    log_message(f"Sales Order {order_id} created for {quantity} Chairs.", "SALES")

def run_mrp():
    """Calculates production and purchase requirements."""
    fg_stock = st.session_state.inventory['FG-CHAIR']['stock']
    total_demand = st.session_state.metrics['demand']
    production_required = max(0, total_demand - fg_stock)
    
    log_message(f"Demand: {total_demand}. Stock: {fg_stock}. Production Required: {production_required}.", "MRP")
    
    if production_required > 0:
        prod_id = f"PROD-{uuid.uuid4().hex[:6]}"
        prod_order = {
            'id': prod_id, 
            'product': 'FG-CHAIR', 
            'quantity': production_required,
            'status': 'PLANNED',
            'materials_needed': {}
        }
        
        # Determine Raw Material Requirements
        for component in BOM['FG-CHAIR']:
            mat_id = component['material_id']
            qty_per_unit = component['quantity']
            total_needed = production_required * qty_per_unit
            stock = st.session_state.inventory[mat_id]['stock']
            purchase_needed = max(0, total_needed - stock)
            
            prod_order['materials_needed'][mat_id] = total_needed
            
            if purchase_needed > 0:
                create_purchase_order(mat_id, purchase_needed, prod_id)
            
        st.session_state.production_orders.append(prod_order)
        st.session_state.main_prod_order = prod_order
    else:
        log_message("Demand met by stock. Skipping production and procurement.", "MRP")

def create_purchase_order(material_id, quantity, related_prod_id):
    """Generates a Purchase Order."""
    po_id = f"MM-{uuid.uuid4().hex[:6]}"
    unit_cost = st.session_state.inventory[material_id]['cost']
    total_cost = quantity * unit_cost
    po = {
        'id': po_id, 
        'material': material_id, 
        'quantity': quantity, 
        'total_cost': total_cost,
        'status': 'SENT_TO_SUPPLIER',
        'related_prod_id': related_prod_id
    }
    st.session_state.purchase_orders.append(po)
    log_message(f"PO {po_id} created for {quantity} x {material_id}.", "PROCUREMENT")

def execute_procurement():
    """Simulates receiving goods and incurring expense."""
    pending_pos = [po for po in st.session_state.purchase_orders if po['status'] == 'SENT_TO_SUPPLIER']
    
    if not pending_pos:
        log_message("No Purchase Orders were pending for receipt.", "PROCUREMENT")
        return

    for po in pending_pos:
        # Update Inventory
        st.session_state.inventory[po['material']]['stock'] += po['quantity']
        po['status'] = 'RECEIVED'
        
        # Log Expense (Supplier Invoice Payment)
        log_transaction('EXPENSE', po['total_cost'], po['id'], f"Payment for {po['material']} procurement.")
        log_message(f"Goods Received (PO {po['id']}). Inventory updated.", "PROCUREMENT")

def execute_production():
    """Consumes RM, incurs labor, and creates FG."""
    prod_order = st.session_state.main_prod_order
    if not prod_order or prod_order['status'] != 'PLANNED':
        return log_message("No production order to execute.", "MFG")

    product_quantity = prod_order['quantity']
    total_rm_cost = 0.0
    
    log_message(f"Starting production of {product_quantity} chairs...", "MFG")

    # 1. Consume Raw Materials
    for mat_id, qty_needed in prod_order['materials_needed'].items():
        cost_per_unit = st.session_state.inventory[mat_id]['cost']
        cost_consumed = qty_needed * cost_per_unit
        total_rm_cost += cost_consumed
        
        st.session_state.inventory[mat_id]['stock'] -= qty_needed
        log_transaction('COGS_RM', cost_consumed, prod_order['id'], f"Consumed {qty_needed} of {mat_id}.")

    # 2. Incur Labor Cost
    total_labor_cost = product_quantity * LABOR_RATE_PER_CHAIR
    log_transaction('COGS_LABOR', total_labor_cost, prod_order['id'], "Incurred labor cost (WIP to FG).")

    # 3. Create Finished Goods (FG)
    st.session_state.inventory['FG-CHAIR']['stock'] += product_quantity
    prod_order['status'] = 'COMPLETED'
    
    total_production_cost = total_rm_cost + total_labor_cost
    cost_per_chair = total_production_cost / product_quantity
    st.session_state.inventory['FG-CHAIR']['cost'] = cost_per_chair # Update FG cost
    
    log_message(f"Production completed. {product_quantity} Chairs added to stock. New Unit Cost: ${cost_per_chair:,.2f}", "MFG")

def ship_order():
    """Ships the finished goods, recognizes revenue and COGS."""
    so_id = st.session_state.main_so_id
    so = next((s for s in st.session_state.sales_orders if s['id'] == so_id), None)
    
    if not so:
        return log_message("Sales Order not found.", "SHIPPING")

    quantity = so['quantity']
    if st.session_state.inventory[so['product']]['stock'] < quantity:
        return log_message(f"CRITICAL: Not enough stock to ship SO {so_id}.", "ERROR")
    
    # 1. Update Inventory and COGS
    st.session_state.inventory[so['product']]['stock'] -= quantity
    unit_cost = st.session_state.inventory[so['product']]['cost']
    cogs = quantity * unit_cost
    
    # 2. Recognize Revenue
    revenue = quantity * SALES_PRICE_PER_CHAIR
    
    # 3. Log Financials
    log_transaction('REVENUE', revenue, so_id, "Sale of Finished Goods (Customer Invoice).")
    log_transaction('COGS_FG', cogs, so_id, "Cost of Goods Sold (Matching Revenue).")
    
    st.session_state.metrics['revenue'] = revenue
    so['status'] = 'SHIPPED_INVOICED'
    log_message(f"SO {so_id} shipped. Revenue: ${revenue:,.2f} | COGS: ${cogs:,.2f}", "SHIPPING")

def generate_report():
    """Calculates and displays the P&L statement."""
    if st.session_state.ledger.empty:
        log_message("Ledger is empty. No financial activity recorded.", "FINANCE")
        return

    # Calculate P&L metrics
    df = st.session_state.ledger
    total_revenue = df[df['type'] == 'REVENUE']['amount'].sum()
    total_cogs = df[df['type'].str.contains('COGS', na=False)]['amount'].sum()
    total_expense = df[df['type'] == 'EXPENSE']['amount'].sum()

    gross_profit = total_revenue - total_cogs
    net_profit = gross_profit - total_expense
    st.session_state.metrics['profit'] = net_profit
    
    # Display P&L Summary (using Streamlit's metric columns)
    st.subheader("Final Financial Report (P&L)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total COGS", f"-${total_cogs:,.2f}")
    col3.metric("Gross Profit", f"${gross_profit:,.2f}")
    col4.metric("Net Profit", f"${net_profit:,.2f}", delta_color=("inverse" if net_profit < 0 else "normal"))
    
    st.markdown("---")
    st.subheader("Detailed Ledger for Reconciliation")
    # Display full ledger
    st.dataframe(st.session_state.ledger, use_container_width=True, hide_index=True)


# --- 4. STEPPER LOGIC ---

def run_step():
    """Executes the logic for the current step and increments the counter."""
    current_step = st.session_state.step
    
    # Execute logic based on the current step number
    if current_step == 1:
        # Now uses the configured demand quantity
        create_sales_order(quantity=st.session_state.initial_demand)
    elif current_step == 2:
        run_mrp()
    elif current_step == 3:
        execute_procurement()
    elif current_step == 4:
        execute_production()
    elif current_step == 5:
        ship_order()
    elif current_step == 6:
        generate_report()
    
    # Only increment if the simulation is not finished
    if current_step < len(STEPS) - 1:
        st.session_state.step += 1
    elif current_step == len(STEPS) - 1:
        log_message("ERP Simulation Cycle Complete.", "INFO")
        st.session_state.step += 1 # Mark as finished

# --- 5. STREAMLIT UI LAYOUT ---

def main():
    st.set_page_config(layout="wide", page_title="ERP Simulator")
    st.title("Custom Chair Manufacturing ERP Simulator")
    st.markdown("Execute the full **Order-to-Cash** and **Procure-to-Pay** cycle step-by-step.")

    # Initialize state (or reset if the button was clicked)
    initialize_state()

    # --- CONFIGURATION (Only visible at Step 0) ---
    if st.session_state.step == 0:
        st.subheader("Simulation Configuration")
        # The sales order quantity is now user-defined here
        st.session_state.initial_demand = st.number_input(
            "Initial Sales Order (SO) Demand Quantity (Chairs)",
            min_value=1,
            max_value=100,
            value=st.session_state.initial_demand,
            step=5,
            help="This is the customer order that triggers the entire ERP cycle."
        )
        st.info(f"The simulation will start with an initial inventory of **10 FG-CHAIRS**.")

    # --- CONTROL AND STATUS BAR ---
    col1, col2 = st.columns([3, 1])
    
    current_step_index = st.session_state.step
    is_finished = current_step_index >= len(STEPS)
    
    if not is_finished:
        current_step_label = STEPS[current_step_index]
    else:
        current_step_label = "Simulation Complete"
        
    col1.subheader(f"Current Process Step: {current_step_label}")
    
    button_label = "Run " + STEPS[current_step_index+1] if current_step_index < len(STEPS) - 1 else "Generate Final Report"
    
    if current_step_index == 0:
        button_label = "Start Simulation (Run Step 1: SO Entry)"
    elif is_finished:
        button_label = "Reset Simulation"
        col1.info("The full ERP cycle has been executed. Click 'Reset Simulation' to start over.")

    if col2.button(button_label, use_container_width=True, type="primary"):
        # The logic here is now correct because initialize_state() fully resets all session variables
        if is_finished:
            # Reruns the entire script, calling initialize_state() which now resets everything
            initialize_state()
        else:
            # Run the next step
            run_step()
            
    # --- METRICS DISPLAY ---
    st.markdown("---")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Demand", f"{st.session_state.metrics['demand']} Chairs")
    colB.metric("Chairs in Stock", f"{st.session_state.inventory['FG-CHAIR']['stock']} EA")
    colC.metric("Total Revenue", f"${st.session_state.metrics['revenue']:,.2f}")
    colD.metric("Net Profit", f"${st.session_state.metrics['profit']:,.2f}", 
                delta_color=("inverse" if st.session_state.metrics['profit'] < 0 else "normal"))

    st.markdown("---")

    # --- DATA DISPLAY AREA ---
    col_inv, col_log = st.columns([1, 2])

    # 1. Inventory Status
    with col_inv:
        st.subheader("Inventory Management (WMS)")
        # Convert inventory dict to DataFrame for display
        inventory_df = pd.DataFrame(st.session_state.inventory).T
        inventory_df['Unit Cost'] = inventory_df['cost'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(inventory_df[['type', 'stock', 'uom', 'Unit Cost']], 
                     use_container_width=True, 
                     hide_index=False,
                     column_config={
                         "type": "Type",
                         "stock": "Stock",
                         "uom": "UOM",
                     })

    # 2. Audit Log
    with col_log:
        st.subheader("Transaction & Audit Log")
        # Display log in reverse order (most recent on top)
        st.text_area("Real-time Console", 
                     "\n".join(st.session_state.log[::-1]), 
                     height=300, 
                     disabled=True)

    # 3. Financial Report (Only show when finished)
    if is_finished:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_report()

if __name__ == '__main__':
    main()
