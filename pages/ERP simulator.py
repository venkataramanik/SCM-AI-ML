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
    'FG-CHAIR': {
        'routing': [
            {'step': 'Cut Wood Planks', 'time': '2 hours', 'cost_driver': LABOR_RATE_PER_CHAIR * 0.4},
            {'step': 'Assemble Chair', 'time': '1 hour', 'cost_driver': LABOR_RATE_PER_CHAIR * 0.2},
            {'step': 'Quality Check & Finish', 'time': '1.5 hours', 'cost_driver': LABOR_RATE_PER_CHAIR * 0.4}
        ],
        'materials': [
            {'material_id': 'RM-WOOD', 'quantity': 5},
            {'material_id': 'RM-SCREW', 'quantity': 20}
        ]
    }
}

# Define the sequential steps
STEPS = [
    "Configuration & Sales Order Entry",        # Index 0: Create SO
    "Run Production Planning (MRP & Work Order)", # Index 1: Run MRP, create PO/Prod Order plans
    "Execute Purchase Order (PO) Creation",     # Index 2: Formalize POs/Commitment
    "Execute Goods Receipt (RM Stock Update)",  # Index 3: Receive goods, update RM inventory, record Expense
    "Manufacturing Execution (WIP & FG Production)", # Index 4: Consume RM (WIP), incur Labor, produce FG
    "Shipping & Customer Invoice (Recognize Revenue & COGS)", # Index 5: Ship, recognize Revenue/COGS
    "Generate Financial Report (P&L)"          # Index 6: Report
]

# --- 2. STATE INITIALIZATION ---

def initialize_state():
    """Sets up or resets the initial state in Streamlit's session_state."""
    st.session_state.step = 0
    st.session_state.inventory = {
        'FG-CHAIR': {'type': 'Finished Goods', 'stock': 10, 'uom': 'EA', 'cost': 120.00},
        'RM-WOOD': {'type': 'Raw Material', 'stock': 100, 'uom': 'PLANK', 'cost': 10.00},
        'RM-SCREW': {'type': 'Raw Material', 'stock': 500, 'uom': 'UNIT', 'cost': 0.10}
    }
    st.session_state.wip = 0.0 # New: Work In Process cost accumulator
    st.session_state.ledger = pd.DataFrame(columns=['timestamp', 'type', 'amount', 'related_id', 'details'])
    st.session_state.sales_orders = []
    st.session_state.purchase_orders = []
    st.session_state.production_orders = []
    st.session_state.log = ["System Initialized. Configure the demand and click the first step to begin the ERP cycle."]
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
    """Calculates production and purchase requirements and creates Prod/Work Order."""
    fg_stock = st.session_state.inventory['FG-CHAIR']['stock']
    total_demand = st.session_state.metrics['demand']
    production_required = max(0, total_demand - fg_stock)
    
    # Reset pending POs created in previous runs if any
    st.session_state.purchase_orders = []

    log_message(f"Demand: {total_demand}. Stock: {fg_stock}. Production Required: {production_required}.", "MRP")
    
    if production_required > 0:
        prod_id = f"PROD-{uuid.uuid4().hex[:6]}"
        prod_order = {
            'id': prod_id, 
            'product': 'FG-CHAIR', 
            'quantity': production_required,
            'status': 'PLANNED',
            'materials_needed': {},
            'routing': BOM['FG-CHAIR']['routing'] # Attach full routing details
        }
        
        # Determine Raw Material Requirements and create *internal* POs for next step
        for component in BOM['FG-CHAIR']['materials']:
            mat_id = component['material_id']
            qty_per_unit = component['quantity']
            total_needed = production_required * qty_per_unit
            stock = st.session_state.inventory[mat_id]['stock']
            purchase_needed = max(0, total_needed - stock)
            
            prod_order['materials_needed'][mat_id] = total_needed
            
            if purchase_needed > 0:
                create_purchase_order(mat_id, purchase_needed, prod_id) 
            
        st.session_state.production_orders = [prod_order] # Overwrite, usually only one main order in this sim
        st.session_state.main_prod_order = prod_order
    else:
        st.session_state.main_prod_order = None # Ensure Prod order is cleared if demand met
        log_message("Demand met by stock. Skipping production and procurement.", "MRP")

def create_purchase_order(material_id, quantity, related_prod_id):
    """Generates a Purchase Order (called by MRP)."""
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

def execute_po_creation():
    """Confirms POs created in MRP and sets them up for receipt."""
    if not st.session_state.purchase_orders:
        log_message("No Purchase Orders were necessary (Raw Materials are sufficient).", "PROCUREMENT")
        return

    log_message(f"{len(st.session_state.purchase_orders)} Purchase Orders created and sent to suppliers.", "PROCUREMENT")
    for po in st.session_state.purchase_orders:
        log_message(f"PO {po['id']} formally issued for {po['quantity']} x {po['material']}. Commitment: ${po['total_cost']:,.2f}", "PROCUREMENT")

def execute_goods_receipt():
    """Simulates receiving goods and incurring expense, updating RM stock."""
    pending_pos = [po for po in st.session_state.purchase_orders if po['status'] == 'SENT_TO_SUPPLIER']
    
    if not pending_pos:
        log_message("No Purchase Orders pending receipt. Skipping Goods Receipt.", "PROCUREMENT")
        return

    for po in pending_pos:
        # 1. Update Inventory (RM Update)
        st.session_state.inventory[po['material']]['stock'] += po['quantity']
        po['status'] = 'RECEIVED'
        
        # 2. Log Expense (Supplier Invoice Payment)
        log_transaction('EXPENSE', po['total_cost'], po['id'], f"Payment for {po['material']} procurement.")
        log_message(f"Goods Receipt (PO {po['id']}). RM Inventory updated by {po['quantity']} {po['material']}.", "PROCUREMENT")

def execute_production():
    """Consumes RM, incurs labor, and creates FG. Uses WIP accounting."""
    prod_order = st.session_state.main_prod_order
    if not prod_order or prod_order['status'] != 'PLANNED':
        return log_message("No production order to execute or production already started.", "MFG")

    product_quantity = prod_order['quantity']
    total_rm_cost = 0.0
    total_labor_cost = product_quantity * LABOR_RATE_PER_CHAIR
    
    log_message(f"Starting production of {product_quantity} chairs. Transferring costs to **WIP**...", "MFG")

    # 1. Consume Raw Materials & Move Cost to WIP
    for mat_id, qty_needed in prod_order['materials_needed'].items():
        cost_per_unit = st.session_state.inventory[mat_id]['cost']
        cost_consumed = qty_needed * cost_per_unit
        total_rm_cost += cost_consumed
        
        st.session_state.inventory[mat_id]['stock'] -= qty_needed
        # Log to WIP (Debit WIP, Credit RM Inventory)
        log_transaction('WIP_RM', cost_consumed, prod_order['id'], f"Consumed {qty_needed} of {mat_id}, transferred cost to WIP.")
    
    # Update WIP accumulator (RM cost)
    st.session_state.wip += total_rm_cost

    # 2. Incur Labor Cost & Move Cost to WIP
    # Log to WIP (Debit WIP, Credit Labor Expense/Cash)
    log_transaction('WIP_LABOR', total_labor_cost, prod_order['id'], "Incurred labor cost, transferred cost to WIP.")
    st.session_state.wip += total_labor_cost
    
    # 3. Create Finished Goods (FG) & Clear WIP
    st.session_state.inventory['FG-CHAIR']['stock'] += product_quantity
    prod_order['status'] = 'COMPLETED'
    
    total_production_cost = st.session_state.wip
    
    # Log WIP completion (Debit FG Inventory, Credit WIP)
    log_transaction('FG_COST', total_production_cost, prod_order['id'], "Total WIP cost capitalized to Finished Goods Inventory.")
    
    # Calculate new FG cost
    cost_per_chair = total_production_cost / product_quantity if product_quantity > 0 else 0
    st.session_state.inventory['FG-CHAIR']['cost'] = cost_per_chair 
    
    # Reset WIP
    st.session_state.wip = 0.0
    
    log_message(f"Production completed. {product_quantity} Chairs added to stock. New Unit Cost: ${cost_per_chair:,.2f}.", "MFG")

def ship_order():
    """Ships the finished goods, recognizes revenue and COGS."""
    so_id = st.session_state.main_so_id
    so = next((s for s in st.session_state.sales_orders if s['id'] == so_id), None)
    
    if not so:
        return log_message("Sales Order not found.", "SHIPPING")

    quantity = so['quantity']
    
    # Check current stock vs required quantity
    if st.session_state.inventory[so['product']]['stock'] < quantity:
        return log_message(f"CRITICAL: Not enough stock ({st.session_state.inventory[so['product']]['stock']}) to ship SO {so_id} for {quantity} units.", "ERROR")
    
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
    # COGS is only derived from the FG that was sold
    total_cogs = df[df['type'] == 'COGS_FG']['amount'].sum() 
    # Total Expense is RM purchase cost (simplified)
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
    
    if current_step == 0:
        create_sales_order(quantity=st.session_state.initial_demand)
    elif current_step == 1:
        run_mrp()
        # Check if MRP found demand met by stock: skip to Shipping (Step 5)
        if st.session_state.main_prod_order is None:
            st.session_state.step = 5 
            log_message("Demand met by stock. Skipping Procurement and Production steps.", "ERP")
            return # Prevent standard increment below
    elif current_step == 2:
        # Execute PO Creation. If no POs, jump to Manufacturing (Step 4)
        if not st.session_state.purchase_orders and st.session_state.main_prod_order:
             st.session_state.step = 4 
             log_message("Sufficient Raw Materials on hand. Skipping PO Creation and Goods Receipt steps.", "ERP")
             return # Prevent standard increment below
        execute_po_creation() 
    elif current_step == 3:
        execute_goods_receipt()
    elif current_step == 4:
        execute_production()
    elif current_step == 5:
        ship_order()
    elif current_step == 6:
        generate_report()
        
    # Standard increment, only if no skip occurred
    if st.session_state.step == current_step and current_step < len(STEPS) - 1:
        st.session_state.step += 1


# --- 5. STREAMLIT UI LAYOUT ---

def main():
    st.set_page_config(layout="wide", page_title="ERP Simulator")
    st.title("Custom Chair Manufacturing ERP Simulator")
    st.markdown("Execute the full **Order-to-Cash** and **Procure-to-Pay** cycle step-by-step. Note the detailed material, manufacturing, and financial impact at each stage.")

    # Initialize state
    # FIX: Check for both 'step' and 'metrics' to ensure robust initialization
    if 'step' not in st.session_state or 'metrics' not in st.session_state:
        initialize_state()
    
    # --- CONFIGURATION (Only visible at Step 0) ---
    if st.session_state.step == 0:
        with st.container():
            st.subheader("Simulation Configuration")
            st.session_state.initial_demand = st.number_input(
                "Initial Sales Order (SO) Demand Quantity (Chairs)",
                min_value=1,
                max_value=100,
                value=st.session_state.initial_demand,
                step=5,
                key='demand_input',
                help="This customer order quantity triggers the entire cycle. Current FG stock: 10."
            )
            st.info(f"The simulation will start with an initial FG stock of **10 Chairs**.")

    # --- STEPPER BUTTONS & RESET ---
    current_step_index = st.session_state.step
    is_finished = current_step_index >= len(STEPS) # Check if we are past the last step

    st.subheader("ERP Process Flow (Click the Active Step to Execute)")
    
    if is_finished:
        st.success("The full ERP cycle has been executed. Click 'Reset Simulation' to start over.")
        if st.button("Reset Simulation", use_container_width=True, type="primary"):
            initialize_state()
    else:
        cols = st.columns(len(STEPS))

        for i, step_name in enumerate(STEPS):
            with cols[i]:
                button_label = f"Step {i+1}:\n{step_name}"
                is_active_step = (i == current_step_index)
                
                st.button(
                    button_label,
                    key=f"step_btn_{i}",
                    disabled=not is_active_step,
                    type="primary" if is_active_step else "secondary",
                    on_click=run_step,
                    use_container_width=True
                )
    st.markdown("---")
            
    # --- METRICS DISPLAY ---
    colA, colB, colC, colD, colE = st.columns(5)
    # This line is now protected by the robust initialization check above
    colA.metric("Demand", f"{st.session_state.metrics['demand']} Chairs")
    colB.metric("FG Stock", f"{st.session_state.inventory['FG-CHAIR']['stock']} EA")
    colC.metric("Work In Process (WIP)", f"${st.session_state.wip:,.2f}")
    colD.metric("Total Revenue", f"${st.session_state.metrics['revenue']:,.2f}")
    colE.metric("Net Profit", f"${st.session_state.metrics['profit']:,.2f}", 
                delta_color=("inverse" if st.session_state.metrics['profit'] < 0 else "normal"))

    # --- PROCESS STATUS DETAILS (Sales Order, Prod Order, BOM, Routing) ---
    st.markdown("---")
    st.subheader("BOM, Routing, and Process Status")
    
    # 1. BOM Display
    st.markdown("#### 📜 Bill of Materials (BOM) for FG-CHAIR")
    bom_data = BOM['FG-CHAIR']['materials']
    bom_df = pd.DataFrame(bom_data).rename(columns={'material_id': 'Material ID', 'quantity': 'Quantity per Unit'})
    st.dataframe(bom_df, hide_index=True, use_container_width=True)

    # 2. Production/Work Order Details (Only visible if an order exists)
    if st.session_state.main_prod_order:
        prod = st.session_state.main_prod_order
        st.markdown(f"#### 🏭 Active Production Order: `{prod['id']}` (Qty: **{prod['quantity']}** | Status: **{prod['status']}**)")
        
        # Display Routing
        st.markdown("##### ⚙️ Production Routing (Work Order Steps)")
        routing_data = prod['routing']
        routing_df = pd.DataFrame(routing_data).rename(columns={'step': 'Step', 'time': 'Time Est.', 'cost_driver': 'Labor Cost Driver'})
        st.dataframe(routing_df, hide_index=True, use_container_width=True)
        
        # Display Material Requirements
        st.markdown("##### 📦 Material Requirements from MRP")
        mat_needed = [{"Material ID": k, "Needed": v, "In Stock": st.session_state.inventory.get(k, {}).get('stock', 0)} for k, v in prod['materials_needed'].items()]
        st.dataframe(mat_needed, hide_index=True, use_container_width=True)
        
        if prod['status'] == 'COMPLETED':
             st.success("Manufacturing is **Complete**. Finished Goods Inventory has been updated.")

    st.markdown("---")

    # --- DATA DISPLAY AREA ---
    col_inv, col_log = st.columns([1, 2])

    # 1. Inventory Status
    with col_inv:
        st.subheader("Inventory Management (RM & FG Stock)")
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
